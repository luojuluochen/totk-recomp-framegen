// Ray tracing groundwork for the port's own passes (TOTK_RT). The game's own NVN commands never ask for it: the port
// builds its own acceleration structures from the draws it replays, and its own ray query passes (shadows, reflections)
// shade into the scene before DLSS. Off by default; vk_device.c enables VK_KHR_acceleration_structure and
// VK_KHR_ray_query only when TOTK_RT is set, and vk_rt_init runs once the device is up.
//
// This file starts with what the passes need first: the driver entry points, and a probe (TOTK_RT=probe) that builds a
// one-triangle bottom level structure and a one-instance top level structure on the GPU and frees them again, so the
// path is known to work before any scene geometry is captured.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"   // gfx_counters.frames, the census window
#include "platform/platform.h"
#include "rt_shadow_temporal_spv.h"
#include "rt_shadow_spv.h"
#include "vk.h"
#include "vk_exec.h"   // VkExec: the replay's command buffer and fence

typedef struct {
    PFN_vkCreateAccelerationStructureKHR create;
    PFN_vkDestroyAccelerationStructureKHR destroy;
    PFN_vkGetAccelerationStructureBuildSizesKHR sizes;
    PFN_vkGetAccelerationStructureDeviceAddressKHR address;
    PFN_vkCmdBuildAccelerationStructuresKHR build;
    PFN_vkGetBufferDeviceAddressKHR buffer_address;
} RtFns;

static RtFns F;
static int rt_up;   // TOTK_RT set, the extensions enabled and every entry point loaded

static int load_fns(VkDevice d) {
    F.create = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(d, "vkCreateAccelerationStructureKHR");
    F.destroy = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(d, "vkDestroyAccelerationStructureKHR");
    F.sizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(d, "vkGetAccelerationStructureBuildSizesKHR");
    F.address = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(d, "vkGetAccelerationStructureDeviceAddressKHR");
    F.build = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(d, "vkCmdBuildAccelerationStructuresKHR");
    F.buffer_address = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(d, "vkGetBufferDeviceAddressKHR");
    return F.create && F.destroy && F.sizes && F.address && F.build && F.buffer_address;
}

// A buffer for one of the structures: host ones are mapped for the caller to fill, device ones stay where the GPU
// builds. Every buffer here takes a device address (VK_KHR_buffer_device_address is on, vk_device.c).
static VkBuffer make_buf(VkDeviceSize size, VkBufferUsageFlags usage, int host, VmaAllocation *alloc, void **map) {
    VkCtx *c = vk_ctx();
    if (!c || !size) return VK_NULL_HANDLE;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
    VmaAllocationCreateInfo ai = {.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (host) ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info;
    VkBuffer b = VK_NULL_HANDLE;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &b, alloc, host ? &info : NULL) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (map) *map = host ? info.pMappedData : NULL;
    return b;
}

static void drop_buf(VkBuffer b, VmaAllocation a) {
    if (b) vmaDestroyBuffer(vk_ctx()->vma, b, a);
}

static VkDeviceAddress address_of(VkBuffer b) {
    VkBufferDeviceAddressInfo i = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b};
    return F.buffer_address(vk_ctx()->device, &i);
}

// One build: the storage buffer keeps the structure alive (returned through st/sta), the scratch buffer is freed here
// once the batch that records the build has completed.
static int build_one(int top, VkAccelerationStructureGeometryKHR *geom, uint32_t prim, VkBuffer *st, VmaAllocation *sta,
                     VkAccelerationStructureKHR *as, VkDeviceSize *bytes) {
    VkCtx *c = vk_ctx();
    VkAccelerationStructureBuildGeometryInfoKHR bi = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = top ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = geom;
    VkAccelerationStructureBuildSizesInfoKHR sz = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    F.sizes(c->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prim, &sz);
    *st = make_buf(sz.accelerationStructureSize,
                   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                   0, sta, NULL);
    if (!*st) { rt_log("rt: %s storage allocation failed (%llu B)", top ? "TLAS" : "BLAS",
                       (unsigned long long)sz.accelerationStructureSize); return 0; }
    VkAccelerationStructureCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                               .buffer = *st,
                                               .offset = 0,
                                               .size = sz.accelerationStructureSize,
                                               .type = bi.type};
    if (F.create(c->device, &ci, NULL, as) != VK_SUCCESS) {
        rt_log("rt: %s creation failed", top ? "TLAS" : "BLAS");
        drop_buf(*st, *sta);
        *st = VK_NULL_HANDLE;
        return 0;
    }
    VmaAllocation sca = NULL;
    VkBuffer sc = make_buf(sz.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           0, &sca, NULL);
    if (!sc) {
        rt_log("rt: %s scratch allocation failed (%llu B)", top ? "TLAS" : "BLAS",
               (unsigned long long)sz.buildScratchSize);
        F.destroy(c->device, *as, NULL);
        *as = VK_NULL_HANDLE;
        drop_buf(*st, *sta);
        *st = VK_NULL_HANDLE;
        return 0;
    }
    bi.dstAccelerationStructure = *as;
    bi.scratchData.deviceAddress = address_of(sc);
    VkAccelerationStructureBuildRangeInfoKHR range = {.primitiveCount = prim};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
    VkCommandBuffer cb = vk_upload_begin();
    F.build(cb, 1, &bi, &ranges);
    vk_upload_end();
    vk_upload_flush(1);   // the build is done when this returns
    drop_buf(sc, sca);
    *bytes = sz.accelerationStructureSize;
    return 1;
}

// TOTK_RT=probe: one triangle, one instance. Nothing reads the result; the point is that the whole path (extensions,
// device addresses, build sizes, the build itself on the graphics queue) ran on this driver.
static void probe(void) {
    VkCtx *c = vk_ctx();
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE, tlas = VK_NULL_HANDLE;
    VkBuffer vb = VK_NULL_HANDLE, blas_st = VK_NULL_HANDLE, inst = VK_NULL_HANDLE, tlas_st = VK_NULL_HANDLE;
    VmaAllocation va = NULL, blas_sta = NULL, insta = NULL, tlas_sta = NULL;
    void *vmap = NULL, *imap = NULL;
    VkDeviceSize blas_bytes = 0, tlas_bytes = 0;
    int ok = 0;

    float verts[9] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    vb = make_buf(sizeof verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                  1, &va, &vmap);
    if (!vb || !vmap) { rt_log("rt: probe: vertex buffer allocation failed"); goto out; }
    memcpy(vmap, verts, sizeof verts);
    vmaFlushAllocation(c->vma, va, 0, VK_WHOLE_SIZE);

    VkAccelerationStructureGeometryKHR geo = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                                              .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                                              .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};
    geo.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geo.geometry.triangles.vertexData.deviceAddress = address_of(vb);
    geo.geometry.triangles.vertexStride = 3 * sizeof(float);
    geo.geometry.triangles.maxVertex = 2;
    geo.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    if (!build_one(0, &geo, 1, &blas_st, &blas_sta, &blas, &blas_bytes)) goto out;
    VkDeviceAddress blas_addr = F.address(c->device, &(VkAccelerationStructureDeviceAddressInfoKHR){
                                                          .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                                                          .accelerationStructure = blas});
    if (!blas_addr) { rt_log("rt: probe: BLAS has no device address"); goto out; }

    VkAccelerationStructureInstanceKHR inst_v = {0};
    inst_v.transform.matrix[0][0] = inst_v.transform.matrix[1][1] = inst_v.transform.matrix[2][2] = 1.0f;
    inst_v.mask = 0xFF;
    inst_v.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst_v.accelerationStructureReference = blas_addr;
    inst = make_buf(sizeof inst_v, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                    1, &insta, &imap);
    if (!inst || !imap) { rt_log("rt: probe: instance buffer allocation failed"); goto out; }
    memcpy(imap, &inst_v, sizeof inst_v);
    vmaFlushAllocation(c->vma, insta, 0, VK_WHOLE_SIZE);

    VkAccelerationStructureGeometryKHR tgeo = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                                               .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                                               .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};
    tgeo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tgeo.geometry.instances.arrayOfPointers = VK_FALSE;
    tgeo.geometry.instances.data.deviceAddress = address_of(inst);
    if (!build_one(1, &tgeo, 1, &tlas_st, &tlas_sta, &tlas, &tlas_bytes)) goto out;
    VkDeviceAddress tlas_addr = F.address(c->device, &(VkAccelerationStructureDeviceAddressInfoKHR){
                                                          .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                                                          .accelerationStructure = tlas});
    if (!tlas_addr) { rt_log("rt: probe: TLAS has no device address"); goto out; }
    rt_log("rt: probe: one-triangle BLAS %llu B, one-instance TLAS %llu B, addresses %#llx/%#llx: built",
           (unsigned long long)blas_bytes, (unsigned long long)tlas_bytes, (unsigned long long)blas_addr,
           (unsigned long long)tlas_addr);
    ok = 1;

out:
    if (tlas) F.destroy(c->device, tlas, NULL);
    if (blas) F.destroy(c->device, blas, NULL);
    drop_buf(tlas_st, tlas_sta);
    drop_buf(inst, insta);
    drop_buf(blas_st, blas_sta);
    drop_buf(vb, va);
    if (!ok) rt_log("rt: probe: failed");
}

void vk_rt_init(void) {
    VkCtx *c = vk_ctx();
    // TOTK_RT is not a settings-table row (the environment or totk.ini only): cfg_get, not cfg_value, so a run
    // without it does not stop in the settings lookup.
    const char *mode = cfg_get("TOTK_RT");
    if (!c || !mode || !*mode || !strcmp(mode, "0")) return;   // off: the log and the device are exactly as without RT
    if (!c->has_accel_struct || !c->has_ray_query) {
        rt_log("rt: TOTK_RT=%s, but this GPU/driver has no acceleration structures + ray query; off", mode);
        return;
    }
    if (!load_fns(c->device)) {
        rt_log("rt: the driver's acceleration-structure entry points are missing; off");
        return;
    }
    rt_up = 1;
    rt_log("rt: on (TOTK_RT=%s): VK_KHR_acceleration_structure + VK_KHR_ray_query loaded", mode);
    if (!strcmp(mode, "probe")) probe();
}

int vk_rt_ready(void) { return rt_up; }

// ---- geometry census (TOTK_RT=geom)
// Counts what the scene's indexed draws offer a BVH: how many, how many triangles, which vertex-shader banks they read
// (an object's own place c7 or bones c5 against terrain that has neither), the position attribute's NVN format, offset
// and stream, and whether the draw's own vertex and index buffers carry device addresses. Nothing here touches the GPU.
static int geom_on = -1;

int vk_rt_geom_on(void) {
    if (geom_on < 0) {
        VkCtx *c = vk_ctx();
        const char *v = cfg_get("TOTK_RT");
        geom_on = c && c->has_accel_struct && v && (!strcmp(v, "geom") || !strcmp(v, "1"));
    }
    return geom_on;
}

static struct {
    uint64_t draws, tris, skinned, place, neither, buildable;
    uint64_t itype[3];
    uint64_t afmt[256], aoff[256], astr[256];
    uint64_t last_frame;
} G;

static struct { VkBuffer b; VkDeviceAddress a; int ok; } ac[1024];

static VkDeviceAddress buf_addr(VkBuffer b) {
    if (!b) return 0;
    unsigned h = (unsigned)((((uintptr_t)b >> 4) * 2654435761u) >> 10) & 1023;
    if (ac[h].b == b) return ac[h].ok ? ac[h].a : 0;
    VkBufferDeviceAddressInfo i = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b};
    VkDeviceAddress a = F.buffer_address(vk_ctx()->device, &i);
    ac[h].b = b, ac[h].a = a, ac[h].ok = a != 0;
    return a;
}

static void rt_collect(const VkRtGeomDraw *d);   // the scene BVH's per-draw instance (below)

void vk_rt_geom_draw(const VkRtGeomDraw *d) {
    if (vk_rt_shadows_on()) rt_collect(d);
    if (!geom_on || !d->scene) return;
    uint32_t fmt = d->attr[0] & 0xffu, off = (d->attr[0] >> 8) & 0x7ffffu, str = (d->attr[0] >> 27) & 0x1fu;
    VkBuffer pb = str < 16 ? d->vb[str] : VK_NULL_HANDLE;
    G.draws++;
    G.tris += d->count / 3;
    G.itype[d->itype & 3]++;
    G.afmt[fmt]++, G.aoff[fmt] = off, G.astr[fmt] = str;
    if (d->c5) G.skinned++;
    if (d->c7) G.place++;
    if (!d->c5 && !d->c7) G.neither++;
    if (buf_addr(pb) && buf_addr(d->ib)) G.buildable++;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (!G.last_frame) G.last_frame = frame;
    if (frame - G.last_frame < 300) return;
    uint64_t n = G.draws;
    char pos[200] = "";
    for (int k = 0; k < 3; k++) {   // the top three position formats, they are consumed
        int best = -1;
        for (int f = 0; f < 256; f++)
            if (G.afmt[f] && (best < 0 || G.afmt[f] > G.afmt[best])) best = f;
        if (best < 0) break;
        char t[72];
        snprintf(t, sizeof t, "%s0x%02x %.0f%% (offset %llu, stream %llu)", k ? ", " : "", best,
                 100.0 * (double)G.afmt[best] / (double)n, (unsigned long long)G.aoff[best],
                 (unsigned long long)G.astr[best]);
        strncat(pos, t, sizeof pos - strlen(pos) - 1);
        G.afmt[best] = 0;
    }
    rt_log("rt: geom: %llu frames: %llu scene draws, %.1f M tris; c5 skinned %.1f%%, c7 place %.1f%%, neither %.1f%%; "
           "vertex+index buffers with device addresses %.1f%%; index u8 %llu u16 %llu u32 %llu; position attr %s",
           (unsigned long long)(frame - G.last_frame), (unsigned long long)n, (double)G.tris / 1e6,
           100.0 * (double)G.skinned / (double)n, 100.0 * (double)G.place / (double)n, 100.0 * (double)G.neither / (double)n,
           100.0 * (double)G.buildable / (double)n, (unsigned long long)G.itype[0], (unsigned long long)G.itype[1],
           (unsigned long long)G.itype[2], pos);
    memset(&G, 0, sizeof G);
    G.last_frame = frame;
}

// ---- scene BVH (TOTK_RT=bvh): a BLAS per unique piece of geometry, a TLAS per frame -------------
//
// The draws are collected while the replay records the scene (vk_rt_geom_draw): outside a pass they only note a
// geometry key and the object's place matrix. At the scene's composite - the point the contact shadows already pick
// (vk_contact.c) - the job's command buffer is not inside a pass, so that is where the new bottom level structures and
// the top level structure are recorded. A key names the draw's own vertex and index data: the game's buffers carry
// device addresses (vk_memory.c pool_usage) and are never copied.
typedef struct {
    VkDeviceAddress vb, ib;      // vertex data (attribute offset and base vertex folded in) and index data
    uint32_t stride, count;      // vertex stride, index count
    uint32_t vkfmt;              // VkFormat of the position attribute
    uint32_t itype;              // 1: u16, 2: u32
    uint32_t maxv;               // highest vertex index the range can hold
} RtKey;

typedef struct {
    RtKey k;
    VkAccelerationStructureKHR as;
    VkDeviceAddress addr;
    VkBuffer st;
    VmaAllocation sta;
} RtBlas;

#define RT_BLAS_CAP 8192
#define RT_HASH_CAP 16384
#define RT_INST_CAP 65536
#define RT_TLAS_SETS 2
#define RT_BUILDS_PER_JOB 256

static RtBlas rt_bl[RT_BLAS_CAP];
static int32_t rt_head[RT_HASH_CAP];   // -1 empty, else the first index of the chain
static int32_t rt_next[RT_BLAS_CAP];
static uint32_t rt_nbl;
static int rt_tables;

typedef struct { RtKey k; VkTransformMatrixKHR m; } RtInst;   // one scene draw: its geometry and its place
static RtInst *rt_inst;
static uint32_t rt_ninst, rt_cinst;

typedef struct {
    VkBuffer inst, st, sc;
    VmaAllocation inst_a, st_a, sc_a;
    VkAccelerationStructureKHR tlas;
    void *inst_map;
    VkDeviceAddress sc_addr;
    VkDeviceSize bytes;
} RtTlasSet;
static RtTlasSet rt_set[RT_TLAS_SETS];
static int rt_set_i;
static int rt_bvh_ready;
static uint32_t rt_scratch_align = 256;
static uint64_t rt_stat_inst, rt_stat_builds, rt_stat_skip_inst, rt_stat_skip_blas, rt_stat_last;
static PlatRMutex rt_lock = PLAT_RMUTEX_INIT;
static int bvh_on = -1;

int vk_rt_shadows_on(void) {
    if (bvh_on < 0) {
        VkCtx *c = vk_ctx();
        const char *v = cfg_get("TOTK_RT");
        bvh_on = c && rt_up && c->has_accel_struct && c->has_ray_query && v &&
                 (!strcmp(v, "bvh") || !strcmp(v, "1") || !strcmp(v, "shadow"));
        if (bvh_on) {
            VkPhysicalDeviceAccelerationStructurePropertiesKHR p = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
            VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p};
            vkGetPhysicalDeviceProperties2(c->phys, &p2);
            if (p.minAccelerationStructureScratchOffsetAlignment > 1)
                rt_scratch_align = p.minAccelerationStructureScratchOffsetAlignment;
            rt_log("rt: scene BVH on: up to %d BLASes, %d instances a frame; scratch align %u", RT_BLAS_CAP, RT_INST_CAP,
                   rt_scratch_align);
        }
    }
    return bvh_on;
}

// What a vertex shader reads: the object's bones (c5) and place (c7), from its resource list.
static struct { const GfxShader *s; uint8_t c5, c7; } rt_smemo[4096];

static void rt_shader_banks(const GfxShader *s, int *c5, int *c7) {
    uint32_t h = (uint32_t)((((uintptr_t)s >> 4) * 2654435761u) >> 20) & 4095;
    if (rt_smemo[h].s != s) {
        const GfxShaderRes *ur = NULL;
        int n = gfx_shader_resources(s, GFX_RES_UBO, &ur), a = 0, b = 0;
        for (int i = 0; i < n; i++) a |= ur[i].slot == 5, b |= ur[i].slot == 7;
        rt_smemo[h].s = s, rt_smemo[h].c5 = (uint8_t)a, rt_smemo[h].c7 = (uint8_t)b;
    }
    *c5 = rt_smemo[h].c5, *c7 = rt_smemo[h].c7;
}

// Components of an NVN vertex format (vk_pipeline.c vtx_formats): a position has three or four.
static int nvn_attr_comps(uint32_t f) {
    if (f >= 0x25 && f <= 0x30) return 4;            // RGBA, A2B10G10R10, R32G32B32A32
    if (f >= 0x19 && f <= 0x24) return 3;            // RGB
    if (f >= 0x0D && f <= 0x18) return 2;            // RG
    if (f >= 0x01 && f <= 0x0C) return 1;            // R
    if (f >= 0x5D && f <= 0x62) return 3;
    if (f >= 0x63 && f <= 0x6C) return 4;
    if (f == 0x3D || f == 0x3E) return 4;
    if (f == 0x3F) return 3;
    if (f == 0x9A) return 4;
    if (f >= 0x51 && f <= 0x54) return 1;
    if (f >= 0x57 && f <= 0x5C) return 2;
    return 0;
}

// One scene draw: a plain triangle list whose vertex shader takes the object's own place from c7 (not a skinned pose
// from c5) contributes its geometry and that place to the frame's TLAS.
static void rt_collect(const VkRtGeomDraw *d) {
    if (!d->scene || d->prim != 0x4 /* NVN_DRAW_PRIMITIVE_TRIANGLES */ || d->count < 3) return;
    if (!d->vs || !d->c7 || !d->c7_addr) return;
    int c5 = 0, c7 = 0;
    rt_shader_banks(d->vs, &c5, &c7);
    if (!c7 || c5) return;
    int ai = -1;
    VkFormat vf = VK_FORMAT_UNDEFINED;
    for (int i = 0; i < 16; i++) {
        uint32_t a = d->attr[i], fmt = a & 0xffu;
        if (!fmt || nvn_attr_comps(fmt) < 3) continue;
        int type = 0;
        VkFormat f = vk_vertex_format((int)fmt, &type);
        if (f == VK_FORMAT_UNDEFINED || type != GFX_TYPE_FLOAT) continue;
        ai = i, vf = f;
        break;
    }
    if (ai < 0) { rt_stat_skip_inst++; return; }
    uint32_t a = d->attr[ai], off = (a >> 8) & 0x7ffffu, str = (a >> 27) & 0x1fu;
    if (str >= 16 || d->itype < 1 || d->itype > 2) { rt_stat_skip_inst++; return; }
    VkBuffer vb = d->vb[str];
    uint32_t stride = (uint32_t)d->vb_stride[str];
    if (!vb || stride < 4) { rt_stat_skip_inst++; return; }
    VkDeviceAddress vbase = buf_addr(vb) + d->vb_off[str] + off + (int64_t)d->base_vertex * stride;
    VkDeviceAddress ibase = buf_addr(d->ib) + d->ib_off;
    if (!vbase || !ibase || (vbase & 3) || (ibase & (d->itype == 1 ? 1u : 3u))) { rt_stat_skip_inst++; return; }
    VkDeviceSize avail = d->vb_size[str] > (VkDeviceSize)off ? d->vb_size[str] - off : 0;
    uint32_t maxv = avail > stride ? (uint32_t)(avail / stride) - 1 : 0;
    if (maxv > 0xFFFFFFu) maxv = 0xFFFFFFu;
    RtKey k = {.vb = vbase, .ib = ibase, .stride = stride, .count = (uint32_t)d->count, .vkfmt = (uint32_t)vf,
               .itype = (uint32_t)d->itype, .maxv = maxv};
    // The TLAS is built in view space: the object's place (world) through the scene camera's view matrix (the block's
    // first 3x4, vk_dlaa.c note_camera). The depth the shader reads reconstructs view-space positions, so the ray query
    // needs no inverse.
    VkTransformMatrixKHR m;
    float mo[12];
    // As of the submit, not the replay: the game may already hold the next frame's blocks there (sync_ahead), and a
    // matrix read one frame ahead of the depth made every shadow edge jump from frame to frame (vk_dlaa.c does the
    // same for the motion vectors).
    const float *ms = (const float *)vk_usnap_bytes(d->c7_addr, sizeof mo);
    memcpy(mo, ms ? ms : (const void *)(uintptr_t)d->c7_addr, sizeof mo);
    if (d->cam_addr) {
        const float *v = (const float *)vk_usnap_bytes(d->cam_addr, 256);
        if (!v) v = (const float *)(uintptr_t)d->cam_addr;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                m.matrix[i][j] = v[i * 4 + 0] * mo[j] + v[i * 4 + 1] * mo[4 + j] + v[i * 4 + 2] * mo[8 + j];
            m.matrix[i][3] = v[i * 4 + 0] * mo[3] + v[i * 4 + 1] * mo[7] + v[i * 4 + 2] * mo[11] + v[i * 4 + 3];
        }
    } else {
        memcpy(&m, mo, sizeof m);
    }
    plat_rmutex_lock(&rt_lock);
    if (rt_ninst == rt_cinst) {
        uint32_t nc = rt_cinst ? rt_cinst * 2 : 256;
        RtInst *ni = realloc(rt_inst, (size_t)nc * sizeof *ni);
        if (!ni) { plat_rmutex_unlock(&rt_lock); return; }
        rt_inst = ni, rt_cinst = nc;
    }
    rt_inst[rt_ninst].k = k, rt_inst[rt_ninst].m = m;
    rt_ninst++;
    plat_rmutex_unlock(&rt_lock);
}

static uint32_t rt_key_hash(const RtKey *k) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t *p = (const uint8_t *)k;
    for (size_t i = 0; i < sizeof *k; i++) h = (h ^ p[i]) * 1099511628211ull;
    return (uint32_t)(h % RT_HASH_CAP);
}

static int32_t rt_bl_find(const RtKey *k) {   // lock held
    if (!rt_tables) {
        for (uint32_t i = 0; i < RT_HASH_CAP; i++) rt_head[i] = -1;
        rt_tables = 1;
    }
    for (int32_t i = rt_head[rt_key_hash(k)]; i >= 0; i = rt_next[i])
        if (!memcmp(&rt_bl[i].k, k, sizeof *k)) return i;
    return -1;
}

static int32_t rt_bl_add(const RtKey *k, VkCommandBuffer cb, uint64_t fence) {   // lock held, outside a pass
    if (rt_nbl >= RT_BLAS_CAP) return -1;
    VkCtx *c = vk_ctx();
    VkAccelerationStructureGeometryKHR g = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                                            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                                            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};
    g.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    g.geometry.triangles.vertexFormat = (VkFormat)k->vkfmt;
    g.geometry.triangles.vertexData.deviceAddress = k->vb;
    g.geometry.triangles.vertexStride = k->stride;
    g.geometry.triangles.maxVertex = k->maxv;
    g.geometry.triangles.indexType = k->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    g.geometry.triangles.indexData.deviceAddress = k->ib;
    VkAccelerationStructureBuildGeometryInfoKHR bi = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = &g;
    uint32_t prim = k->count / 3;
    VkAccelerationStructureBuildSizesInfoKHR sz = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    F.sizes(c->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prim, &sz);
    RtBlas *b = &rt_bl[rt_nbl];
    b->k = *k;
    b->st = make_buf(sz.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     0, &b->sta, NULL);
    if (!b->st) return -1;
    VkAccelerationStructureCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                               .buffer = b->st,
                                               .offset = 0,
                                               .size = sz.accelerationStructureSize,
                                               .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
    if (F.create(c->device, &ci, NULL, &b->as) != VK_SUCCESS) {
        vmaDestroyBuffer(c->vma, b->st, b->sta);
        return -1;
    }
    VmaAllocation sca = NULL;
    VkBuffer sc = make_buf(sz.buildScratchSize + rt_scratch_align,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0, &sca, NULL);
    if (!sc) {
        F.destroy(c->device, b->as, NULL);
        vmaDestroyBuffer(c->vma, b->st, b->sta);
        return -1;
    }
    VkDeviceAddress sa = address_of(sc);
    sa = (sa + rt_scratch_align - 1) & ~(VkDeviceAddress)(rt_scratch_align - 1);
    bi.dstAccelerationStructure = b->as;
    bi.scratchData.deviceAddress = sa;
    VkAccelerationStructureBuildRangeInfoKHR range = {.primitiveCount = prim};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
    F.build(cb, 1, &bi, &ranges);
    vk_defer_buffer_until(sc, sca, fence);
    b->addr = F.address(c->device, &(VkAccelerationStructureDeviceAddressInfoKHR){
                                         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                                         .accelerationStructure = b->as});
    uint32_t h = rt_key_hash(k);
    rt_next[rt_nbl] = rt_head[h];
    rt_head[h] = (int32_t)rt_nbl;
    rt_nbl++;
    rt_stat_builds++;
    return (int32_t)(rt_nbl - 1);
}

static int rt_tlas_set_init(RtTlasSet *s) {
    VkCtx *c = vk_ctx();
    s->inst = make_buf((VkDeviceSize)RT_INST_CAP * sizeof(VkAccelerationStructureInstanceKHR),
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                       1, &s->inst_a, &s->inst_map);
    if (!s->inst || !s->inst_map) return 0;
    VkAccelerationStructureGeometryKHR g = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                                            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                                            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};
    g.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    g.geometry.instances.arrayOfPointers = VK_FALSE;
    g.geometry.instances.data.deviceAddress = address_of(s->inst);
    VkAccelerationStructureBuildGeometryInfoKHR bi = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = &g;
    uint32_t prim = RT_INST_CAP;
    VkAccelerationStructureBuildSizesInfoKHR sz = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    F.sizes(c->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prim, &sz);
    s->st = make_buf(sz.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     0, &s->st_a, NULL);
    if (!s->st) return 0;
    VkAccelerationStructureCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                               .buffer = s->st,
                                               .offset = 0,
                                               .size = sz.accelerationStructureSize,
                                               .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
    if (F.create(c->device, &ci, NULL, &s->tlas) != VK_SUCCESS) return 0;
    s->sc = make_buf(sz.buildScratchSize + rt_scratch_align,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0, &s->sc_a, NULL);
    if (!s->sc) return 0;
    s->sc_addr = address_of(s->sc);
    s->sc_addr = (s->sc_addr + rt_scratch_align - 1) & ~(VkDeviceAddress)(rt_scratch_align - 1);
    s->bytes = sz.accelerationStructureSize;
    return 1;
}

// ---- the ray-query shadow pass (runtime/shaders/rt_shadow.comp) ---------------------------------
static struct {
    int tried, ok;
    int strength, debug;
    uint64_t n;   // dispatches so far, for the stats line
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler samp;
    PFN_vkCmdPushDescriptorSetKHR push;
    // the occupancy mask rt_shadow.comp writes and rt_shadow_soft.comp softens (and the softening pass' own pipeline)
    VkImage mask;
    VmaAllocation mask_a;
    VkImageView mask_v;
    uint32_t mask_w, mask_h;
    VkDescriptorSetLayout soft_dsl;
    VkPipelineLayout soft_lay;
    VkPipeline soft_pipe;
    // the temporal pass at the DLAA point: last frame's smoothed occupancy and the camera's motion between them
    VkImage hist[2];
    VmaAllocation hist_a[2];
    VkImageView hist_v[2];
    int hist_cur;
    uint32_t hist_w, hist_h;
    float prev_vp[16];
    int have_prev;
    uint32_t hist_frames;
    PFN_vkCmdPushDescriptorSetKHR push_t;
} S;

static int rt_shadow_init(void) {
    if (S.tried) return S.ok;
    S.tried = 1;
    VkCtx *c = vk_ctx();
    const char *sv = cfg_get("TOTK_RT_SHADOW");
    // Full strength turns a shadowed pixel black: the composite's colour holds ambient light too, and only the direct
    // sun may go. 55% reads like the game's own shadows; TOTK_RT_SHADOW=0..100 sets it.
    S.strength = sv && *sv ? atoi(sv) : 55;
    if (S.strength <= 0) {
        rt_log("rt: shadows off (TOTK_RT_SHADOW=0)");
        return 0;
    }
    if (S.strength > 100) S.strength = 100;
    const char *dv = cfg_get("TOTK_RT_SHADOW_DEBUG");
    S.debug = dv && *dv && strcmp(dv, "0") ? atoi(dv) : 0;
    S.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[5] = {
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 5, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &S.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = sizeof rt_shadow_spv, .pCode = rt_shadow_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!S.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &S.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &S.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &S.samp) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) {
        rt_log("rt: shadows off: their pipeline could not be created");
        return 0;
    }
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = S.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &S.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) {
        rt_log("rt: shadows off: the compute pipeline failed (%d)", (int)r);
        return 0;
    }
    // the temporal pass (rt_shadow_temporal.comp): the mask blurred and reprojected into the history, then onto the
    // colour at the DLAA point, where the camera matrices live
    VkDescriptorSetLayoutBinding sb[4] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo sdci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 4, .pBindings = sb};
    VkPushConstantRange tpr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 96};
    VkPipelineLayoutCreateInfo sli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                      .pSetLayouts = &S.soft_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &tpr};
    VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = sizeof rt_shadow_temporal_spv, .pCode = rt_shadow_temporal_spv};
    VkShaderModule sm = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(c->device, &sdci, NULL, &S.soft_dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &sli, NULL, &S.soft_lay) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &smi, NULL, &sm) != VK_SUCCESS) {
        rt_log("rt: shadows off: the temporal pass could not be created");
        return 0;
    }
    VkComputePipelineCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                       .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"},
                                       .layout = S.soft_lay};
    VkResult sr = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &sci, NULL, &S.soft_pipe);
    vkDestroyShaderModule(c->device, sm, NULL);
    if (sr != VK_SUCCESS) {
        rt_log("rt: shadows off: the temporal pipeline failed (%d)", (int)sr);
        return 0;
    }
    rt_log("rt: shadows on: rays a pixel in a small sun cone, strength %d%%%s", S.strength,
           S.debug ? " (debug view: the shadow factor in grey)" : "");
    S.ok = 1;
    return 1;
}

static void rt_shadow_record(struct VkExec *ex, const struct VkExecJob *j, VkAccelerationStructureKHR tlas) {
    if (!rt_shadow_init()) return;
    VkImageView in = vk_exec_target_view(j->depth.img, VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D,
                                         VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView out = vk_exec_target_view(j->scene.img, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_VIEW_TYPE_2D,
                                          VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    VkImageView sun = j->sun.img ? vk_exec_target_view(j->sun.img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D,
                                                       VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0)
                                 : VK_NULL_HANDLE;
    // the occupancy mask rt_shadow.comp writes (R16F, the scene's size)
    if (!S.mask || S.mask_w != j->scene.width || S.mask_h != j->scene.height) {
        VkCtx *mc = vk_ctx();
        if (S.mask_v) vkDestroyImageView(mc->device, S.mask_v, NULL), S.mask_v = VK_NULL_HANDLE;
        if (S.mask) vmaDestroyImage(mc->vma, S.mask, S.mask_a), S.mask = VK_NULL_HANDLE;
        VkImageCreateInfo mci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                 .imageType = VK_IMAGE_TYPE_2D,
                                 .format = VK_FORMAT_R16G16_SFLOAT,
                                 .extent = {j->scene.width, j->scene.height, 1},
                                 .mipLevels = 1,
                                 .arrayLayers = 1,
                                 .samples = VK_SAMPLE_COUNT_1_BIT,
                                 .tiling = VK_IMAGE_TILING_OPTIMAL,
                                 .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                 .initialLayout = VK_IMAGE_LAYOUT_GENERAL};
        VmaAllocationCreateInfo mai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(mc->vma, &mci, &mai, &S.mask, &S.mask_a, NULL) != VK_SUCCESS) return;
        VkImageViewCreateInfo mvi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                     .image = S.mask,
                                     .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                     .format = VK_FORMAT_R16G16_SFLOAT,
                                     .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(mc->device, &mvi, NULL, &S.mask_v) != VK_SUCCESS) return;
        S.mask_w = j->scene.width, S.mask_h = j->scene.height;
    }
    static int logged;
    if (!logged) {
        logged = 1;
        rt_log("rt: shadow dispatch: scene %ux%u, depth view %d, scene view %d, sun view %d; proj %.3f %.3f far %.1f, "
               "sun %.3f %.3f %.3f", j->scene.width, j->scene.height, in != VK_NULL_HANDLE, out != VK_NULL_HANDLE,
               sun != VK_NULL_HANDLE, j->pc[0], j->pc[1], j->pc[2], j->pc[4], j->pc[5], j->pc[6]);
    }
    if (!in || !out) return;
    vk_exec_barrier(ex);   // the G-buffer images in GENERAL with their writes visible (as vk_contact.c)
    VkCommandBuffer cb = ex->cb;
    float pcv[16];
    memcpy(pcv, j->pc, sizeof pcv);
    pcv[3] = (float)S.strength / 100.0f;
    pcv[12] = (float)S.debug;
    const char *iv = cfg_get("TOTK_RT_SHADOW_IGNORE_VIS");   // test: darken even where the game's shadow maps shadow
    pcv[13] = sun && !(iv && *iv && strcmp(iv, "0") != 0) ? 1.0f : 0.0f;
    // the cone the rays take around the sun (radians) and how many rays a pixel: 4 softens the edge but costs 4x
    // on the 65k-instance BVH, which is the frame's most expensive pass here
    static float cone = 0.006f, rays = 2.0f;
    static int opts_read;
    if (!opts_read) {
        opts_read = 1;
        const char *cv = cfg_get("TOTK_RT_SHADOW_CONE");
        if (cv && *cv) cone = (float)atof(cv);
        const char *rv = cfg_get("TOTK_RT_SHADOW_RAYS");
        if (rv && *rv) rays = (float)atoi(rv);
    }
    pcv[14] = cone, pcv[15] = rays;
    VkDescriptorImageInfo ii = {S.samp, in, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo si = {S.samp, sun ? sun : in, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSetAccelerationStructureKHR asw = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &tlas};
    VkDescriptorImageInfo mi = {VK_NULL_HANDLE, S.mask_v, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[5] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, .pNext = &asw},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 3, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &si},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 4, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &mi}};
    if (S.debug) {   // the debug views write the colour themselves: no mask, no softening
        pcv[3] = 1.0f;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.pipe);
    S.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.layout, 0, 5, w);
    vkCmdPushConstants(cb, S.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pcv, pcv);
    vkCmdDispatch(cb, (j->scene.width + 7) / 8, (j->scene.height + 7) / 8, 1);
    S.n++;
    vk_texture_storage_written(j->scene.img);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    ex->recorded += 6;
    vk_exec_draw_new_cb(ex);
}

// At the scene's composite (the contact shadows' point, vk_exec.c run_job): create the BLASes the draws asked for
// since the last one, then record the frame's TLAS build into the job's command buffer. Two sets alternate, since the
// replay may run a frame ahead of the GPU: a set is not written while the other is read.
void vk_rt_shadow_exec(struct VkExec *ex, const struct VkExecJob *j) {
    if (!vk_rt_shadows_on() || !ex || !ex->cb) return;
    plat_rmutex_lock(&rt_lock);
    uint32_t n = rt_ninst;
    if (!n) {
        plat_rmutex_unlock(&rt_lock);
        return;
    }
    if (!rt_bvh_ready) {
        int ok = 1;
        for (int i = 0; i < RT_TLAS_SETS; i++) ok &= rt_tlas_set_init(&rt_set[i]);
        if (!ok) {
            rt_log("rt: scene BVH off: the TLAS buffers could not be created");
            bvh_on = 0;
            plat_rmutex_unlock(&rt_lock);
            return;
        }
        rt_bvh_ready = 1;
        rt_log("rt: scene BVH: a TLAS set is %llu KB, %d sets; instance buffer %llu KB a set",
               (unsigned long long)(rt_set[0].bytes / 1024), RT_TLAS_SETS,
               (unsigned long long)((VkDeviceSize)RT_INST_CAP * sizeof(VkAccelerationStructureInstanceKHR) / 1024));
    }
    RtTlasSet *s = &rt_set[rt_set_i];
    VkAccelerationStructureInstanceKHR *dst = s->inst_map;
    uint32_t m = 0, nbl = 0;
    for (uint32_t i = 0; i < n && m < RT_INST_CAP; i++) {
        int32_t idx = rt_bl_find(&rt_inst[i].k);
        if (idx < 0) {
            if (nbl >= RT_BUILDS_PER_JOB) { rt_stat_skip_blas++; continue; }
            idx = rt_bl_add(&rt_inst[i].k, ex->cb, ex->fence);
            if (idx < 0) { rt_stat_skip_blas++; continue; }
            nbl++;
        }
        VkAccelerationStructureInstanceKHR *o = &dst[m++];
        memset(o, 0, sizeof *o);
        o->transform = rt_inst[i].m;
        o->mask = 0xFF;
        o->flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        o->accelerationStructureReference = rt_bl[idx].addr;
    }
    if (!m) {
        rt_ninst = 0;
        plat_rmutex_unlock(&rt_lock);
        return;
    }
    vmaFlushAllocation(vk_ctx()->vma, s->inst_a, 0, (VkDeviceSize)m * sizeof *dst);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkAccelerationStructureGeometryKHR g = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                                            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                                            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};
    g.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    g.geometry.instances.arrayOfPointers = VK_FALSE;
    g.geometry.instances.data.deviceAddress = address_of(s->inst);
    VkAccelerationStructureBuildGeometryInfoKHR bi = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.dstAccelerationStructure = s->tlas;
    bi.geometryCount = 1;
    bi.pGeometries = &g;
    bi.scratchData.deviceAddress = s->sc_addr;
    VkAccelerationStructureBuildRangeInfoKHR range = {.primitiveCount = m};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
    F.build(ex->cb, 1, &bi, &ranges);
    // The shadow pass reads what was just built: AS write -> compute shader read.
    VkMemoryBarrier2 ab = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                           .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR};
    VkDependencyInfo ad = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &ab};
    vkCmdPipelineBarrier2(ex->cb, &ad);
    rt_shadow_record(ex, j, s->tlas);
    rt_stat_inst += m;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (!rt_stat_last) rt_stat_last = frame;
    if (frame - rt_stat_last >= 300) {
        rt_log("rt: bvh: %llu frames: %u BLASes (%llu new), %llu instances (%.0f a frame), %.0f KB a TLAS, %llu shadow "
               "dispatches; skipped %llu instances, %llu over the %d-build cap",
               (unsigned long long)(frame - rt_stat_last), rt_nbl, (unsigned long long)rt_stat_builds,
               (unsigned long long)rt_stat_inst, (double)rt_stat_inst / (double)(frame - rt_stat_last),
               (double)s->bytes / 1024.0, (unsigned long long)S.n, (unsigned long long)rt_stat_skip_inst,
               (unsigned long long)rt_stat_skip_blas, RT_BUILDS_PER_JOB);
        rt_stat_inst = rt_stat_builds = rt_stat_skip_inst = rt_stat_skip_blas = 0;
        S.n = 0;
        rt_stat_last = frame;
    }
    rt_set_i = (rt_set_i + 1) % RT_TLAS_SETS;
    rt_ninst = 0;
    plat_rmutex_unlock(&rt_lock);
}



// Views of the game's images for the temporal pass. A view destroyed right after recording is used by the GPU a frame
// later, which faults there: keep them, keyed by image and format, and let go only at shutdown.
static struct { VkImage img; VkFormat fmt; VkImageView view; } *TV;
static int TV_n, TV_cap;

static VkImageView temp_view(VkCtx *c, VkImage img, VkFormat fmt) {
    for (int i = 0; i < TV_n; i++)
        if (TV[i].img == img && TV[i].fmt == fmt) return TV[i].view;
    if (TV_n == TV_cap) {
        int nc = TV_cap ? TV_cap * 2 : 8;
        void *np = realloc(TV, (size_t)nc * sizeof *TV);
        if (!np) return VK_NULL_HANDLE;
        TV = np, TV_cap = nc;
    }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = fmt;
    vi.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(c->device, &vi, NULL, &v) != VK_SUCCESS) return VK_NULL_HANDLE;
    TV[TV_n].img = img, TV[TV_n].fmt = fmt, TV[TV_n].view = v, TV_n++;
    return v;
}

static void temp_views_drop(VkCtx *c) {
    for (int i = 0; i < TV_n; i++) vkDestroyImageView(c->device, TV[i].view, NULL);
    TV_n = 0, TV_cap = 0;
    free(TV), TV = NULL;
}

// ---- the ray shadows' temporal half (rt_shadow_temporal.comp), at the DLAA point where the camera matrices are ------
static void rows_to_col(const float *rows, int nrows, float *m) {
    memset(m, 0, 16 * sizeof *m);
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < 4; c++) m[c * 4 + r] = rows[r * 4 + c];
    if (nrows == 3) m[15] = 1.0f;
}

static int inv4m(const float *m, float *o) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) a[r][c] = m[c * 4 + r], a[r][4 + c] = r == c ? 1.0 : 0.0;
    for (int c = 0; c < 4; c++) {
        int pv = c;
        for (int r = c + 1; r < 4; r++)
            if (fabs(a[r][c]) > fabs(a[pv][c])) pv = r;
        if (fabs(a[pv][c]) < 1e-12) return 0;
        if (pv != c)
            for (int k = 0; k < 8; k++) {
                double t = a[c][k];
                a[c][k] = a[pv][k], a[pv][k] = t;
            }
        double d = a[c][c];
        for (int k = 0; k < 8; k++) a[c][k] /= d;
        for (int r = 0; r < 4; r++) {
            if (r == c) continue;
            double f = a[r][c];
            if (!f) continue;
            for (int k = 0; k < 8; k++) a[r][k] -= f * a[c][k];
        }
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) o[c * 4 + r] = (float)a[r][4 + c];
    return 1;
}

static void mul4m(const float *a, const float *b, float *o) {
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            double t = 0.0;
            for (int k = 0; k < 4; k++) t += (double)a[k * 4 + r] * b[c * 4 + k];
            o[c * 4 + r] = (float)t;
        }
}

static int hist_make(VkCtx *c, uint32_t w, uint32_t h) {
    for (int i = 0; i < 2; i++) {
        if (S.hist_v[i]) vkDestroyImageView(c->device, S.hist_v[i], NULL), S.hist_v[i] = VK_NULL_HANDLE;
        if (S.hist[i]) vmaDestroyImage(c->vma, S.hist[i], S.hist_a[i]), S.hist[i] = VK_NULL_HANDLE;
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R16G16_SFLOAT,
                                .extent = {w, h, 1},
                                .mipLevels = 1,
                                .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                .initialLayout = VK_IMAGE_LAYOUT_GENERAL};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &S.hist[i], &S.hist_a[i], NULL) != VK_SUCCESS) return 0;
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                    .image = S.hist[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                    .format = VK_FORMAT_R16G16_SFLOAT,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &S.hist_v[i]) != VK_SUCCESS) return 0;
    }
    S.hist_w = w, S.hist_h = h, S.hist_cur = 0, S.have_prev = 0, S.hist_frames = 0;
    return 1;
}

void vk_rt_shadow_apply(void *cb_, uint64_t scene, uint32_t w, uint32_t h, const float *cam) {
    if (!vk_rt_shadows_on() || !S.ok || S.debug || !cb_ || !scene || !cam || !w || !h) return;
    if (!S.mask || S.mask_w != w || S.mask_h != h) return;   // no mask this frame
    VkCtx *c = vk_ctx();
    VkCommandBuffer cb = (VkCommandBuffer)cb_;
    if (S.hist_w != w || S.hist_h != h || !S.hist[0])
        if (!hist_make(c, w, h)) return;
    VkImageView sv = temp_view(c, (VkImage)scene, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    if (!sv) return;
    float view[16], vp[16], invv[16], m[16];
    rows_to_col(cam, 3, view);
    rows_to_col(cam + 12, 4, vp);
    if (!inv4m(view, invv)) return;
    if (!S.have_prev) {
        memcpy(S.prev_vp, vp, sizeof vp);
        S.have_prev = 1;
    }
    mul4m(S.prev_vp, invv, m);
    VkDescriptorImageInfo mi = {S.samp, S.mask_v, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo hi = {S.samp, S.hist_v[S.hist_cur], VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo ho = {VK_NULL_HANDLE, S.hist_v[S.hist_cur ^ 1], VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo si = {VK_NULL_HANDLE, sv, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet ws[4] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &mi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &hi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ho},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 3, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &si}};
    float px = cam[7 * 4 + 0] > 0 ? cam[7 * 4 + 0] : 1.0f;
    float py = cam[8 * 4 + 1] > 0 ? cam[8 * 4 + 1] : 1.0f;
    static float hist_blend = -1.0f;   // TOTK_RT_SHADOW_HIST (0..0.99): the history's share, 0.85 by default
    if (hist_blend < 0) {
        const char *hv = cfg_get("TOTK_RT_SHADOW_HIST");
        hist_blend = hv && *hv ? (float)atof(hv) : 0.85f;
        if (hist_blend < 0) hist_blend = 0;
        if (hist_blend > 0.99f) hist_blend = 0.99f;
    }
    float blend = S.hist_frames ? hist_blend : 0.0f;
    float p4[24] = {px, py, (float)S.strength / 100.0f, blend, (float)w, (float)h, 1.0f / (float)w, 1.0f / (float)h,
                    m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                    m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]};
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.soft_pipe);
    S.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.soft_lay, 0, 4, ws);
    vkCmdPushConstants(cb, S.soft_lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p4, p4);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    vkCmdPipelineBarrier2(cb, &di);
    memcpy(S.prev_vp, vp, sizeof vp);
    S.hist_cur ^= 1;
    S.hist_frames++;
}

void vk_rt_shutdown(void) {
    VkCtx *c = vk_ctx();
    if (!c || !rt_up) return;
    plat_rmutex_lock(&rt_lock);
    for (int i = 0; i < RT_TLAS_SETS; i++) {
        RtTlasSet *s = &rt_set[i];
        if (s->tlas) F.destroy(c->device, s->tlas, NULL);
        if (s->st) vmaDestroyBuffer(c->vma, s->st, s->st_a);
        if (s->sc) vmaDestroyBuffer(c->vma, s->sc, s->sc_a);
        if (s->inst) vmaDestroyBuffer(c->vma, s->inst, s->inst_a);
        memset(s, 0, sizeof *s);
    }
    for (uint32_t i = 0; i < rt_nbl; i++) {
        if (rt_bl[i].as) F.destroy(c->device, rt_bl[i].as, NULL);
        if (rt_bl[i].st) vmaDestroyBuffer(c->vma, rt_bl[i].st, rt_bl[i].sta);
    }
    rt_nbl = 0, rt_tables = 0, rt_bvh_ready = 0, rt_set_i = 0;
    free(rt_inst);
    rt_inst = NULL, rt_ninst = rt_cinst = 0;
    temp_views_drop(c);
    plat_rmutex_unlock(&rt_lock);
}
