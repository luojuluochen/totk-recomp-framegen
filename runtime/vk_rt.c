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
#include "vk.h"

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
    if (!c || !cfg_bool("TOTK_RT")) return;   // off: the log and the device are exactly as without RT
    const char *mode = cfg_get("TOTK_RT");
    if (!c->has_accel_struct || !c->has_ray_query) {
        rt_log("rt: TOTK_RT=%s, but this GPU/driver has no acceleration structures + ray query; off", mode ? mode : "");
        return;
    }
    if (!load_fns(c->device)) {
        rt_log("rt: the driver's acceleration-structure entry points are missing; off");
        return;
    }
    rt_up = 1;
    rt_log("rt: on (TOTK_RT=%s): VK_KHR_acceleration_structure + VK_KHR_ray_query loaded", mode ? mode : "1");
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

void vk_rt_geom_draw(const VkRtGeomDraw *d) {
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
