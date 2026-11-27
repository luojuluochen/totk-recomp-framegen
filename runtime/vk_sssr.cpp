// AMD FidelityFX Stochastic Screen Space Reflections (third_party/FidelityFX/sdk, MIT) on the port's Vulkan device,
// through the same SDK backend FSR 3 uses. The SDK's classifier, hierarchical-Z marching, blue-noise sampling and its
// temporal reflection denoiser do the work; this file feeds it the scene and puts its result back.
//
// Two points of the frame:
// - vk_sssr_prepare, at the scene's composite (the contact shadows' point): the view-space normal and roughness from the
//   port's own linear depth (sssr_prep.comp). That image is what the port's own compute passes already sample; the
//   hardware D32 depth is not (sampling it from a compute pass faulted the GPU here).
// - vk_sssr_exec, right before DLSS/FSR 3 (vk_dlaa.c evaluate): the normal turned to world space (sssr_conv.comp, which
//   SSSR's shaders expect), the SDK's dispatch over the scene colour, D32 depth and motion vectors, then the reflection
//   radiance onto the scene (sssr_apply.comp).
#include "vk_sssr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_denoiser.h>
#include <FidelityFX/host/ffx_sssr.h>
#include <ffx_shader_blobs.h>

extern "C" {
#include "cfg.h"
#include "host.h"
#include "vk.h"
}

#include "sssr_apply_spv.h"   // tools/build_shaders.py
#include "sssr_conv_spv.h"
#include "sssr_mv_spv.h"
#include "sssr_prep_spv.h"

// The SDK's shaders: tools/ffx_shaders.py --effects sssr (spirv/ffx_sssr_vk.h)
#include "ffx_sssr_vk.h"

enum { SSSR_PERM_DEPTH_INVERTED = 1 << 2, DEN_PERM_DEPTH_INVERTED = 1 << 2 };

extern "C" int vk_sssr_blob(uint32_t effect, uint32_t pass, uint32_t options, void *out) {
    if (effect == FFX_EFFECT_SSSR && pass < FFX_SSSR_PASS_COUNT) {
        if (options & SSSR_PERM_DEPTH_INVERTED)
            rt_log("sssr: inverted-depth permutation asked (%#x): the non-inverted shader serves", options);
        memcpy(out, &ffx_sssr_blobs[pass], sizeof(FfxShaderBlob));
        return 1;
    }
    if (effect == FFX_EFFECT_DENOISER && pass < FFX_DENOISER_PASS_COUNT) {
        if (!ffx_den_blobs[pass].data) return 0;   // the shadow passes: this context denoises reflections only
        memcpy(out, &ffx_den_blobs[pass], sizeof(FfxShaderBlob));
        return 1;
    }
    return 0;
}

static struct {
    bool tried, ok, failed;
    int strength, debug;
    FfxSssrContext ctx;
    FfxInterface iface;
    void *scratch;
    uint32_t w, h;
    VkImage nv, nw, mat, out, mv, env, lut;   // view-space normal, world-space normal, roughness, radiance, env cube, BRDF LUT
    VmaAllocation nv_a, nw_a, mat_a, out_a, mv_a, env_a, lut_a;
    VkImageView nv_v, nw_v, mat_v, out_v, mv_v, env_v, lut_v;
    VkSampler samp, splin;                // nearest (prep's depth), linear (SSSR's result, conv's normal)
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout prep_dsl, conv_dsl, mv_dsl, apply_dsl;
    VkPipelineLayout prep_lay, conv_lay, mv_lay, apply_lay;
    VkPipeline prep_pipe, conv_pipe, mv_pipe, apply_pipe;
    float prev_vp[16];
    bool have_prev;
    uint32_t prep_w, prep_h;              // the size prepare ran at this frame
    bool prepared;
    uint64_t ring[FFX_MAX_QUEUED_FRAMES];
    uint32_t n;
    uint64_t waits, frames;
} S;

static void *scratch_alloc(size_t n) {
#ifdef _WIN32
    void *p = _aligned_malloc(n, 256);
#else
    void *p = aligned_alloc(256, n);
#endif
    if (p) memset(p, 0, n);
    return p;
}

static void scratch_free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static void message(FfxMsgType type, const wchar_t *msg) {
    static int n;
    if (n++ >= 24) return;
    char buf[512];
    snprintf(buf, sizeof buf, "%ls", msg);
    rt_log("sssr: SDK %s: %s", type == FFX_MESSAGE_TYPE_ERROR ? "error" : "warning", buf);
}

static VKAPI_ATTR void VKAPI_CALL no_label(VkCommandBuffer, const VkDebugUtilsLabelEXT *) {}
static VKAPI_ATTR void VKAPI_CALL no_label_end(VkCommandBuffer) {}
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice d, const char *name) {
    PFN_vkVoidFunction f = vkGetDeviceProcAddr(d, name);
    if (f) return f;
    if (!strcmp(name, "vkGetBufferMemoryRequirements2KHR")) return vkGetDeviceProcAddr(d, "vkGetBufferMemoryRequirements2");
    if (!strcmp(name, "vkCmdBeginDebugUtilsLabelEXT")) return (PFN_vkVoidFunction)no_label;
    if (!strcmp(name, "vkCmdEndDebugUtilsLabelEXT")) return (PFN_vkVoidFunction)no_label_end;
    return nullptr;
}

// ---- 4x4 matrices (column-major floats; the camera block is row-major 4-wide rows) -----------------------------
static void rows_to_col(const float *rows, int nrows, float *m) {
    memset(m, 0, 16 * sizeof *m);
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < 4; c++) m[c * 4 + r] = rows[r * 4 + c];
    if (nrows == 3) m[15] = 1.0f;
}

static bool inv4(const float *m, float *o) {   // double precision; false for a singular matrix
    double a[4][8];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) a[r][c] = m[c * 4 + r], a[r][4 + c] = r == c ? 1.0 : 0.0;
    }
    for (int c = 0; c < 4; c++) {
        int p = c;
        for (int r = c + 1; r < 4; r++)
            if (fabs(a[r][c]) > fabs(a[p][c])) p = r;
        if (fabs(a[p][c]) < 1e-12) return false;
        if (p != c)
            for (int k = 0; k < 8; k++) {
                double t = a[c][k];
                a[c][k] = a[p][k], a[p][k] = t;
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
    return true;
}

static void mul4(const float *a, const float *b, float *o) {   // o = a * b, column-major
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            double s_ = 0.0;
            for (int k = 0; k < 4; k++) s_ += (double)a[k * 4 + r] * b[c * 4 + k];
            o[c * 4 + r] = (float)s_;
        }
}

// ---- views of the game's images (kept: a view destroyed while a recorded command uses it faults the driver) --------
static struct { VkImage img; VkFormat fmt; VkImageAspectFlags asp; VkImageView view; } *VC;
static int VC_n, VC_cap;

static VkImageView view_of(VkCtx *c, VkImage img, VkFormat fmt, VkImageAspectFlags asp) {
    for (int i = 0; i < VC_n; i++)
        if (VC[i].img == img && VC[i].fmt == fmt && VC[i].asp == asp) return VC[i].view;
    if (VC_n == VC_cap) {   // (never waits for the device: this runs on the replay thread)
        int nc = VC_cap ? VC_cap * 2 : 8;
        void *np = realloc(VC, (size_t)nc * sizeof *VC);
        if (!np) return VK_NULL_HANDLE;
        VC = (decltype(VC))np, VC_cap = nc;
    }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = fmt, vi.subresourceRange = {asp, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(c->device, &vi, NULL, &v) != VK_SUCCESS) return VK_NULL_HANDLE;
    VC[VC_n].img = img, VC[VC_n].fmt = fmt, VC[VC_n].asp = asp, VC[VC_n].view = v, VC_n++;
    return v;
}

static void views_drop(VkCtx *c) {
    for (int i = 0; i < VC_n; i++) vkDestroyImageView(c->device, VC[i].view, NULL);
    VC_n = 0, VC_cap = 0;
    free(VC), VC = nullptr;
}

// ---- images ----------------------------------------------------------------------------------------------------
static bool img2d(VkCtx *c, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags use, VkImage *img, VmaAllocation *a,
                  VkImageView *v) {
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = f, ci.extent = {w, h, 1}, ci.mipLevels = 1, ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT, ci.tiling = VK_IMAGE_TILING_OPTIMAL, ci.usage = use;
    ci.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    VmaAllocationCreateInfo ai = {VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ci, &ai, img, a, NULL) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = *img, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = f;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(c->device, &vi, NULL, v) == VK_SUCCESS;
}

static void destroy_images(VkCtx *c) {
    VkImage *im[7] = {&S.nv, &S.nw, &S.mat, &S.out, &S.mv, &S.env, &S.lut};
    VmaAllocation *al[7] = {&S.nv_a, &S.nw_a, &S.mat_a, &S.out_a, &S.mv_a, &S.env_a, &S.lut_a};
    VkImageView *vi[7] = {&S.nv_v, &S.nw_v, &S.mat_v, &S.out_v, &S.mv_v, &S.env_v, &S.lut_v};
    for (int i = 0; i < 7; i++) {
        if (*vi[i]) vkDestroyImageView(c->device, *vi[i], NULL), *vi[i] = VK_NULL_HANDLE;
        if (*im[i]) vmaDestroyImage(c->vma, *im[i], *al[i]), *im[i] = VK_NULL_HANDLE;
    }
}

static bool make_images(VkCtx *c, uint32_t w, uint32_t h) {
    const VkImageUsageFlags su = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    if (!img2d(c, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, su, &S.nv, &S.nv_a, &S.nv_v) ||
        !img2d(c, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, su, &S.nw, &S.nw_a, &S.nw_v) ||
        !img2d(c, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, su, &S.mat, &S.mat_a, &S.mat_v) ||
        !img2d(c, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, su, &S.out, &S.out_a, &S.out_v) ||
        !img2d(c, w, h, VK_FORMAT_R16G16_SFLOAT, su, &S.mv, &S.mv_a, &S.mv_v))
        return false;
    // The environment map SSSR falls back to when a ray leaves the screen, and the BRDF LUT its composite reads: black,
    // since the port has no sky probe, so a missed ray adds nothing.
    {
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        ci.imageType = VK_IMAGE_TYPE_2D, ci.format = VK_FORMAT_R16G16B16A16_SFLOAT, ci.extent = {16, 16, 1};
        ci.mipLevels = 1, ci.arrayLayers = 6, ci.samples = VK_SAMPLE_COUNT_1_BIT, ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        VmaAllocationCreateInfo ai = {VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &S.env, &S.env_a, NULL) != VK_SUCCESS) return false;
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = S.env, vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE, vi.format = ci.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        if (vkCreateImageView(c->device, &vi, NULL, &S.env_v) != VK_SUCCESS) return false;
    }
    return img2d(c, 128, 128, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                 &S.lut, &S.lut_a, &S.lut_v);
}

static void clear_black(VkCommandBuffer cb, VkImage img) {
    VkImageSubresourceRange r = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    VkClearColorValue z = {{0.0f, 0.0f, 0.0f, 1.0f}};
    vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r);
}

// ---- the three compute passes ---------------------------------------------------------------------------------
static bool make_one_pipeline(VkCtx *c, const uint32_t *code, size_t size, uint32_t nbind,
                              const VkDescriptorType *types, uint32_t pc_size, VkDescriptorSetLayout *dsl,
                              VkPipelineLayout *lay, VkPipeline *pipe) {
    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i < nbind; i++) {
        b[i].binding = i;
        b[i].descriptorType = types[i];
        b[i].descriptorCount = 1, b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, b[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, lci.bindingCount = nbind, lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(c->device, &lci, NULL, dsl) != VK_SUCCESS) return false;
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1, pl.pSetLayouts = dsl, pl.pushConstantRangeCount = 1, pl.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(c->device, &pl, NULL, lay) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = size, mi.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = m, ci.stage.pName = "main";
    ci.layout = *lay;
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    return r == VK_SUCCESS;
}

static bool make_pipelines(VkCtx *c) {
    S.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    if (!S.push) return false;
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0;
    if (vkCreateSampler(c->device, &si, NULL, &S.samp) != VK_SUCCESS) return false;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    if (vkCreateSampler(c->device, &si, NULL, &S.splin) != VK_SUCCESS) return false;
    const VkDescriptorType sam = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sto = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const VkDescriptorType tprep[4] = {sam, sam, sto, sto}, tconv[2] = {sam, sto}, tap[3] = {sam, sam, sto};
    const VkDescriptorType tmv[2] = {sam, sto};
    return make_one_pipeline(c, sssr_prep_spv, sizeof sssr_prep_spv, 4, tprep, 32, &S.prep_dsl, &S.prep_lay,
                             &S.prep_pipe) &&
           make_one_pipeline(c, sssr_conv_spv, sizeof sssr_conv_spv, 2, tconv, 64, &S.conv_dsl, &S.conv_lay,
                             &S.conv_pipe) &&
           make_one_pipeline(c, sssr_mv_spv, sizeof sssr_mv_spv, 2, tmv, 96, &S.mv_dsl, &S.mv_lay, &S.mv_pipe) &&
           make_one_pipeline(c, sssr_apply_spv, sizeof sssr_apply_spv, 3, tap, 96, &S.apply_dsl, &S.apply_lay,
                             &S.apply_pipe);
}

// ---- the SDK's view of an image --------------------------------------------------------------------------------
static FfxResource res(VkImage img, VkFormat f, uint32_t w, uint32_t h, bool write, const wchar_t *name) {
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = f, ci.extent = {w, h, 1}, ci.mipLevels = 1, ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | (write ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
    return ffxGetResourceVK((void *)img, ffxGetImageResourceDescriptionVK(img, ci, write ? FFX_RESOURCE_USAGE_UAV
                                                                                        : FFX_RESOURCE_USAGE_READ_ONLY),
                            name, write ? FFX_RESOURCE_STATE_UNORDERED_ACCESS : FFX_RESOURCE_STATE_GENERIC_READ);
}

static FfxResource res_cube(VkImage img, uint32_t w) {
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = VK_FORMAT_R16G16B16A16_SFLOAT, ci.extent = {w, w, 1};
    ci.mipLevels = 1, ci.arrayLayers = 6, ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    return ffxGetResourceVK((void *)img, ffxGetImageResourceDescriptionVK(img, ci, FFX_RESOURCE_USAGE_READ_ONLY),
                            L"SSSR_Env", FFX_RESOURCE_STATE_GENERIC_READ);
}

static void destroy_all(VkCtx *c) {
    if (S.ok) {
        ffxSssrContextDestroy(&S.ctx);
        S.ok = false;
    }
    if (c && c->device) {
        destroy_images(c);
        if (S.prep_pipe) vkDestroyPipeline(c->device, S.prep_pipe, NULL), S.prep_pipe = VK_NULL_HANDLE;
        if (S.conv_pipe) vkDestroyPipeline(c->device, S.conv_pipe, NULL), S.conv_pipe = VK_NULL_HANDLE;
        if (S.mv_pipe) vkDestroyPipeline(c->device, S.mv_pipe, NULL), S.mv_pipe = VK_NULL_HANDLE;
        if (S.apply_pipe) vkDestroyPipeline(c->device, S.apply_pipe, NULL), S.apply_pipe = VK_NULL_HANDLE;
        if (S.prep_lay) vkDestroyPipelineLayout(c->device, S.prep_lay, NULL), S.prep_lay = VK_NULL_HANDLE;
        if (S.conv_lay) vkDestroyPipelineLayout(c->device, S.conv_lay, NULL), S.conv_lay = VK_NULL_HANDLE;
        if (S.mv_lay) vkDestroyPipelineLayout(c->device, S.mv_lay, NULL), S.mv_lay = VK_NULL_HANDLE;
        if (S.apply_lay) vkDestroyPipelineLayout(c->device, S.apply_lay, NULL), S.apply_lay = VK_NULL_HANDLE;
        if (S.prep_dsl) vkDestroyDescriptorSetLayout(c->device, S.prep_dsl, NULL), S.prep_dsl = VK_NULL_HANDLE;
        if (S.conv_dsl) vkDestroyDescriptorSetLayout(c->device, S.conv_dsl, NULL), S.conv_dsl = VK_NULL_HANDLE;
        if (S.mv_dsl) vkDestroyDescriptorSetLayout(c->device, S.mv_dsl, NULL), S.mv_dsl = VK_NULL_HANDLE;
        if (S.apply_dsl) vkDestroyDescriptorSetLayout(c->device, S.apply_dsl, NULL), S.apply_dsl = VK_NULL_HANDLE;
        if (S.samp) vkDestroySampler(c->device, S.samp, NULL), S.samp = VK_NULL_HANDLE;
        if (S.splin) vkDestroySampler(c->device, S.splin, NULL), S.splin = VK_NULL_HANDLE;
        views_drop(c);
    }
    scratch_free(S.scratch);
    S.scratch = nullptr;
    S.have_prev = false;
    S.n = 0, S.prepared = false;
}

#define TRY(x, what)                                                                                                   \
    do {                                                                                                               \
        FfxErrorCode e_ = (x);                                                                                         \
        if (e_ != FFX_OK) {                                                                                            \
            rt_log("sssr: %s failed (%#x) at %ux%u: reflections off", what, (unsigned)e_, S.w, S.h);                \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

static bool create(VkCtx *c, uint32_t w, uint32_t h) {
    S.w = w, S.h = h;
    size_t n = (ffxGetScratchMemorySizeVK(c->phys, FFX_SSSR_CONTEXT_COUNT) + 255) & ~(size_t)255;
    if (!(S.scratch = scratch_alloc(n))) return false;
    VkDeviceContext dc = {c->device, c->phys, device_proc};
    FfxSssrContextDescription cd = {};
    TRY(ffxGetInterfaceVK(&cd.backendInterface, ffxGetDeviceVK(&dc), S.scratch, n, FFX_SSSR_CONTEXT_COUNT),
        "the backend");
    cd.flags = 0;   // depth not inverted
    cd.renderSize = {w, h};
    cd.normalsHistoryBufferFormat = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    TRY(ffxSssrContextCreate(&S.ctx, &cd), "context creation");
    S.iface = cd.backendInterface;
    S.ok = true;
    if (!make_images(c, w, h) || !make_pipelines(c)) {
        rt_log("sssr: the input images or the passes could not be created: reflections off");
        return false;
    }
    S.have_prev = false, S.n = 0, S.frames = 0, S.prepared = false;
    memset(S.ring, 0, sizeof S.ring);
    rt_log("sssr: FidelityFX SSSR up to %ux%u (MIT, the SDK 1.1.4)", w, h);
    return true;
}

static bool ensure(VkCtx *c, uint32_t w, uint32_t h, VkCommandBuffer cb) {
    if (S.ok && S.w == w && S.h == h) return true;
    if (S.ok) {
        vkDeviceWaitIdle(c->device);
        destroy_all(c);
    }
    if (!create(c, w, h)) {
        destroy_all(c);
        S.failed = true;
        return false;
    }
    if (cb) {
        clear_black(cb, S.env);
        clear_black(cb, S.lut);
        clear_black(cb, S.out);
        clear_black(cb, S.nw);
        clear_black(cb, S.mv);
    }
    return true;
}

static void wait_ring(uint64_t fence) {
    if (S.n >= FFX_MAX_QUEUED_FRAMES - 1) {
        uint64_t back = S.ring[(S.n + 1) % FFX_MAX_QUEUED_FRAMES];
        if (back && vk_sync_fence_done() < back) {
            if (!S.waits++) rt_log("sssr: waiting for a dispatch %d back", FFX_MAX_QUEUED_FRAMES - 1);
            vk_sync_fence_wait(back, 2000000000ull);
        }
    }
    S.ring[S.n % FFX_MAX_QUEUED_FRAMES] = fence;
}

extern "C" int vk_sssr_on(void) {
    static int v = -1;
    static unsigned seen;
    unsigned ver = cfg_version();
    if (v < 0 || ver != seen) {   // the menu's row sets it while the game runs
        seen = ver;
        const char *sv = cfg_get("TOTK_SSSR");
        int s = sv && *sv ? atoi(sv) : 0;
        v = s < 0 ? 0 : s > 200 ? 200 : s;
    }
    return v;
}

static void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
}

extern "C" int vk_sssr_prepare(void *cb_, uint64_t lin_depth, uint64_t lit, uint32_t w, uint32_t h, float px, float py,
                              float far) {
    if (S.failed || !vk_sssr_on() || !cb_ || !lin_depth || !w || !h) return 0;
    VkCtx *c = vk_ctx();
    VkCommandBuffer cb = (VkCommandBuffer)cb_;
    // The context follows the scene's render size (DLSS modes and resolution change it), but not the inventory's icon
    // targets: rebuilding for those (and back) every frame took the device down.
    if (S.ok && (w != S.w || h != S.h)) {
        if (w < 1280 || h < 720) return 0;
        vkDeviceWaitIdle(c->device);
        destroy_all(c);
    }
    if (!ensure(c, w, h, cb)) return 0;
    VkImageView dv = view_of(c, (VkImage)lin_depth, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!dv) return 0;
    VkImageView lv = view_of(c, (VkImage)lit, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!lv) return 0;
    VkDescriptorImageInfo ii = {S.samp, dv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo li = {S.samp, lv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo ni = {VK_NULL_HANDLE, S.nv_v, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo mi = {VK_NULL_HANDLE, S.mat_v, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet ws[4] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &li, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ni, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mi, NULL, NULL}};
    float p4[8] = {px, py, far, 0, (float)w, (float)h, 1.0f / (float)w, 1.0f / (float)h};
    barrier(cb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.prep_pipe);
    S.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.prep_lay, 0, 4, ws);
    vkCmdPushConstants(cb, S.prep_lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p4, p4);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    barrier(cb);
    S.prep_w = w, S.prep_h = h, S.prepared = true;
    return 1;
}

extern "C" int vk_sssr_exec(void *cb_, uint64_t scene, uint64_t depth, uint64_t mv, uint32_t w, uint32_t h,
                            const float *cam, uint64_t fence) {
    if (S.failed || !vk_sssr_on() || !cb_ || !scene || !depth || !mv || !cam || !w || !h) return 0;
    if (!S.prepared || w != S.prep_w || h != S.prep_h) return 0;   // no normals for this frame (or another size)
    VkCtx *c = vk_ctx();
    VkCommandBuffer cb = (VkCommandBuffer)cb_;
    static int cfgd = -1;
    if (cfgd < 0) {
        const char *dv = cfg_get("TOTK_SSSR_DEBUG");
        cfgd = dv && *dv ? atoi(dv) : 0;
    }
    S.debug = cfgd;
    static int nodisp = -1, noapply = -1, noconv = -1;   // bisect switches (tests): which part writes the picture
    if (nodisp < 0) {
        const char *v = cfg_get("TOTK_SSSR_NODISP");
        nodisp = v && *v && strcmp(v, "0") != 0;
        v = cfg_get("TOTK_SSSR_NOAPPLY");
        noapply = v && *v && strcmp(v, "0") != 0;
        v = cfg_get("TOTK_SSSR_NOCONV");
        noconv = v && *v && strcmp(v, "0") != 0;
    }
    float view[16], vp[16], proj[16], invp[16], invvp[16], invv[16];
    rows_to_col(cam, 3, view);          // rows 0-2: view
    rows_to_col(cam + 12, 4, vp);       // rows 3-6: view-projection
    rows_to_col(cam + 28, 4, proj);     // rows 7-10: projection
    if (!inv4(proj, invp) || !inv4(vp, invvp) || !inv4(view, invv)) return 0;
    static int logged;
    if (!logged) {
        logged = 1;
        rt_log("sssr: proj [%g %g %g %g] [%g %g %g %g] [%g %g %g %g] [%g %g %g %g]", proj[0], proj[1], proj[2], proj[3],
               proj[4], proj[5], proj[6], proj[7], proj[8], proj[9], proj[10], proj[11], proj[12], proj[13], proj[14],
               proj[15]);
    }
    if (!S.have_prev) {
        memcpy(S.prev_vp, vp, sizeof vp);
        S.have_prev = true;
    }
    wait_ring(fence);
    // 1. the view-space normal to world space (SSSR multiplies by the view matrix itself)
    if (!noconv) {
        VkDescriptorImageInfo ni = {S.splin, S.nv_v, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo oi = {VK_NULL_HANDLE, S.nw_v, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet ws[3] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ni, NULL, NULL},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &oi, NULL, NULL},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &oi, NULL, NULL}};
        float p4[16] = {(float)w, (float)h, 1.0f / (float)w, 1.0f / (float)h,
                        invv[0], invv[1], invv[2], 0, invv[4], invv[5], invv[6], 0, invv[8], invv[9], invv[10], 0};
        barrier(cb);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.conv_pipe);
        S.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.conv_lay, 0, 2, ws);
        vkCmdPushConstants(cb, S.conv_lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p4, p4);
        vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
        barrier(cb);
    }
    // 1b. the camera's own motion vectors (prevVP * invView, in UV): the port's carry jitter and the objects' motion
    {
        float m[16], inv[16];
        mul4(S.prev_vp, invv, m);
        VkDescriptorImageInfo mi = {S.samp, S.mat_v, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo oi = {VK_NULL_HANDLE, S.mv_v, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet ws[2] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &mi, NULL, NULL},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &oi, NULL, NULL}};
        float p4[24] = {proj[0], proj[5], 0, 0, (float)w, (float)h, 1.0f / (float)w, 1.0f / (float)h,
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                        m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]};
        (void)inv;
        barrier(cb);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.mv_pipe);
        S.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.mv_lay, 0, 2, ws);
        vkCmdPushConstants(cb, S.mv_lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p4, p4);
        vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
        barrier(cb);
    }
    // 2. the SDK's dispatch
    FfxSssrDispatchDescription dd = {};
    dd.commandList = ffxGetCommandListVK(cb);
    dd.color = res((VkImage)scene, VK_FORMAT_B10G11R11_UFLOAT_PACK32, w, h, false, L"SSSR_Color");
    dd.depth = res((VkImage)depth, VK_FORMAT_D32_SFLOAT, w, h, false, L"SSSR_Depth");
    dd.motionVectors = res(S.mv, VK_FORMAT_R16G16_SFLOAT, w, h, false, L"SSSR_MV");
    dd.normal = res(S.nw, VK_FORMAT_R16G16B16A16_SFLOAT, w, h, false, L"SSSR_Normal");
    dd.materialParameters = res(S.mat, VK_FORMAT_R16G16B16A16_SFLOAT, w, h, false, L"SSSR_Material");
    dd.environmentMap = res_cube(S.env, 16);
    dd.brdfTexture = res(S.lut, VK_FORMAT_R16G16_SFLOAT, 128, 128, false, L"SSSR_BRDF");
    dd.output = res(S.out, VK_FORMAT_R16G16B16A16_SFLOAT, w, h, true, L"SSSR_Output");
    memcpy(dd.invViewProjection, invvp, sizeof invvp);
    memcpy(dd.projection, proj, sizeof proj);
    memcpy(dd.invProjection, invp, sizeof invp);
    memcpy(dd.view, view, sizeof view);
    memcpy(dd.invView, invv, sizeof invv);
    memcpy(dd.prevViewProjection, S.prev_vp, sizeof S.prev_vp);
    dd.renderSize = {w, h};
    // The port's motion vectors are in pixels (what DLSS reads); the SDK's temporal reprojection adds them to a UV,
    // so the scale takes them there. TOTK_SSSR_MV_SIGN=-1 / TOTK_SSSR_MV_YFLIP=1 test the convention: the history
    // reprojection is the only consumer, and its sign shows as grain in motion.
    static float mv_sign = 1.0f, mv_yflip = 1.0f;
    static int mv_read;
    if (!mv_read) {
        mv_read = 1;
        mv_sign = -1.0f;   // measured: the port's vectors run current minus previous, the other way round from the SDK's
        const char *sv = cfg_get("TOTK_SSSR_MV_SIGN");
        if (sv && *sv && atoi(sv) > 0) mv_sign = 1.0f;
        const char *yv = cfg_get("TOTK_SSSR_MV_YFLIP");
        if (yv && *yv && strcmp(yv, "0") != 0) mv_yflip = -1.0f;
    }
    dd.motionVectorScale = {1.0f, 1.0f};   // (the pass writes UV already)
    dd.iblFactor = 0.0f;
    dd.normalUnPackMul = 2.0f;
    dd.normalUnPackAdd = -1.0f;
    dd.roughnessChannel = 1;   // G: roughness squared
    dd.isRoughnessPerceptual = false;
    dd.temporalStabilityFactor = 0.9f;   // more history: the grain of one-ray-per-pixel tracing averages out
    dd.depthBufferThickness = 0.015f;
    dd.roughnessThreshold = 0.26f;   // more of the scene counts as reflective
    dd.varianceThreshold = 0.0f;
    dd.maxTraversalIntersections = 64;   // (4K full screen: 128 was far too slow here)
    dd.minTraversalOccupancy = 4;
    dd.mostDetailedMip = 0;
    dd.samplesPerQuad = 1;
    dd.temporalVarianceGuidedTracingEnabled = 0;
    FfxErrorCode e = nodisp ? FFX_OK : ffxSssrContextDispatch(&S.ctx, &dd);
    S.n++, S.frames++;
    memcpy(S.prev_vp, vp, sizeof vp);
    S.prepared = false;
    if (e != FFX_OK) {
        rt_log("sssr: dispatch failed (%#x) at %ux%u: reflections off", (unsigned)e, w, h);
        S.failed = true;
        return 0;
    }
    // 3. onto the scene
    if (!noapply) {
        VkDescriptorImageInfo ri = {S.splin, S.out_v, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo ni = {S.splin, S.nw_v, VK_IMAGE_LAYOUT_GENERAL};
        VkImageView ov = view_of(c, (VkImage)scene, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_ASPECT_COLOR_BIT);
        if (!ov) return 0;
        VkDescriptorImageInfo oi = {VK_NULL_HANDLE, ov, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet ws[3] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ri, NULL, NULL},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ni, NULL, NULL},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &oi, NULL, NULL}};
        float px = proj[0], py = proj[5];
        float farf = proj[10] * (proj[14] / proj[10]) / (1.0f + proj[10]);
        float p4[24] = {px, py, farf, (float)vk_sssr_on() / 100.0f, (float)w, (float)h,
                        1.0f / (float)w, 1.0f / (float)h, (float)S.debug, 0, 0, 0,
                        view[0], view[1], view[2], 0, view[4], view[5], view[6], 0, view[8], view[9], view[10], 0};
        barrier(cb);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.apply_pipe);
        S.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.apply_lay, 0, 3, ws);
        vkCmdPushConstants(cb, S.apply_lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p4, p4);
        vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
        barrier(cb);
    }
    if (S.frames == 1)
        rt_log("sssr: reflected at %ux%u, strength %d%% (TOTK_SSSR, FidelityFX SSSR)", w, h, vk_sssr_on());
    return 1;
}

extern "C" void vk_sssr_release(void) {
    VkCtx *c = vk_ctx();
    if (!c || !c->device || (!S.ok && !S.scratch)) return;
    vkDeviceWaitIdle(c->device);
    destroy_all(c);
}
