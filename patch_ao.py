p = 'runtime/vk_contact.c'
s = open(p, encoding='utf-8').read()

s = s.replace('#include "contact_shadow_spv.h"', '#include "contact_shadow_spv.h"\n#include "gtao_spv.h"', 1)

a_struct = '''
// TOTK_GTAO (percent, 0 = off): screen-space ambient occlusion at the same composite point, after the contact shadows.
// gtao.comp marches the linear depth around each pixel (horizon integration) and multiplies the scene colour by the
// occlusion; the depth normals keep it independent of the G-buffer.
static struct {
    int read, tried, ok, strength, debug, logged;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkSampler sampler;
} A;

static int ao_strength(void) {
    if (!A.read) {
        int s = cfg_int("TOTK_GTAO");
        A.strength = s < 0 ? 0 : s > 100 ? 100 : s;
        A.read = 1;
    }
    return A.strength;
}

static int ao_init(VkCtx *c) {
    A.tried = 1;
    if (!ao_strength()) return 0;
    const char *e = cfg_get("TOTK_GTAO_DEBUG");
    A.debug = e ? atoi(e) : 0;
    A.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[2] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 2, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &A.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = sizeof gtao_spv, .pCode = gtao_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!A.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &A.dsl) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &li, NULL, &A.layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &A.sampler) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) {
        rt_log("vk: GTAO off: its pipeline could not be created");
        return 0;
    }
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = A.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &A.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) {
        rt_log("vk: GTAO off: its pipeline could not be created");
        return 0;
    }
    return A.ok = 1;
}
'''
s = s.replace('} C;\n\nint vk_contact_strength(void) {', '} C;\n' + a_struct + '\nint vk_contact_strength(void) {', 1)

ao_dispatch = '''
// The AO dispatch (TOTK_GTAO): the scene colour multiplied by the occlusion from gtao.comp, over the same images.
static void ao_dispatch(VkCommandBuffer cb, const VkExecJob *j, VkImageView in, VkImageView out) {
    VkDescriptorImageInfo ii = {A.sampler, in, VK_IMAGE_LAYOUT_GENERAL}, oi = {VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi}};
    float w_ = (float)j->scene.width, h_ = (float)j->scene.height;
    float ap[16] = {j->pc[0], j->pc[1], j->pc[2], (float)A.strength / 100.0f,
                    w_, h_, 1.0f / w_, 1.0f / h_, (float)A.debug, 0.6f, 4.0f, 8.0f, 0, 0, 0, 0};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, A.pipe);
    rec_push(A.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, A.layout, 0, 2, w);
    vkCmdPushConstants(cb, A.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof ap, ap);
    vkCmdDispatch(cb, (j->scene.width + 7) / 8, (j->scene.height + 7) / 8, 1);
    vk_texture_storage_written(j->scene.img);
    mem_barrier(cb);
    if (!A.logged) {
        A.logged = 1;
        rt_log("vk: GTAO on, strength %d%%%% on a %ux%u scene", A.strength, j->scene.width, j->scene.height);
    }
}
'''
marker = '// The half-resolution lighting pass after the G-buffer'
s = s.replace(marker, ao_dispatch + '\n' + marker, 1)

s = s.replace('if (!vk_contact_strength()) return 0;', 'if (!vk_contact_strength() && !ao_strength()) return 0;', 1)

old_head = '''    if (!C.tried) init(vk_ctx());
    if (!C.ok) return;'''
new_head = '''    if (!C.tried) init(vk_ctx());
    if (!A.tried) ao_init(vk_ctx());
    if (!C.ok) {
        if (!A.ok) return;   // AO alone: TOTK_CONTACT_SHADOWS=0 with TOTK_GTAO on
        VkImageView a_in = vk_exec_target_view(j->depth.img, VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D,
                                               VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
        VkImageView a_out = vk_exec_target_view(j->scene.img, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_VIEW_TYPE_2D,
                                                VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
        if (!a_in || !a_out) return;
        vk_exec_barrier(ex);
        ao_dispatch(ex->cb, j, a_in, a_out);
        ex->recorded += 6;
        vk_exec_draw_new_cb(ex);
        return;
    }'''
s = s.replace(old_head, new_head, 1)

old_tail = '''    ex->recorded += 6;
    vk_exec_draw_new_cb(ex);   // the game's bound compute pipeline and descriptors were replaced: bind them again
}'''
new_tail = '''    ex->recorded += 6;
    vk_exec_draw_new_cb(ex);   // the game's bound compute pipeline and descriptors were replaced: bind them again
    if (A.ok) {   // the AO pass (TOTK_GTAO), after the contact shadows, over the same colour and depth
        ao_dispatch(cb, j, in, out);
        ex->recorded += 6;
        vk_exec_draw_new_cb(ex);
    }
}'''
s = s.replace(old_tail, new_tail, 1)

s = s.replace('''    if (C.pipe) vkDestroyPipeline(c->device, C.pipe, NULL);''', '''    if (A.pipe) vkDestroyPipeline(c->device, A.pipe, NULL);
    if (A.sampler) vkDestroySampler(c->device, A.sampler, NULL);
    if (A.layout) vkDestroyPipelineLayout(c->device, A.layout, NULL);
    if (A.dsl) vkDestroyDescriptorSetLayout(c->device, A.dsl, NULL);
    memset(&A, 0, sizeof A);
    if (C.pipe) vkDestroyPipeline(c->device, C.pipe, NULL);''', 1)

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched A refs:', s.count('A.'), 'gtao:', s.count('gtao_spv'))
