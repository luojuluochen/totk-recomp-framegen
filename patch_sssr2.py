import io

p = 'runtime/shaders/sssr_prep.comp'
s = open(p, encoding='utf-8').read()
s = s.replace('''layout(binding = 0) uniform sampler2D lin_depth;              // view distance / far
layout(binding = 1, rgba16f) uniform image2D normal_out;      // xyz: the view-space normal, w: 1
layout(binding = 2, rgba16f) uniform image2D material_out;    // r: 1 (AO), g: roughness squared, b: 0, a: 1''',
'''layout(binding = 0) uniform sampler2D lin_depth;              // view distance / far
layout(binding = 1) uniform sampler2D lit;                    // the lit scene colour: its local detail says how rough
layout(binding = 2, rgba16f) uniform image2D normal_out;      // xyz: the view-space normal, w: 1
layout(binding = 3, rgba16f) uniform image2D material_out;    // r: 1 (AO), g: roughness squared, b: 0, a: 1''')
s = s.replace('''    float sm2 = dot(n, n2), sm3 = dot(n, n3);
    float smooth_w = smoothstep(0.75, 0.97, min(sm2, sm3));
    float rough = mix(0.6, 0.08, smooth_w);''',
'''    float sm2 = dot(n, n2), sm3 = dot(n, n3);
    float smooth_w = smoothstep(0.75, 0.97, min(sm2, sm3));
    // A flat surface in the depth is not a polished one: sand, gravel and soil keep the depth plane too. Their lit colour
    // carries texture at pixel scale, a glazed floor or water does not, so the local contrast raises the roughness.
    vec3 c0 = texelFetch(lit, px, 0).rgb;
    float det = 0.0;
    for (int k = 0; k < 4; k++) {
        ivec2 q = px + ivec2(k == 0 ? 3 : k == 1 ? -3 : 0, k == 2 ? 3 : k == 3 ? -3 : 0);
        vec3 c1 = texelFetch(lit, clamp(q, ivec2(0), ivec2(pc.size.xy) - 1), 0).rgb;
        det += abs(dot(c1 - c0, vec3(0.299, 0.587, 0.114)));
    }
    det *= 0.25 / max(max(c0.r, max(c0.g, c0.b)), 0.01);
    float plain = 1.0 - smoothstep(0.03, 0.12, det);
    float rough = mix(0.6, 0.08, smooth_w * plain);''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)

p = 'runtime/vk_sssr.cpp'
s = open(p, encoding='utf-8').read()
s = s.replace('''static bool make_one_pipeline(VkCtx *c, const uint32_t *code, size_t size, uint32_t sampler_bindings, uint32_t pc_size,
                              VkDescriptorSetLayout *dsl, VkPipelineLayout *lay, VkPipeline *pipe) {
    VkDescriptorSetLayoutBinding b[3];
    for (uint32_t i = 0; i < 3; i++) {
        b[i].binding = i;
        b[i].descriptorType = i < sampler_bindings ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[i].descriptorCount = 1, b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, b[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, lci.bindingCount = 3, lci.pBindings = b;''',
'''static bool make_one_pipeline(VkCtx *c, const uint32_t *code, size_t size, uint32_t nbind,
                              const VkDescriptorType *types, uint32_t pc_size, VkDescriptorSetLayout *dsl,
                              VkPipelineLayout *lay, VkPipeline *pipe) {
    VkDescriptorSetLayoutBinding b[4];
    for (uint32_t i = 0; i < nbind; i++) {
        b[i].binding = i;
        b[i].descriptorType = types[i];
        b[i].descriptorCount = 1, b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, b[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, lci.bindingCount = nbind, lci.pBindings = b;''')
s = s.replace('''    return make_one_pipeline(c, sssr_prep_spv, sizeof sssr_prep_spv, 1, 32, &S.prep_dsl, &S.prep_lay, &S.prep_pipe) &&
           make_one_pipeline(c, sssr_conv_spv, sizeof sssr_conv_spv, 1, 64, &S.conv_dsl, &S.conv_lay, &S.conv_pipe) &&
           make_one_pipeline(c, sssr_apply_spv, sizeof sssr_apply_spv, 2, 96, &S.apply_dsl, &S.apply_lay, &S.apply_pipe);''',
'''    const VkDescriptorType sam = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sto = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const VkDescriptorType tprep[4] = {sam, sam, sto, sto}, tconv[2] = {sam, sto}, tap[3] = {sam, sam, sto};
    return make_one_pipeline(c, sssr_prep_spv, sizeof sssr_prep_spv, 4, tprep, 32, &S.prep_dsl, &S.prep_lay,
                             &S.prep_pipe) &&
           make_one_pipeline(c, sssr_conv_spv, sizeof sssr_conv_spv, 2, tconv, 64, &S.conv_dsl, &S.conv_lay,
                             &S.conv_pipe) &&
           make_one_pipeline(c, sssr_apply_spv, sizeof sssr_apply_spv, 3, tap, 96, &S.apply_dsl, &S.apply_lay,
                             &S.apply_pipe);''')
s = s.replace('''extern "C" int vk_sssr_prepare(void *cb_, uint64_t lin_depth, uint32_t w, uint32_t h, float px, float py, float far) {''',
'''extern "C" int vk_sssr_prepare(void *cb_, uint64_t lin_depth, uint64_t lit, uint32_t w, uint32_t h, float px, float py,
                              float far) {''')
s = s.replace('''    VkDescriptorImageInfo ii = {S.samp, dv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo ni = {VK_NULL_HANDLE, S.nv_v, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo mi = {VK_NULL_HANDLE, S.mat_v, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet ws[3] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ni, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mi, NULL, NULL}};''',
'''    VkImageView lv = view_of(c, (VkImage)lit, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!lv) return 0;
    VkDescriptorImageInfo ii = {S.samp, dv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo li = {S.samp, lv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo ni = {VK_NULL_HANDLE, S.nv_v, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo mi = {VK_NULL_HANDLE, S.mat_v, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet ws[4] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &li, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ni, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, VK_NULL_HANDLE, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mi, NULL, NULL}};''')
s = s.replace('''    S.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.prep_lay, 0, 3, ws);''',
              '''    S.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, S.prep_lay, 0, 4, ws);''')
s = s.replace('''        const char *sv = cfg_get("TOTK_SSSR_MV_SIGN");
        if (sv && *sv && atoi(sv) < 0) mv_sign = -1.0f;''',
'''        mv_sign = -1.0f;   // measured: the port's vectors run current minus previous, the other way round from the SDK's
        const char *sv = cfg_get("TOTK_SSSR_MV_SIGN");
        if (sv && *sv && atoi(sv) > 0) mv_sign = 1.0f;''')
s = s.replace('''        int s = sv && *sv ? atoi(sv) : 0;
        v = s < 0 ? 0 : s > 100 ? 100 : s;''',
'''        int s = sv && *sv ? atoi(sv) : 0;
        v = s < 0 ? 0 : s > 200 ? 200 : s;''')
s = s.replace('    dd.temporalStabilityFactor = 0.85f;', '    dd.temporalStabilityFactor = 0.9f;')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ok')
