p = 'runtime/vk_rt.c'
s = open(p, encoding='utf-8').read()

func = '''
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
    VkImageView sv = VK_NULL_HANDLE;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                .image = (VkImage)scene,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &sv) != VK_SUCCESS) return;
    float view[16], vp[16], invv[16], m[16];
    rows_to_col(cam, 3, view);
    rows_to_col(cam + 12, 4, vp);
    if (!inv4m(view, invv)) {
        vkDestroyImageView(c->device, sv, NULL);
        return;
    }
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
    float blend = S.hist_frames ? 0.85f : 0.0f;
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
    vkDestroyImageView(c->device, sv, NULL);
    memcpy(S.prev_vp, vp, sizeof vp);
    S.hist_cur ^= 1;
    S.hist_frames++;
}

'''
s = s.replace('void vk_rt_shutdown(void) {', func + 'void vk_rt_shutdown(void) {', 1)
s = s.replace('    float prev_vp[16];\n    int have_prev;',
              '    float prev_vp[16];\n    int have_prev;\n    uint32_t hist_frames;', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ok')
