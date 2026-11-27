p = 'runtime/vk_rt.c'
s = open(p, encoding='utf-8').read()

# a persistent view cache: a view destroyed while a recorded command still uses it faults the GPU
helper = '''
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
        TV = (decltype(TV))np, TV_cap = nc;
    }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
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
'''
s = s.replace('// ---- the ray shadows temporal half', helper + '\n// ---- the ray shadows temporal half', 1)
s = s.replace("// ---- the ray shadows' temporal half", helper + "\n// ---- the ray shadows' temporal half", 1)

# use it instead of the per-frame view
s = s.replace('''    VkImageView sv = VK_NULL_HANDLE;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                .image = (VkImage)scene,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &sv) != VK_SUCCESS) return;''',
'''    VkImageView sv = temp_view(c, (VkImage)scene, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    if (!sv) return;''')
s = s.replace('''    if (!inv4m(view, invv)) {
        vkDestroyImageView(c->device, sv, NULL);
        return;
    }''','''    if (!inv4m(view, invv)) return;''')
s = s.replace('    vkDestroyImageView(c->device, sv, NULL);\n    memcpy(S.prev_vp, vp, sizeof vp);',
              '    memcpy(S.prev_vp, vp, sizeof vp);')

# free them at shutdown
s = s.replace('''    free(rt_inst);
    rt_inst = NULL, rt_ninst = rt_cinst = 0;''','''    free(rt_inst);
    rt_inst = NULL, rt_ninst = rt_cinst = 0;
    temp_views_drop(c);''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ok')
