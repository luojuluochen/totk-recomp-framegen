import re

# cfg: the key
p = 'runtime/cfg.c'
s = open(p, encoding='utf-8').read()
s = s.replace('"COLOR_GAMMA",\n', '"COLOR_FOG", "COLOR_GAMMA",\n', 1)
if '"COLOR_FOG"' not in s:
    s = s.replace('"COLOR_PRESET", "COLOR_SATURATION", "COLOR_TINT",', '"COLOR_FOG", "COLOR_PRESET", "COLOR_SATURATION", "COLOR_TINT",', 1)
s = s.replace('{"COLOR_TINT", "display"},', '{"COLOR_FOG", "display"}, {"COLOR_TINT", "display"},', 1)
s = s.replace('{"COLOR_TINT", "0", "-100..100", LIVE},', '{"COLOR_TINT", "0", "-100..100", LIVE},\n    {"COLOR_FOG", "0", "0..100", LIVE},', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# vk_color.c: a sixth value
p = 'runtime/vk_color.c'
s = open(p, encoding='utf-8').read()
s = s.replace('    float adj[5];                  // brightness (added), contrast, saturation, gamma, green/magenta tint',
              '    float adj[6];                  // brightness (added), contrast, saturation, gamma, green/magenta tint, fog')
s = s.replace('        C.adj[4] = (float)cfg_double("TOTK_COLOR_TINT") / 100.0f;',
              '        C.adj[4] = (float)cfg_double("TOTK_COLOR_TINT") / 100.0f;\n'
              '        C.adj[5] = (float)cfg_double("TOTK_COLOR_FOG") / 100.0f;')
s = s.replace(' || C.adj[4] != 0.0f;', ' || C.adj[4] != 0.0f || C.adj[5] != 0.0f;')
s = s.replace('    struct { int32_t offset[2], size[2]; float adj[4], tint[4]; } pc = {\n'
              '        {crop[0], crop[1]}, {crop[2], crop[3]}, {C.adj[0], C.adj[1], C.adj[2], C.adj[3]}, {C.adj[4], 0, 0, 0}};',
              '    struct { int32_t offset[2], size[2]; float adj[4], tint[4], fog[4]; } pc = {\n'
              '        {crop[0], crop[1]}, {crop[2], crop[3]}, {C.adj[0], C.adj[1], C.adj[2], C.adj[3]},\n'
              '        {C.adj[4], 0, 0, 0}, {C.adj[5], 0, 0, 0}};')
s = s.replace('VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 48};',
              'VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};')
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# the shader
p = 'runtime/shaders/color_grade.comp'
s = open(p, encoding='utf-8').read()
s = s.replace('    vec4 tint;      // x: green/magenta, -1 (magenta) .. 1 (green), 0 leaves it',
              '    vec4 tint;      // x: green/magenta, -1 (magenta) .. 1 (green), 0 leaves it\n'
              '    vec4 fog;       // x: 0..1, how much of the milky veil (a black point) to take out')
s = s.replace('''    c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, pc.adj.z);''',
              '''    // The fog's veil sits on the whole picture as a lifted black: subtracting it and stretching back is what makes
    // a hazy scene read clear. The game's own fog is untouched; this is the veil it leaves.
    float fg = clamp(pc.fog.x, 0.0, 1.0) * 0.20;
    if (fg > 0.001) c = max(c - fg, vec3(0.0)) / (1.0 - fg);
    c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, pc.adj.z);''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# text ids and strings
p = 'runtime/ui_text.h'
s = open(p, encoding='utf-8').read()
s = s.replace('UI_MENU_COLOR_TINT, UI_MENU_COLOR_TINT_DESC,', 'UI_MENU_COLOR_TINT, UI_MENU_COLOR_TINT_DESC, UI_MENU_COLOR_FOG, UI_MENU_COLOR_FOG_DESC,', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)

p = 'runtime/ui_text.c'
s = open(p, encoding='utf-8').read()
s = s.replace('''    [UI_MENU_COLOR_TINT_DESC] = "Shifts the picture between magenta (below 0) and green (above 0); 0 leaves it.",''',
              '''    [UI_MENU_COLOR_TINT_DESC] = "Shifts the picture between magenta (below 0) and green (above 0); 0 leaves it.",
    [UI_MENU_COLOR_FOG] = "Haze",
    [UI_MENU_COLOR_FOG_DESC] = "Takes out the milky veil a hazy scene leaves on the picture; the game's fog itself "
                               "stays. 0 leaves it.",''', 1)
s = s.replace('''    [UI_MENU_COLOR_TINT_DESC] = "在偏洋红（小于 0）与偏绿（大于 0）之间调整画面；0 表示不调整。",''',
              '''    [UI_MENU_COLOR_TINT_DESC] = "在偏洋红（小于 0）与偏绿（大于 0）之间调整画面；0 表示不调整。",
    [UI_MENU_COLOR_FOG] = "雾感",
    [UI_MENU_COLOR_FOG_DESC] = "去掉雾天在画面上留下的那层灰白；游戏自身的雾不变。0 表示不调整。",''', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# the menu row
p = 'runtime/ui/menu.cpp'
s = open(p, encoding='utf-8').read()
s = s.replace('    moved |= opt_slider("TOTK_COLOR_TINT", UI_MENU_COLOR_TINT, -100, 100, UI_MENU_COLOR_TINT_DESC, 5, "%+.0f %%");',
              '    moved |= opt_slider("TOTK_COLOR_TINT", UI_MENU_COLOR_TINT, -100, 100, UI_MENU_COLOR_TINT_DESC, 5, "%+.0f %%");\n'
              '    moved |= opt_slider("TOTK_COLOR_FOG", UI_MENU_COLOR_FOG, 0, 100, UI_MENU_COLOR_FOG_DESC, 5, "%.0f %%");')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched')
