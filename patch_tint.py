import re

# 1. cfg.c: the key, its group and its row
p = 'runtime/cfg.c'
s = open(p, encoding='utf-8').read()
s = s.replace('"COLOR_PRESET", "COLOR_SATURATION",', '"COLOR_PRESET", "COLOR_SATURATION", "COLOR_TINT",', 1)
s = s.replace('{"COLOR_GAMMA", "display"},', '{"COLOR_GAMMA", "display"}, {"COLOR_TINT", "display"},', 1)
s = s.replace('{"COLOR_GAMMA", "1", "0.8..1.2", LIVE},', '{"COLOR_GAMMA", "1", "0.8..1.2", LIVE},\n    {"COLOR_TINT", "0", "-100..100", LIVE},', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# 2. vk_color.c: the fifth value
p = 'runtime/vk_color.c'
s = open(p, encoding='utf-8').read()
s = s.replace('    float adj[4];                  // brightness (added), contrast, saturation, gamma',
              '    float adj[5];                  // brightness (added), contrast, saturation, gamma, green/magenta tint')
s = s.replace('        C.adj[3] = (float)cfg_double("TOTK_COLOR_GAMMA");',
              '        C.adj[3] = (float)cfg_double("TOTK_COLOR_GAMMA");\n        C.adj[4] = (float)cfg_double("TOTK_COLOR_TINT") / 100.0f;')
s = s.replace('    return C.adj[0] != 0.0f || C.adj[1] != 1.0f || C.adj[2] != 1.0f || C.adj[3] != 1.0f;',
              '    return C.adj[0] != 0.0f || C.adj[1] != 1.0f || C.adj[2] != 1.0f || C.adj[3] != 1.0f || C.adj[4] != 0.0f;')
s = s.replace('    struct { int32_t offset[2], size[2]; float adj[4]; } pc = {{crop[0], crop[1]}, {crop[2], crop[3]},\n'
              '                                                              {C.adj[0], C.adj[1], C.adj[2], C.adj[3]}};',
              '    struct { int32_t offset[2], size[2]; float adj[4], tint[4]; } pc = {\n'
              '        {crop[0], crop[1]}, {crop[2], crop[3]}, {C.adj[0], C.adj[1], C.adj[2], C.adj[3]}, {C.adj[4], 0, 0, 0}};')
s = s.replace('VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};',
              'VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 48};')
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# 3. the shader
p = 'runtime/shaders/color_grade.comp'
s = open(p, encoding='utf-8').read()
s = s.replace('    vec4 adj;       // brightness (added), contrast, saturation, gamma (exponent)',
              '    vec4 adj;       // brightness (added), contrast, saturation, gamma (exponent)\n'
              '    vec4 tint;      // x: green/magenta, -1 (magenta) .. 1 (green), 0 leaves it')
s = s.replace('''    c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, pc.adj.z);''',
              '''    c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, pc.adj.z);
    // The green/magenta axis, the one a scene's haze tips without the other two: at +1 the green goes up 15 % and red
    // and blue down 7.5 % (a greener picture), at -1 the other way.
    float t = clamp(pc.tint.x, -1.0, 1.0);
    c *= vec3(1.0 - 0.075 * t, 1.0 + 0.15 * t, 1.0 - 0.075 * t);''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# 4. the text ids
p = 'runtime/ui_text.h'
s = open(p, encoding='utf-8').read()
s = s.replace('UI_MENU_COLOR_GAMMA, UI_MENU_COLOR_GAMMA_DESC,',
              'UI_MENU_COLOR_GAMMA, UI_MENU_COLOR_GAMMA_DESC, UI_MENU_COLOR_TINT, UI_MENU_COLOR_TINT_DESC,', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# 5. English and Chinese strings
p = 'runtime/ui_text.c'
s = open(p, encoding='utf-8').read()
s = s.replace('''    [UI_MENU_COLOR_GAMMA_DESC] = "Below 1 brightens dark and middle tones, above 1 darkens them; 1 leaves them.",''',
              '''    [UI_MENU_COLOR_GAMMA_DESC] = "Below 1 brightens dark and middle tones, above 1 darkens them; 1 leaves them.",
    [UI_MENU_COLOR_TINT] = "Tint",
    [UI_MENU_COLOR_TINT_DESC] = "Shifts the picture between magenta (below 0) and green (above 0); 0 leaves it.",''', 1)
s = s.replace('''    [UI_MENU_COLOR_GAMMA_DESC] = "小于 1 提亮暗部与中间调，大于 1 压暗；1 表示不调整。",''',
              '''    [UI_MENU_COLOR_GAMMA_DESC] = "小于 1 提亮暗部与中间调，大于 1 压暗；1 表示不调整。",
    [UI_MENU_COLOR_TINT] = "色调微调",
    [UI_MENU_COLOR_TINT_DESC] = "在偏洋红（小于 0）与偏绿（大于 0）之间调整画面；0 表示不调整。",''', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# 6. the menu row
p = 'runtime/ui/menu.cpp'
s = open(p, encoding='utf-8').read()
s = s.replace('''    moved |= opt_slider("TOTK_COLOR_GAMMA", UI_MENU_COLOR_GAMMA, 0.8f, 1.2f, UI_MENU_COLOR_GAMMA_DESC, 0.02f, "%.2f");''',
              '''    moved |= opt_slider("TOTK_COLOR_GAMMA", UI_MENU_COLOR_GAMMA, 0.8f, 1.2f, UI_MENU_COLOR_GAMMA_DESC, 0.02f, "%.2f");
    moved |= opt_slider("TOTK_COLOR_TINT", UI_MENU_COLOR_TINT, -100, 100, UI_MENU_COLOR_TINT_DESC, 5, "%+.0f %%");''', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched')
