import re

# cfg: SSSR as a real setting (0..200, live)
p = 'runtime/cfg.c'
s = open(p, encoding='utf-8').read()
s = s.replace('"SHADOW_RES", "SHADOW_SCALE",', '"SHADOW_RES", "SHADOW_SCALE", "SSSR",', 1)
s = s.replace('{"SHADOW_SCALE", "display"},', '{"SHADOW_SCALE", "display"}, {"SSSR", "display"},', 1)
s = s.replace('{"GTAO", "0", "0..100", RESTART},', '{"GTAO", "0", "0..100", RESTART}, {"SSSR", "0", "0..200", LIVE},', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# vk_sssr.cpp: read the strength again when the settings change (the menu's row is live)
p = 'runtime/vk_sssr.cpp'
s = open(p, encoding='utf-8').read()
s = s.replace('''extern "C" int vk_sssr_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *sv = cfg_get("TOTK_SSSR");   // (no settings row: the environment or totk.ini, as TOTK_RT)
        int s = sv && *sv ? atoi(sv) : 0;
        v = s < 0 ? 0 : s > 200 ? 200 : s;
    }
    return v;
}''','''extern "C" int vk_sssr_on(void) {
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
}''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# text ids
p = 'runtime/ui_text.h'
s = open(p, encoding='utf-8').read()
s = s.replace('UI_MENU_SHADOW_SCALE, UI_MENU_SHADOW_SCALE_DESC,',
              'UI_MENU_SHADOW_SCALE, UI_MENU_SHADOW_SCALE_DESC, UI_MENU_SSSR, UI_MENU_SSSR_DESC,', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)

p = 'runtime/ui_text.c'
s = open(p, encoding='utf-8').read()
s = s.replace('''    [UI_MENU_COLOR_TINT] = "Tint",''',
              '''    [UI_MENU_SSSR] = "Reflections",
    [UI_MENU_SSSR_DESC] = "Screen-space reflections from AMD's FidelityFX SSSR at the scene's composite: wet ground, "
                          "stone, metal and glass reflect what is on screen. Off leaves the game's own look. Needs "
                          "ray tracing on (rt = bvh) in totk.ini.",
    [UI_MENU_COLOR_TINT] = "Tint",''', 1)
s = s.replace('''    [UI_MENU_COLOR_TINT] = "色调微调",''',
              '''    [UI_MENU_SSSR] = "反射",
    [UI_MENU_SSSR_DESC] = "在场景合成点用 AMD FidelityFX SSSR 做屏幕空间反射：湿地面、石头、金属和琉璃会映出屏幕上已有的景物。"
                          "关闭即游戏原样。需要 totk.ini 里开启光追（rt = bvh）。",
    [UI_MENU_COLOR_TINT] = "色调微调",''', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)

# the menu row, on the graphics page next to the shadows
p = 'runtime/ui/menu.cpp'
s = open(p, encoding='utf-8').read()
s = s.replace('''static void row_grass_range(void) {''','''static void row_sssr(void) {
    static const char *const vals[] = {"0", "50", "100", "150", "200"};
    const char *names[] = {menu_text(UI_MENU_OFF), "50 %", "100 %", "150 %", "200 %"};
    opt_choice_list("TOTK_SSSR", UI_MENU_SSSR, vals, names, 5, UI_MENU_SSSR_DESC);
}

static void row_grass_range(void) {''', 1)
s = s.replace('    row_grass_range();', '    row_grass_range();\n    row_sssr();', 1)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ok')
