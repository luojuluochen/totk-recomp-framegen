// First-run defaults chosen by the GPU's VRAM (runtime/auto_preset.c): DLSS mode, frame generation, sun shadow
// resolution and contact shadows are the rows a player would otherwise have to guess on a new machine, and the wrong
// ones either exceed a small card's memory or leave a big one half idle. A key the player's totk.ini (or the
// environment) already sets is left alone, so this only fills in what was never chosen. AUTO_PRESET = 0 turns it off.
#include "cfg.h"
#include "host.h"
#include "vk.h"

void cfg_auto_preset(VkPhysicalDevice phys) {
    const char *off = cfg_get("TOTK_AUTO_PRESET");
    if (off && *off && !strcmp(off, "0")) return;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint64_t vram = 0;
    for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
        if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram += mp.memoryHeaps[h].size;
    double gb = (double)vram / (1024.0 * 1024.0 * 1024.0);
    // < 10 GB: the game's texture set only, DLSS quality (the scene at 1440p for a 4K output), no frame generation.
    // 10..15 GB: DLSS quality with 2x frame generation.
    // >= 15 GB: native 4K (DLAA) with 2x frame generation and 2x sun shadow maps.
    const char *dlss = gb >= 15.0 ? "dlaa" : "quality";
    const char *fg = gb >= 10.0 ? "1" : "0";
    const char *fgx = "2";
    const char *shadow = gb >= 15.0 ? "2" : "1";
    const char *contact = gb >= 15.0 ? "30" : "20";
    int set = 0;
    if (!cfg_get("TOTK_DLSS")) cfg_set("TOTK_DLSS", dlss), set++;
    if (!cfg_get("TOTK_DLSS_FG")) cfg_set("TOTK_DLSS_FG", fg), set++;
    if (!cfg_get("TOTK_DLSS_FG_X")) cfg_set("TOTK_DLSS_FG_X", fgx), set++;
    if (!cfg_get("TOTK_SHADOW_SCALE")) cfg_set("TOTK_SHADOW_SCALE", shadow), set++;
    if (!cfg_get("TOTK_CONTACT_SHADOWS")) cfg_set("TOTK_CONTACT_SHADOWS", contact), set++;
    if (set)
        rt_log("auto: %.1f GB of VRAM: preset %s, frame generation %s x%s, shadow maps x%s, contact shadows %s%% "
               "(%d keys that totk.ini did not set)", gb, dlss, fg, fgx, shadow, contact, set);
}
