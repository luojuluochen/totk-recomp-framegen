# TotK Recomp

**TotK Recomp** is a static recompilation of *The Legend of Zelda: Tears of the Kingdom* (Nintendo Switch, 2023) into a native x86-64 application for Windows and Linux ([docs/architecture.md](docs/architecture.md)).

> [!IMPORTANT]
> **No game files are included.** You need your own legally obtained copy of *Tears of the Kingdom*, dumped from your own console: the base game and the 1.1.0, 1.2.1 or 1.4.3 update. Instructions for obtaining those files will not be provided.
>
> The game is built from your files on your machine, so the result contains the game. It is for your own use. Do not share it.

## This fork: frame-rate work

This fork (luojuluochen/totk-pc-recomp, main) is a derivative of upstream TotK Recomp whose changes aim at the frame
rate on high-refresh displays: NVIDIA DLSS Frame Generation on top of the current main branch, the stability fixes the
unmerged draft needed to run for hours, and a frame limit that the frame generator cannot overrun. The frame generation
integration comes from the draft in upstream issue #1069 (al4xdev, MIT); the fixes and the overlay change are ours.

### Frame-rate changes

1. **DLSS Frame Generation (DLSS-G)**, dlss_fg and dlss_fg_x (2 to 6 pictures for each frame the game draws), for
   RTX 40 series and newer with nvngx_dlssg.dll next to totk.exe. It runs with the DLSS scene upscaler; DLSS-G is
   off while the port's own FSR 3 generator (frame_gen) is on, and once the game is played the multiplier can be
   changed in the menu (F1) or in totk.ini.
2. **The feature is rebuilt when the internal scene size changes.** The NGX feature keeps the internal render size it
   was created with. Dynamic resolution or a teleport changed the render size while the feature kept the old one, and
   the game ended with VK_ERROR_DEVICE_LOST (vk: device fault address 0: write invalid at 0) within minutes; now a
   size change releases the feature and creates it again at the new size.
3. **Generated images are released only after the present thread is done with them.** A multiplier or swapchain change
   recreated the generated images while the present thread could still be showing the previous ones; the release now
   waits for the present thread (vk_present_drain), which removed the second kind of device-lost crash.
4. **The frame limit is divided by the multiplier** (runtime/frame_rate.c). fps_limit counts the pictures shown, so
   the game draws fps_limit / dlss_fg_x and the generated pictures fill the rest: 4x on a 240 Hz display draws 60 and
   shows 240. The draft divided by the pictures of the last present instead, so every frame without a generated picture
   lifted the limit back to the full rate and the generator over-produced.
5. **The full performance overlay shows the frame-rate line too.** The first F10 press shows the drawn rate and, with
   frame generation, the pictures shown a second; the second press (the full panel) now shows the same line, so the
   drawn rate is not lost when the panel is open.
6. **Recommended combination on an RTX 5070 Ti, a Ryzen 7 9700X and a 240 Hz 4K display** (the full settings used on
   the test machine are below):

   ```ini
   [display]
   scene_upscaler = dlss
   dlss = quality          # 2560x1440 -> 4K
   dlss_fg = 1
   dlss_fg_x = 3           # 80 drawn / 240 shown
   dynamic_res = 0

   [frame rate]
   fps_limit = display
   sync_ahead = 1
   replay_workers = 4
   async_preempt = 2
   ```

   About 75-80 drawn (220-240 shown) in the lighter areas and 50-60 drawn in heavy ones. replay_workers = 6 and
   frame_queue = 1 measured lower frame rates here; dlaa_objmv = 0 and lower shadow settings are the remaining
   CPU/GPU levers (see "Known issue" below for what they cost in picture quality).

### Complete settings used on the test machine

The settings the numbers below were measured with, as they were played (RTX 5070 Ti, Ryzen 7 9700X, 3840x2160 at
240 Hz). trace and perf_slow_ms are diagnostics and can be left out for normal play.

```ini
[frame rate]
fps_limit = display        ; 240 shown, 80 drawn with 3x, 60 with 4x
replay_workers = 4
sync_ahead = 1

[display]
window_size = 3840x2160
fullscreen = 1
output_res = 3840x2160
render_scale = 100
dynamic_res = 0
aniso = 16
grass_range = 200
scene_upscaler = dlss
dlss = dlaa                ; quality when the higher frame rate matters more
dlss_fg = 1
dlss_fg_x = 4
shadow_res = 2048
shadow_scale = 2
shadow_range = 100
contact_shadows = 70
pcss = 70
perf_overlay = full

[diagnostics]
trace = perf
perf_slow_ms = 25

[language]
lang = zh-Hans
menu_lang = zh-Hans
```

### Known issue: whole-picture flicker with frame generation

1. **Symptom.** With dlss_fg on, the whole picture flickers (the HUD and the game's own image together) in some areas.
   The amplitude grows with dlss_fg_x: 2x is mild, 4x and 6x are clearly visible.
2. **Where and how to reproduce.** Just outside the north gate of Lookout Landing, at about X -0231, Z 0020 (12 m
   above sea level): the spot by the gate's wooden palisade and the lone tree beside it, with the fort wall, the banner
   and the tree in the frame (the map view at X -0224, Z 0019 marks the same place). Stand still with the camera still
   and watch the picture; it flickers within a second or two. The same view with frame generation off (menu, F1) is
   steady. High-contrast spots elsewhere in the world show it too; most of the map does not.
3. **Workarounds.** Keep the camera moving (the motion masks the flicker), turn frame generation off in the menu (F1)
   while playing in such an area, or use dlss_fg_x = 2, which makes it much less noticeable. Turning the scene upscaler
   off also stops it, but that also disables DLSS-G, which needs the upscaler.

### 中文说明

这个 fork 基于上游 TotK Recomp，主要解决高刷新率下的帧率问题：把未合并的 DLSS 帧生成草案移植到最新 main，并修复了它
跑几分钟就会 VK_ERROR_DEVICE_LOST 的两处问题（场景尺寸变化时重建 NGX 特征、释放生成图前排空呈现线程），同时让帧上限
按倍率计算、完整性能面板也显示绘制/显示帧率。已知问题是部分高对比区域（如监视堡垒周边）开启帧生成后整屏闪烁，倍率越高
越明显，原因和已尝试的修法见上面的 "Known issue"。

## Status

Early development. The game is playable on Windows and Linux; visual bugs and crashes are still expected.

Report problems as issues, following [CONTRIBUTING.md](CONTRIBUTING.md).

## Features

- Native code, not emulation
- Game updates 1.1.0, 1.2.1 and 1.4.3; 1.2.1 and 1.4.3 saves carry over between them
- Takes NSP and XCI game files, and compressed NSZ and XCZ as they are: the setup unpacks them itself
- 60 fps by default; up to 144 fps or the monitor's refresh rate (experimental)
- Custom resolutions (1080p, 2k...), supersampling and AMD FSR 1 upscaling
- NVIDIA DLSS / DLAA on RTX GPUs: the setup downloads nvngx_dlss.dll from NVIDIA's DLSS SDK repository
  (github.com/NVIDIA/DLSS, NVIDIA RTX SDK license) when it finds an RTX GPU; untick it in the setup to skip it
- HDR10 output with peak, game and HUD brightness settings
- 16:10 (Steam Deck, handhelds), 3:2 (Surface, Framework), 21:9 and 32:9 (ultrawide) game resolutions, with the camera and the interface in that shape
- Anisotropic texture filtering up to 16x, for sharp ground into the distance
- In-game settings menu (F1) and performance overlay (F10)
- Play time across runs on the Game tab of the settings menu, kept in playtime.txt next to the saves
- Discord Rich Presence ("Playing TotK Recomp" with the play time), off by default, own application id supported
- Setup in 11 languages, in-game menu in 4 ([Languages](#languages))
- Keyboard, mouse and gamepad with remapping, rumble and gyro
- Stereo or 5.1 surround sound output
- Xbox and PlayStation button prompts: Xbox's LB, RB, LT, RT, Menu, View and A, B, X, Y, or PlayStation's ○ × △ □, L1, R1, L2, R2, Options and Share, as labelled on the button to press, with either button layout
- Shaders prepared at install instead of during play
- Mods made for emulators: unpack a mod into the mods folder next to totk.exe; switch and order them in the menu. RomFS
  mods, TKMM's merged output and ExeFS code patches (.ips, .pchtxt) work; the game rebuilds itself for code patches
  ([docs/mods.md](docs/mods.md)). HD texture packs work: the port grows the texture heap a pack asks for by itself. Tested
  with the HDPacks.com 4K Texture Pack v1.0 on 1.4.3 and 1.2.1; other packs come with no guarantee
- Atmosphère cheats: put a cheat file into a mod and switch its cheats in the menu ([docs/cheats.md](docs/cheats.md))

## Languages

The game keeps the text and voices of every language it shipped with. The port's own setup and in-game menu (F1) come
in these; both follow the Windows display language, and the menu has its own choice on its Menu tab. A Traditional
Chinese Windows gets the Simplified Chinese setup.

| Language | Setup | In-game menu |
|---|:---:|:---:|
| English | ✅ | ✅ |
| Русский | ✅ | ✅ |
| Português (Brasil) | ✅ | ✅ |
| 简体中文 | ✅ | ✅ |
| Deutsch | ✅ | English |
| Français | ✅ | English |
| Español | ✅ | English |
| Italiano | ✅ | English |
| Nederlands | ✅ | English |
| 日本語 | ✅ | English |
| 한국어 | ✅ | English |

Spotted a clumsy phrase, or want your language added? The setup texts are in installer/text/<language>.json, the
menu texts in runtime/ui_text.c; open an issue or a pull request ([CONTRIBUTING.md](CONTRIBUTING.md)). Fixes from
native speakers are welcome.

## Planned

- VRR

## Installing

### Windows

Requirements: Windows 10 or later, an x86-64 CPU with SSE4.2 (AVX2/FMA machines get a faster build), a GPU with Vulkan 1.3 and VK_EXT_external_memory_host, about 36 GiB free during setup (20 GiB after).

1. Download TotK-Setup-<version>-windows.zip from [Releases](../../releases/latest) and unpack it.
2. Start TotK-Setup.exe. If SmartScreen warns about an unsigned app, choose *More info*, then *Run anyway*.
3. Pick your game files, the language and an install folder, and press *Start*. The setup builds the game once. With
   an NVIDIA RTX GPU it also downloads NVIDIA DLSS, unless you untick *Download NVIDIA DLSS*; turn DLSS on in the
   menu (F1).
4. Play totk.exe from the install folder.

### Linux

Requirements: as on Windows, plus Linux 6.7 or later and glibc 2.34 or later (Ubuntu 22.04, Debian 12, Fedora 35, SteamOS 3).

1. Download TotK-Setup-x86_64.AppImage from [Releases](../../releases/latest), or build it ([docs/installer.md](docs/installer.md), "Linux release").
2. Make it executable (chmod +x TotK-Setup-x86_64.AppImage) and start it. On a Steam Deck, do this in Desktop Mode.
3. Pick your game files, the language and an install folder, and press *Start*.
4. Play totk from the install folder.

To update, run a newer setup on the same folder; saves and settings are kept. Settings are in the in-game menu and in totk.ini ([totk.ini.example](totk.ini.example) lists every key). Controls: [docs/controls.md](docs/controls.md).

## Credits

TotK Recomp builds on the projects below. What was taken from each, with its license, is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- [LAS Recomp](https://github.com/peiikai/LASRecomp)
- [MacNativeGames](https://github.com/ly05010419/MacNativeGames)
- [d2r-mobile](https://github.com/jhaizhou-ops/d2r-mobile) and [nxnative](https://github.com/seedds/nxnative)
- [SavageWindRecomp](https://github.com/Revivedx/SavageWindRecomp)
- [Ryujinx](https://github.com/Ryujinx/Ryujinx)
- [deko3d](https://github.com/devkitPro/deko3d)
- [hactool](https://github.com/SciresM/hactool)
- [SDL 3](https://libsdl.org), [Dear ImGui](https://github.com/ocornut/imgui), [AMD FSR 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR), [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator), [libopus](https://opus-codec.org), [libvpx](https://chromium.googlesource.com/webm/libvpx), [Zstandard](https://github.com/facebook/zstd)
- [Switchbrew](https://switchbrew.org) and the Switch homebrew community

## License

The project's own code is under the [MIT License](LICENSE); third-party code keeps its own license ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). The license grants nothing for Nintendo's game.

TotK Recomp is an independent, non-commercial fan project, not affiliated with or endorsed by Nintendo. *The Legend of Zelda*, *Tears of the Kingdom* and *Nintendo Switch* are trademarks of Nintendo.

## Building from source

[docs/building.md](docs/building.md) lists the toolchain and the steps. Everything the build writes stays under build\ and generated\, which git ignores; run tools/check_content.py before a commit. Testers: [docs/tester-guide.md](docs/tester-guide.md).

Most of the code and documentation is written by AI coding agents (Claude Code) under the maintainer's direction; the maintainer plays every build and decides what is merged. [AGENTS.md](AGENTS.md) holds the rules they work under.
