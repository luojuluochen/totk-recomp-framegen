// FidelityFX SSSR (third_party/FidelityFX, MIT) on the port's own Vulkan device: the reflection pass over the scene's
// colour, depth and motion vectors at the point vk_dlaa.c has them (before DLSS/FSR 3), and its result added back into
// the scene colour there.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// TOTK_SSSR: 0 off (the default), 1..100 the strength of the reflections added to the scene.
int vk_sssr_on(void);

// The SDK's shader blobs (vk_fg.cpp ffxGetPermutationBlobByIndex): SSSR's passes and its reflections denoiser's.
int vk_sssr_blob(uint32_t effect, uint32_t pass, uint32_t options, void *out);

// The inputs (normal, roughness) from the port's linear depth at the scene's composite: cb records; 0: not made.
int vk_sssr_prepare(void *cb, uint64_t lin_depth, uint64_t lit, uint32_t w, uint32_t h, float proj_x, float proj_y,
                    float far);

// One frame: cb records; scene is the lit colour (B10G11R11) mixed in place; depth the composite's D32; mv the motion
// vectors (RG16F, pixels, as DLSS reads them). cam: the camera block's first 11 rows (view 0-2, view-projection 3-6,
// projection 7-10, row-major 4-wide), which vk_dlaa.c already reads for the motion vectors. 0: not dispatched.
int vk_sssr_exec(void *cb, uint64_t scene, uint64_t depth, uint64_t mv, uint32_t w, uint32_t h, const float *cam,
                 uint64_t fence);

void vk_sssr_release(void);

#ifdef __cplusplus
}
#endif
