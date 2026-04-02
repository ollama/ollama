#pragma once

#include "mtmd.h"

#include <vector>

// INTERNAL HEADER FOR DEBUGGING PURPOSES ONLY
// NOT INTENDED FOR PUBLIC USE
// Do not raise issues related to this debugging API

// encode take the pre-processed f32 values, print the intermidiate values via cb_eval callback
MTMD_API void mtmd_debug_encode_image(mtmd_context * ctx, const std::vector<std::vector<float>> & image);
MTMD_API void mtmd_debug_encode_audio(mtmd_context * ctx, const std::vector<float> & input); // will be broadcasted to fit n_mel

// preprocess take the raw input values
MTMD_API void mtmd_debug_preprocess_image(mtmd_context * ctx, const std::vector<uint8_t> & rgb_values, int nx, int ny);
MTMD_API void mtmd_debug_preprocess_audio(mtmd_context * ctx, const std::vector<float> & pcm_samples);
