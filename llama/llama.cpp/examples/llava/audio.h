#ifndef AUDIO_H
#define AUDIO_H

#include <stddef.h>
#include <stdint.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define AUDIO_API __declspec(dllexport)
#        else
#            define AUDIO_API __declspec(dllimport)
#        endif
#    else
#        define AUDIO_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define AUDIO_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct audio_u8;
struct audio_f32;
struct extractor_config;
struct audio_ctx;

AUDIO_API bool feature_extract_f(struct extractor_config config,
                     const char * fname,
                     struct audio_f32* features,
                     int n_output);

AUDIO_API size_t audio_embd_nbytes();
AUDIO_API struct audio_ctx* audio_ctx_init(const char * fname);

AUDIO_API void audio_ctx_free(struct audio_ctx* ctx);

AUDIO_API bool audio_wav_preprocess(struct audio_ctx* ctx, const struct audio_u8 * aud, struct audio_f32 * res_auds, int n_output);

// AUDIO_API bool audio_wav_preprocess_file(struct audio_ctx* ctx, std::string fname, audio_f32* res_auds);
AUDIO_API bool audio_encode(struct audio_ctx * ctx, const int n_threads, struct audio_f32 * aud, struct audio_f32 * ret);

#ifdef __cplusplus
}
#endif
#endif // AUDIO_H
