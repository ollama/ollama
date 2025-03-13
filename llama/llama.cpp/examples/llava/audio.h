#ifndef AUDIO_H
#define AUDIO_H

#include <string>
#include <vector>

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

struct audio_ctx;

struct audio_u8 {
    std::vector<uint8_t> buf;
};

struct audio_f32 {
    int n_mel;
    int n_len;
    std::vector<float> buf;
};

struct audio_u8_batch {
    struct audio_u8 * data;
    size_t size;
};

struct audio_f32_batch {
    struct audio_f32 * data;
    size_t size;
};

// samples_per_sec + N_FFT - HOP_LENGTH
//static constexpr int FIXED_INPUT_SIZE = 16240;
static constexpr int FIXED_INPUT_SIZE = 480000;
//static constexpr int FIXED_SHAPE_SIZE = 80 * 102;
static constexpr int FIXED_SHAPE_SIZE = 80 * 735;
static constexpr int OTHER_FIXED_SHAPE_SIZE = 80 * 102;

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft = 400;
    std::vector<float> data;
};

struct extractor_config {
    int32_t feature_size = 80;
    int32_t n_samples = 480000;
    int32_t sampling_rate = 16000;
    int32_t frame_size = 400;
    int32_t hop_length = 160;
    int32_t chunk_length = 30;
    float padding_value = 0.0;
    whisper_filters mel_filters;

    extractor_config(){
        init();
    }
    void init() {
        int num_frequency_bins = frame_size / 2 + 1;
        mel_filters.n_mel = feature_size;
        mel_filters.n_fft = num_frequency_bins;
        mel_filters.data.resize(num_frequency_bins * feature_size);
        compute_mel_filter_bank(true);
    }

    // 从 JSON 配置文件中初始化配置
    bool init_from_file(const std::string& file_path);

    void compute_mel_filter_bank(bool norm_slaney);
};

AUDIO_API bool feature_extract_f(extractor_config config,
                     const std::string& fname,
                     audio_f32* features,
                     int n_output = FIXED_SHAPE_SIZE);

AUDIO_API bool feature_extract_v(extractor_config config,
                     const std::vector<float>& pcmf32,
                     audio_f32* features,
                     int n_output = FIXED_SHAPE_SIZE);

AUDIO_API size_t audio_embd_nbytes();
AUDIO_API struct audio_ctx* audio_ctx_init(const std::string& model_path);

AUDIO_API void audio_ctx_free(audio_ctx* ctx);

// for debug only

AUDIO_API bool audio_wav_preprocess(struct audio_ctx* ctx, const audio_u8 * aud, audio_f32* res_auds, int n_output);

// AUDIO_API bool audio_wav_preprocess_file(struct audio_ctx* ctx, std::string fname, audio_f32* res_auds);
AUDIO_API bool audio_encode(struct audio_ctx * ctx, const int n_threads, audio_f32 * aud, audio_f32& ret);

#ifdef __cplusplus
}
#endif
#endif // AUDIO_H
