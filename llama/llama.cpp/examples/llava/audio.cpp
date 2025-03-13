#include "audio.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cinttypes>
#include <limits>
#include <thread>

#include <algorithm>

#include "utils/audio_common.h"
#include <sys/stat.h>
#include "json.hpp"


#define LOG_INF(...) do { fprintf(stdout, __VA_ARGS__); } while (0)
#define LOG_WRN(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_ERR(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_DBG(...) do { fprintf(stderr, __VA_ARGS__); } while (0)

struct audio_hparams {
    int32_t n_vocab       = 51864;
    int32_t n_audio_ctx   = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head  = 6;
    int32_t n_audio_layer = 4;
    int32_t n_text_ctx    = 448;
    int32_t n_text_state  = 384;
    int32_t n_text_head   = 6;
    int32_t n_text_layer  = 4;
    int32_t n_mels        = 80;
    int32_t ftype         = 1;
    float   eps           = 1e-5f;
};

struct audio_layer {
    // self attention 部分
    struct ggml_tensor * self_attn_k_proj_w;
    struct ggml_tensor * self_attn_q_proj_w;
    struct ggml_tensor * self_attn_v_proj_w;
    struct ggml_tensor * self_attn_out_proj_w;
    struct ggml_tensor * self_attn_q_proj_b;
    struct ggml_tensor * self_attn_v_proj_b;
    struct ggml_tensor * self_attn_out_proj_b;

    // 自注意力层的 LayerNorm
    struct ggml_tensor * self_attn_layer_norm_w;
    struct ggml_tensor * self_attn_layer_norm_b;

    // MLP 部分
    struct ggml_tensor * fc1_w;
    struct ggml_tensor * fc1_b;
    struct ggml_tensor * fc2_w;
    struct ggml_tensor * fc2_b;

    // 最后的 LayerNorm
    struct ggml_tensor * final_layer_norm_w;
    struct ggml_tensor * final_layer_norm_b;
};

struct audio_model {
    // 卷积层
    struct ggml_tensor * conv1_w;
    struct ggml_tensor * conv1_b;
    struct ggml_tensor * conv2_w;
    struct ggml_tensor * conv2_b;

    // 位置嵌入
    struct ggml_tensor * embed_positions;

    // LayerNorm
    struct ggml_tensor * layer_norm_w;
    struct ggml_tensor * layer_norm_b;

    // 编码器层列表
    std::vector<audio_layer> layers;

    // 音频投影层
    struct ggml_tensor * audio_proj_layer_linear1_w;
    struct ggml_tensor * audio_proj_layer_linear1_b;
    struct ggml_tensor * audio_proj_layer_linear2_w;
    struct ggml_tensor * audio_proj_layer_linear2_b;

    struct audio_hparams hparams;

    ggml_backend_buffer_t buffer = nullptr;
};

struct audio_encoder_config {
    ggml_backend_t backend = NULL;
    ggml_gallocr_t compute_alloc = NULL;

    // struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_data;
    ggml_backend_buffer_t params_buffer  = NULL;
};

struct audio_ctx {
    struct extractor_config config; // defined in audio_preprocessor.h
    struct audio_encoder_config encoder_config;
    struct audio_model model;
    bool npu_use_ane = false;

    std::vector<uint8_t> buf_compute_meta;

    ggml_backend_t backend       = NULL;
    ggml_gallocr_t compute_alloc = NULL;
};

// 打印 tensor 的形状信息
static void print_tensor_shape(struct ggml_tensor *tensor) {
    printf("tensor(");
    int n_dims = 0;
    // 首先计算实际的维度数
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] > 1 || i == 0) {  // 包含第一维即使是 1
            n_dims = i + 1;
        }
    }
    // 打印所有维度
    for (int i = 0; i < n_dims; i++) {
        printf("%lld", tensor->ne[i]);
        if (i < n_dims - 1) {
            printf(", ");
        }
    }
    printf(")\n");
}

static void print_tensor(struct ggml_tensor *tensor) {
    print_tensor_shape(tensor);
    
    // 计算实际维度数
    int n_dims = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] > 1 || i == 0) {
            n_dims = i + 1;
        }
    }

    int num_display = 3; // 每个维度显示的头尾元素数
    float* data = (float*)tensor->data;

    if (n_dims == 1) {
        // 1 维张量打印
        for (int i = 0; i < num_display && i < tensor->ne[0]; ++i) {
            printf("%f ", data[i]);
        }
        if (tensor->ne[0] > 2 * num_display) {
            printf("... ");
        }
        for (int i = std::max(num_display, (int)tensor->ne[0] - num_display); 
             i < tensor->ne[0]; ++i) {
            printf("%f ", data[i]);
        }
        printf("\n");
    }
    else if (n_dims == 2) {
        // 2 维张量打印
        int cols = tensor->ne[0]; // 列数
        int rows = tensor->ne[1]; // 行数
        
        // 打印前几行
        for (int i = 0; i < num_display && i < rows; ++i) {
            for (int j = 0; j < num_display && j < cols; ++j) {
                printf("%f ", data[i * cols + j]); // 行优先顺序
            }
            if (cols > 2 * num_display) {
                printf("... ");
            }
            for (int j = std::max(num_display, cols - num_display); j < cols; ++j) {
                printf("%f ", data[i * cols + j]); // 行优先顺序
            }
            printf("\n");
        }

        // 如果有更多行，打印省略号
        if (rows > 2 * num_display) {
            printf("...\n");
        }

        // 打印最后几行
        for (int i = std::max(num_display, rows - num_display); i < rows; ++i) {
            for (int j = 0; j < num_display && j < cols; ++j) {
                printf("%f ", data[i * cols + j]); // 行优先顺序
            }
            if (cols > 2 * num_display) {
                printf("... ");
            }
            for (int j = std::max(num_display, cols - num_display); j < cols; ++j) {
                printf("%f ", data[i * cols + j]); // 行优先顺序
            }
            printf("\n");
        }
    }
    else {
        printf("Tensor has %d dimensions, only support up to 2 dimensions for display\n", n_dims);
    }
}

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

// Hertz to Mel conversion
static float hz2mel_htk(float f) {
    return 2595*std::log10 (1+f/700);
}

// Mel to Hertz conversion
//double mel2hz (double m) {
//    return 700*(std::pow(10,m/2595)-1);
//}

static float hz2mel_slaney(const float freq) {
    static const float min_log_hertz = 1000.0;
    static const float min_log_mel = 15.0;
    static const float logstep = 27.0 / std::log(6.4);

    float mels = 3.0 * freq / 200.0;

    if (freq >= min_log_hertz) {
        mels = min_log_mel + std::log(freq / min_log_hertz) * logstep;
    }

    return mels;
}

static float mel2hz_slaney(float mels) {
    static const double min_log_hertz = 1000.0;
    static const double min_log_mel = 15.0;
    static const double logstep = std::log(6.4) / 27.0;

    float freq = 200.0 * mels / 3.0;

    if (mels >= min_log_mel) {
        freq = min_log_hertz * std::exp(logstep * (mels - min_log_mel));
    }
    return freq;
}

static std::vector<float> linspace(float start, float stop, int num) {
    std::vector<float> result(num);
    float step = (stop - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + i * step;
    }
    return result;
}

// 从 JSON 配置文件中初始化配置
bool extractor_config::init_from_file(const std::string& file_path) {
    struct stat statbuf;
    if (stat(file_path.c_str(), &statbuf) != 0) {
        return false;
    }
    std::string config_file;
    if (S_ISDIR(statbuf.st_mode)) {
        config_file = file_path + "/preprocessor_config.json";
    } else if (S_ISREG(statbuf.st_mode)) {
        config_file = file_path;
    }

    std::ifstream file(config_file);
    if (!file.is_open()) {
        return false;
    }

    nlohmann::json config_json;
    try {
        file >> config_json;
        if (file.fail()) {
            std::cerr << __func__ << " failed to read JSON from file." << std::endl;
            return false;
        }
        //if (file.peek() != EOF) {
        //    std::cerr << __func__ << " trailing bytes after read." << std::endl;
        //    return false;
        //}
    } catch (const nlohmann::json::parse_error& e) {
        auto error = std::string("JSON parse error: ") + e.what();
        std::cerr << __func__ << error << std::endl;
        return false;
    } catch(const nlohmann::json::exception& e) {
        auto error = std::string("other json error: ") + e.what();
        std::cerr << __func__ << error << std::endl;
        return false;
    }
    file.close();

    feature_size = config_json["feature_size"];
    sampling_rate = config_json["sampling_rate"];
    n_samples = config_json["n_samples"];
    hop_length = config_json["hop_length"];
    chunk_length = config_json["chunk_length"];
    padding_value = config_json["padding_value"];
    frame_size = config_json["n_fft"];

    // 解析 mel_filters
    // mel_filters data size = (feature_size * num_frequency_bins).
    int num_frequency_bins = frame_size / 2 + 1;
    mel_filters.n_mel = feature_size;
    mel_filters.n_fft = num_frequency_bins;
    if (config_json.contains("mel_filters")) {
        std::cout << "Loading mel_filters from config file." << std::endl;
        try {
            for (const auto& filter : config_json.at("mel_filters")) {
                std::copy(filter.begin(), filter.end(), std::back_inserter(mel_filters.data));
            }
        } catch(const nlohmann::json::exception& e) {
            std::cerr << "Failed to load mel_filter from " << config_file << std::endl;
            mel_filters.data =std::vector<float>();
        }
    } else {
        mel_filters.data.resize(num_frequency_bins * feature_size);
        compute_mel_filter_bank(true);
    }
    return true;
}

// mel_min  0.0
// mel_max  45.245640471924965
// mel_freqs size:  130
// filter_freqs size:  130
// fft_freqs size:  201
// mel_filters size:  25728
// inspired by transformers/audio_utils.py: mel_filter_bank, default
// mel_scale = slaney.
void extractor_config::compute_mel_filter_bank(bool norm_slaney) {
    int num_frequency_bins = mel_filters.n_fft;
    // min_frequency=0.0
    // max_frequency=8000.0,
    // Hz to Mel，并等距分布
    float mel_low_freq = hz2mel_slaney(0.0);
    float mel_high_freq = hz2mel_slaney(sampling_rate / 2.0);
    std::vector<float> mel_freqs = linspace(mel_low_freq, mel_high_freq, feature_size + 2);

    // mel to hz
    int num_filter_freqs = feature_size + 2;
    std::vector<float> filter_freqs(num_filter_freqs, 0.0);
    for (int i = 0; i < num_filter_freqs; i++) {
        filter_freqs[i] = mel2hz_slaney(mel_freqs[i]);
    }
    std::vector<float> fft_freqs = linspace(0.0, sampling_rate/2.0, num_frequency_bins);

    // create_triangular_filter_bank
    // Initialize the filter bank matrix
    std::vector<std::vector<float>> filter_bank(
            num_frequency_bins, std::vector<float>(feature_size, 0.0));
    // Calculate the differences between consecutive filter frequencies
    std::vector<float> filter_diff(filter_freqs.size() - 1);
    for (size_t i = 0; i < filter_diff.size(); ++i) {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
    }
    // Calculate the slopes
    for (int i = 0; i < num_frequency_bins; i++) {
        for (int j = 1; j < num_filter_freqs - 1; j++) {
            float slope = filter_freqs[j-1] - fft_freqs[i];
            float down_slope = -slope / filter_diff[j-1];
            float up_slope = (filter_freqs[j + 1] - fft_freqs[i]) / filter_diff[j];
            // std::cout << "<" << slope << ", " << down_slope << ", " << up_slope << ">, ";
            // Apply the maximum and minimum operations
            filter_bank[i][j - 1] = std::fmax(0.0, std::fmin(down_slope, up_slope));
            //std::cout << std::fixed << std::setprecision(6) << filter_bank[i][j + 1] << ",";
        }
        //std::cout<< std::endl;
    }


    // apply slaney normalization if needed
    if (norm_slaney) {
        for (int i = 0; i < num_frequency_bins; i++) {
            for (int j = 0; j < feature_size; j++) {
                float factor = 2.0 / (filter_freqs[j + 2] - filter_freqs[j]);
                filter_bank[i][j] *= factor;
                //std::cout << std::fixed << std::setprecision(6) << filter_bank[i][j] << ", ";
            }
            //std::cout << std::endl;
        }
    }

    // Transpose to shape (80, 201)
    for (int i = 0; i < feature_size; i++) {
        for (int j = 0; j < num_frequency_bins; j++) {
            mel_filters.data[i * num_frequency_bins + j] = filter_bank[j][i];
        }
    }
    // for (const auto d: mel_filters.data) {
    //     std::cout << d << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << "mel_filters size: " << mel_filters.data.size() << std::endl;
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued

#define SIN_COS_N_COUNT 400
namespace {
struct whisper_global_cache {
    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    // torch.hann_window(self.n_fft)
    float hann_window[SIN_COS_N_COUNT];

    whisper_global_cache() {
        fill_sin_cos_table();
        fill_hann_window(sizeof(hann_window)/sizeof(hann_window[0]), true, hann_window);
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic, float * output) {
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }
} global_cache;
}

static void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n]*global_cache.cos_vals[idx]; // cos(t)
            im -= in[n]*global_cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N*2 == 1) {
        dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i]= in[2*i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2*i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = global_cache.cos_vals[idx]; // cos(t)
        float im = -global_cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        // First half of output: X[k] = E[k] + W_N^k * O[k]
        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        // Second half of output: X[k+N/2] = E[k] - W_N^k * O[k]
        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

static void rfft_magnitude(float* rfft_out, int N, float* magnitude) {
    // DC component
    magnitude[0] = std::abs(rfft_out[0]);

    // Middle frequencies
    for (int k = 1; k < N/2; k++) {
        float re = rfft_out[2*k + 0];
        float im = rfft_out[2*k + 1];
        magnitude[k] = std::sqrt(re*re + im*im);
    }

    // Nyquist frequency
    magnitude[N/2] = std::abs(rfft_out[N/2]);
}

// n_samples,       // 480000
// sampling_rate,   // 16000
// frame_size,      // 400
// frame_step,      // 160
// n_mel,           // 80
// refer to huggingface/transformers/blob/main/src/transformers/audio_utils.py#L383
static void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters & filters,
                                              bool rfft,
                                              whisper_mel & mel) {

    int n_fft = filters.n_fft;
    int i = ith;
    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    assert(n_fft == 1 + (frame_size / 2));
    int n_frames = std::min(n_samples / frame_step + 1, mel.n_len);
    // std::cout << "n_samples " << n_samples << ", n_frames " << n_frames << std::endl;

    //std::cout << "padded inputs: ";
    //for (const auto& s : samples) {
    //    std::cout << std::fixed << std::setprecision(6) << s << ", ";
    //}
    //std::cout << std::endl;
    // calculate FFT only when fft_in are not all zero
    for (; i < n_frames; i += n_threads) {
        std::vector<float> fft_in(frame_size * 2, 0.0);
        std::vector<float> fft_out(frame_size * 2 * 2 * 2);
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
            if (i == 0) {
                // std::cout << std::fixed << std::setprecision(6) << "(" << hann[j] << "," << samples[offset + j] << ") ";
                // std::cout << std::fixed << std::setprecision(6) << fft_in[j] << ", ";
            }
        }

        // if (i == 0) {
        //     std::cout << "fft_in" << i << ": ";
        //     for (const auto& fi : fft_in) {
        //         std::cout << std::fixed << std::setprecision(6) << fi << ", ";
        //     }
        //     std::cout << std::endl;
        // }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            // std::cout << "fill the rest with zeros " << fft_in.size() - n_samples + offset << std::endl;
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        fft(fft_in.data(), frame_size, fft_out.data());
        // only use the first-half part
        if (rfft) {
            std::fill(fft_out.begin() + n_fft*2, fft_out.end(), 0.0);
        }

        // if (i == 101 || i == 102) {
        //     std::cout << "fft_out" << i << ": ";
        //     for (size_t k = 0; k < fft_out.size(); k += 2) {
        //         std::cout << std::fixed << std::setprecision(6) << "(" << fft_out[k] << " ";
        //         std::cout << std::fixed << std::setprecision(6) << fft_out[k+1] << "), ";
        //     }
        //     std::cout << std::endl;
        // }

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }
        if (rfft) {
            std::fill(fft_out.begin() + n_fft, fft_out.end(), 0.0);
        }

        // if (i == 101 || i == 102) {
        //     std::cout << "after power fft_out " << i << ": ";
        //     for (const auto& fo : fft_out) {
        //         std::cout << std::fixed << std::setprecision(6) << fo << ", ";
        //     }
        //     std::cout << std::endl;
        // }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

// refer to huggingface/transformers/blob/main/src/transformers/audio_utils.py#L383
static bool log_mel_spectrogram(
              const float * samples,
              const int   n_samples,       // wav size
              const int   sampling_rate,   // 16000
              const int   frame_size,      // 400
              const int   frame_step,      // 160
              const int   n_mel,           // 128
              const int   n_threads,
              const whisper_filters & filters,
              const bool   debug,
              whisper_mel & mel) {
    // if (ctx == nullptr) {
    //     std::cerr << "run time error." << std::endl;
    //     return false;
    // }
    const int64_t t_start_us = ggml_time_us();

    // Hann window
    const float * hann = global_cache.hann_window;
    // for (int i = 0; i < SIN_COS_N_COUNT; i++) {
    //     std::cout << hann[i] << ", ";
    // }
    // std::cout << std::endl;

    // Calculate the length of padding
    int64_t stage_1_pad = sampling_rate * 30;  // 480000
    int64_t stage_2_pad = frame_size / 2;

    // copy and pad waveform to max_length and put in middle.
    std::vector<float> samples_padded(stage_1_pad + stage_2_pad*2, 0.0);
    std::copy(samples, samples+n_samples, samples_padded.begin()+stage_2_pad);

    // pad reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin()+n_samples+stage_2_pad,
              samples_padded.begin()+stage_1_pad+2*stage_2_pad, 0.0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples+1, samples+1+stage_2_pad, samples_padded.begin());
    // std::cout << "samples_padded size " << samples_padded.size() << std::endl;
    // for (const auto& s : samples_padded) {
    //     std::cout << s << ", ";
    // }
    // std::cout << std::endl;

    mel.n_mel     = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len     = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    // std::cout << "mel.n_len " << mel.n_len << ", mel.n_len_org " << mel.n_len_org << std::endl;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                    log_mel_spectrogram_worker_thread, iw + 1, hann, samples_padded,
                    stage_1_pad + stage_2_pad*2, frame_size, frame_step, n_threads,
                    std::cref(filters), true, std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(
                0, hann, samples_padded, stage_1_pad+ stage_2_pad*2,
                frame_size, frame_step, n_threads, filters, true, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }

    auto t_mel_us = ggml_time_us() - t_start_us;
    // std::cout << "used time: " << t_mel_us/1000.0 << " ms" << std::endl;

    // Dump log_mel_spectrogram
    if (debug) {
        std::ofstream outFile("log_mel_spectrogram.txt");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
            if ((i+1) % mel.n_len == 0) outFile << "\n";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }
    return true;
}

static int whisper_pcm_to_mel(extractor_config config, const float * samples,
                       int n_samples, int n_threads, bool debug,
                       whisper_mel& mel) {
    // auto config = ctx->config;
    if (!log_mel_spectrogram(samples, n_samples, config.sampling_rate,
                config.frame_size, config.hop_length, config.mel_filters.n_mel,
                n_threads, config.mel_filters, debug, mel)) {
        std::cerr << __func__ << " failed to compute mel spectrogram\n";
        return -1;
    }

    return 0;
}

bool feature_extract_v(extractor_config config,
                     const std::vector<float>& pcmf32,
                     audio_f32* features,
                     int n_output) {
    if(false){
        FILE *fp = fopen("pcmf32.txt", "w");
        for(int i = 0; i < pcmf32.size(); i++){
            fprintf(fp, "%f ", pcmf32[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
    }
    // auto config = ctx->config;
    whisper_mel * mel_out = new whisper_mel;
    // printf("feature_extract 1-1 :\n");
    auto ret = whisper_pcm_to_mel(config, pcmf32.data(), pcmf32.size(),
            1 /* n_processors */, false /* debug */,  *mel_out);
    // printf("feature_extract 1-2 :\n");
    if (ret != 0) {
        std::cerr << __func__ << ": encountered errors." << std::endl;
        return false;
    }
    // printf("feature_extract 2 :\n");
    // std::cout << "mel_out: n_mel " << mel_out->n_mel << ", n_len " << mel_out->n_len << " " << mel_out->n_len_org << std::endl;
    features->n_mel = mel_out->n_mel;
    if(false){
        FILE *fp = fopen("input_features.txt", "w");
        for(int i = 0; i < mel_out->data.size(); i++){
            fprintf(fp, "%f ", mel_out->data[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
    }
    // output features is padded to n_mel * sr/hop_length*30
    if (n_output > 0) {
        int length = n_output / mel_out->n_mel;
        features->n_len = length;
        // std::cout << "truncate output features to size " << n_output << " " << features->buf.size() << std::endl;
        // printf("feature_extract 3 :\n");
        features->buf.resize(n_output);
        for (int i = 0; i < mel_out->n_mel; i++) {
            std::copy(mel_out->data.begin() + i * mel_out->n_len, mel_out->data.begin() + i * mel_out->n_len + length, features->buf.begin() + i * length);
        }
    } else {
        features->n_len = mel_out->n_len;
        features->buf.resize(mel_out->data.size());
        std::copy(mel_out->data.begin(), mel_out->data.end(), features->buf.begin());
    }
    // printf("feature_extract 4 :\n");
    // std::cout << "features size " << features->buf.size() << ", n_len = " << features->n_len << std::endl;
    if(false){
        FILE *fp = fopen("audio_features.txt", "w");
        for(int i = 0; i < features->buf.size(); i ++){
            if(i < 10){
                printf("%f ", features->buf[i]);
            }
            fprintf(fp, "%.5f ", features->buf[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
        printf("\n");
    }
    return true;
}

bool feature_extract_f(extractor_config config,
                     const std::string& fname,
                     audio_f32* features,
                     int n_output) {
    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM
    if (!read_wav(fname, pcmf32, pcmf32s, false)) {
        std::cerr << "failed to read wav from " << fname << std::endl;
        return false;
    }
    // printf("pcmf32: %d, pcmf32s: %d\n", pcmf32.size(), pcmf32s.size()); 
    // if (n_output == FIXED_SHAPE_SIZE) {
    //     std::cout << "truncate input size from " << pcmf32.size()
    //         << " to " << FIXED_INPUT_SIZE<< std::endl;
    //     // truncate input to FIXED_INPUT_SIZE.
    //     std::fill(pcmf32.begin() + FIXED_INPUT_SIZE, pcmf32.end(), 0.0);
    // }
    return feature_extract_v(config, pcmf32, features, n_output);
}

static void whisper_encoder_config_free(struct audio_encoder_config *config) {
    if (config) {
        ggml_backend_buffer_free(config->params_buffer);
        ggml_gallocr_free(config->compute_alloc);
        // gguf_free(config->ctx_gguf);
        ggml_free(config->ctx_data);
        ggml_backend_free(config->backend);
        // delete config;
    }
}

void audio_ctx_free(struct audio_ctx* ctx) {
    if (ctx) {
        whisper_encoder_config_free(&ctx->encoder_config);
        delete ctx;
    }
}

static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        throw std::runtime_error("Failed to find tensor " + name);
    }

    return cur;
}


// 加载 GGUF 模型的函数
static bool load_gguf_model(const std::string &gguf_path, audio_ctx &ctx) {
    auto & config = ctx.encoder_config;
    const char *fname = gguf_path.c_str();
    struct ggml_context * meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(fname, params);
    if (!gguf_ctx) {
        throw std::runtime_error("Failed to load GGUF model from " + gguf_path);
    }

    const int n_tensors = gguf_get_n_tensors(gguf_ctx);

    // kv
    const int n_kv = gguf_get_n_kv(gguf_ctx);
    LOG_INF("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n",
        __func__, n_kv, n_tensors, fname);

    int n_audio_layer = 24;

#ifdef GGML_USE_CUDA
    config.backend = ggml_backend_cuda_init(0);
    LOG_INF("%s: Audio encoder using CUDA backend\n", __func__);
#endif

#ifdef GGML_USE_METAL
    //config.backend = ggml_backend_metal_init();
    //LOG_INF("%s: Audio encoder using Metal backend\n", __func__);
#endif

#ifdef GGML_USE_CANN
    config.backend = ggml_backend_cann_init(0);
    LOG_INF("%s: Audio encoder using CANN backend\n", __func__);
#endif

#ifdef GGML_USE_VULKAN
    config.backend = ggml_backend_vk_init(0);
    LOG_INF("%s: Audio encoder using Vulkan backend\n", __func__);
#endif

    if (!config.backend) {
        config.backend = ggml_backend_cpu_init();
        LOG_INF("%s: Audio encoder using CPU backend\n", __func__);
    }
    
    // load tensors
    {
        std::vector<uint8_t> read_buf;
        struct ggml_init_params params = {
            /*.mem_size =*/ (n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };

        config.ctx_data = ggml_init(params);
        if (!config.ctx_data) {
            LOG_ERR("%s: ggml_init() failed\n", __func__);
            audio_ctx_free(&ctx);
            gguf_free(gguf_ctx);
            return false;
        }

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            LOG_ERR("cannot open model file for loading tensors\n");
            audio_ctx_free(&ctx);
            gguf_free(gguf_ctx);
            return false;
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            struct ggml_tensor * t = ggml_get_tensor(meta, name);
            struct ggml_tensor * cur = ggml_dup_tensor(config.ctx_data, t);
            ggml_set_name(cur, name);
        }

        // alloc memory and offload data
        config.params_buffer = ggml_backend_alloc_ctx_tensors(config.ctx_data, config.backend);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            struct ggml_tensor * cur = ggml_get_tensor(config.ctx_data, name);
            const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                LOG_ERR("%s: failed to seek for tensor %s\n", __func__, name);
                audio_ctx_free(&ctx);
                gguf_free(gguf_ctx);
                return false;   
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(config.params_buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        fin.close();
    }

    auto & model = ctx.model;
    auto & hparams = model.hparams;

    hparams.n_audio_layer = n_audio_layer;

    model.conv1_w = get_tensor(config.ctx_data, "apm.conv1.weight");
    model.conv1_b = get_tensor(config.ctx_data, "apm.conv1.bias");
    model.conv2_w = get_tensor(config.ctx_data, "apm.conv2.weight");
    model.conv2_b = get_tensor(config.ctx_data, "apm.conv2.bias");
    model.embed_positions = get_tensor(config.ctx_data, "apm.embed_positions.weight");
    model.layer_norm_w = get_tensor(config.ctx_data, "apm.layer_norm.weight");
    model.layer_norm_b = get_tensor(config.ctx_data, "apm.layer_norm.bias");
    // 对于 layers，可以循环映射张量
    for (int i = 0; i < hparams.n_audio_layer; ++i) {
        std::string layer_name = "apm.layers." + std::to_string(i) + ".";
        audio_layer layer;
        layer.self_attn_k_proj_w = get_tensor(config.ctx_data, layer_name + "self_attn.k_proj.weight");
        layer.self_attn_q_proj_w = get_tensor(config.ctx_data, layer_name + "self_attn.q_proj.weight");
        layer.self_attn_v_proj_w = get_tensor(config.ctx_data, layer_name + "self_attn.v_proj.weight");
        layer.self_attn_out_proj_w = get_tensor(config.ctx_data, layer_name + "self_attn.out_proj.weight");
        layer.self_attn_q_proj_b = get_tensor(config.ctx_data, layer_name + "self_attn.q_proj.bias");
        layer.self_attn_v_proj_b = get_tensor(config.ctx_data, layer_name + "self_attn.v_proj.bias");
        layer.self_attn_out_proj_b = get_tensor(config.ctx_data, layer_name + "self_attn.out_proj.bias");
        layer.self_attn_layer_norm_w = get_tensor(config.ctx_data, layer_name + "self_attn_layer_norm.weight");
        layer.self_attn_layer_norm_b = get_tensor(config.ctx_data, layer_name + "self_attn_layer_norm.bias");
        layer.fc1_w = get_tensor(config.ctx_data, layer_name + "fc1.weight");
        layer.fc1_b = get_tensor(config.ctx_data, layer_name + "fc1.bias");
        layer.fc2_w = get_tensor(config.ctx_data, layer_name + "fc2.weight");
        layer.fc2_b = get_tensor(config.ctx_data, layer_name + "fc2.bias");
        layer.final_layer_norm_w = get_tensor(config.ctx_data, layer_name + "final_layer_norm.weight");
        layer.final_layer_norm_b = get_tensor(config.ctx_data, layer_name + "final_layer_norm.bias");
        model.layers.push_back(layer);
    }

    model.audio_proj_layer_linear1_w = get_tensor(config.ctx_data, "audio_projection_layer.linear1.weight");
    model.audio_proj_layer_linear1_b = get_tensor(config.ctx_data, "audio_projection_layer.linear1.bias");
    model.audio_proj_layer_linear2_w = get_tensor(config.ctx_data, "audio_projection_layer.linear2.weight");
    model.audio_proj_layer_linear2_b = get_tensor(config.ctx_data, "audio_projection_layer.linear2.bias");

    ggml_free(meta);
    gguf_free(gguf_ctx);

    return true;
}

size_t audio_embd_nbytes() {
    return 25 * 2560 * sizeof(float);
}

struct audio_ctx* audio_ctx_init(const std::string& model_path) {
    // std::cout << __func__ << model_path << std::endl;
    audio_ctx* ctx = new audio_ctx;
    // if (!ctx->config.init_from_file(model_path)) {
    //     std::cout << __func__ << ": failed to load preprocessor config file." << std::endl
    //               << __func__ << ": init with default values." << std::endl;
    //     ctx->config.init();  // init with default value.
    // }
    // if (model_path.empty()) {
    //     return ctx;
    // }

    // std::string gguf_path = model_path + "/whisper.gguf";
    //std::string gguf_path = "/Users/zkh/Downloads/whisper-new.gguf";
    LOG_INF("Loading GGUF model from: %s\n", model_path.c_str());
    if (!load_gguf_model(model_path, *ctx)) {
        std::cerr << __func__ << ": failed to load GGUF model." << std::endl;
        delete ctx;
        return nullptr;
    }

    return ctx;
}


bool audio_wav_preprocess(struct audio_ctx* ctx, const audio_u8 * aud, audio_f32* res_auds, int n_output) {
    if (res_auds == nullptr) {
        res_auds = new audio_f32();
    }
    // printf("audio_wav_preprocess 1 :\n");
    if (!feature_extract_f(ctx->config, std::string(aud->buf.begin(), aud->buf.end()), res_auds, n_output)) {
        std::cerr << __func__ << ": failed extract features from audio buffer." << std::endl;
        return false;
    }
    // printf("audio_wav_preprocess 2 :\n");
    return true;
}


static ggml_cgraph * whisper_build_graph(audio_ctx * ctx) {

    const auto & model = ctx->model;
    const auto & hparams = model.hparams;

    const int n_state_head = hparams.n_audio_state/hparams.n_audio_head;
    const float KQscale = 1.0f/sqrtf(float(n_state_head));

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_audio_ctx, hparams.n_mels);
    // printf("mel shape: %d %d\n", hparams.n_audio_ctx, hparams.n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    struct ggml_tensor * cur = nullptr;

    // {
    //     FILE* fp1 = fopen("conv1_weight.bin", "rb");
    //     FILE* fp2 = fopen("conv1_bias.bin", "rb");
    //     fread(model.conv1_w->data, 1, 1024 * 80 * 3 * sizeof(int16_t), fp1);
    //     fread(model.conv1_b->data, 1, 1024 * sizeof(int16_t), fp2);
    //     fclose(fp1);
    //     fclose(fp2);
    // }

    cur = ggml_conv_1d_ph(ctx0, model.conv1_w, mel, 1, 1);
    cur = ggml_add(ctx0, cur, model.conv1_b);
    cur = ggml_gelu(ctx0, cur);

    cur = ggml_conv_1d_ph(ctx0, model.conv2_w, cur, 2, 1);
    cur = ggml_add(ctx0, cur, model.conv2_b);
    cur = ggml_gelu(ctx0, cur);

    // printf("embed_pos: %d %d %d %d\n", model.embed_positions->ne[0], model.embed_positions->ne[1], model.embed_positions->ne[2], model.embed_positions->ne[3]);
    // printf("cur: %d %d %d %d\n", cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    // printf("n_audio_ctx %d\n", hparams.n_audio_ctx);
    const size_t e_pe_stride = model.embed_positions->ne[0]*ggml_element_size(model.embed_positions);
    const size_t e_pe_offset = 0;
    struct ggml_tensor * e_pe = ggml_view_2d(ctx0, model.embed_positions, model.embed_positions->ne[0], cur->ne[0], e_pe_stride, e_pe_offset);
    // printf("e_pe.shape = %d %d %d %d\n", e_pe->ne[0], e_pe->ne[1], e_pe->ne[2], e_pe->ne[3]);
    cur = ggml_add(ctx0, e_pe, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));
    // if(false){
    //     ggml_set_name(cur, "layer_input");
    //     ggml_build_forward_expand(gf, cur);
    //     ggml_free(ctx0);
    //     return gf;
    // }


    struct ggml_tensor * inpL = cur;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        // if (il == 1) {
        //     break;
        // }
        const auto & layer = model.layers[il];
        // norm
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);
            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0, cur, layer.self_attn_layer_norm_w),
                    layer.self_attn_layer_norm_b);
        }
 

    
        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.self_attn_q_proj_w, cur), layer.self_attn_q_proj_b);
            Qcur = ggml_scale_inplace(ctx0, Qcur, KQscale);
            // printf("qcur: %d %d %d %d\n", Qcur->ne[0], Qcur->ne[1], Qcur->ne[2], Qcur->ne[3]);
            // printf("n_state_head = %d, n_audio_head = %d\n", n_state_head, hparams.n_audio_head);
            struct ggml_tensor * Q = 
                ggml_permute(ctx0, 
                    ggml_reshape_3d(ctx0, Qcur, n_state_head, hparams.n_audio_head, Qcur->ne[1]), 
                    0, 2, 1, 3);
            // printf("q: %d %d %d %d\n", Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3]);
            // note: no bias for Key
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.self_attn_k_proj_w, cur);
            struct ggml_tensor * K = ggml_permute(ctx0, ggml_cast(ctx0, ggml_reshape_3d(ctx0, Kcur, n_state_head, hparams.n_audio_head, Kcur->ne[1]), GGML_TYPE_F16), 0, 2, 1, 3);
                ggml_permute(ctx0,
                        ggml_cast(ctx0,
                            ggml_reshape_3d(ctx0, Kcur, n_state_head, hparams.n_audio_head, Kcur->ne[1]),
                            GGML_TYPE_F16),
                        0, 2, 1, 3);

            struct ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.self_attn_v_proj_w, cur), layer.self_attn_v_proj_b);
            struct ggml_tensor * V =
                ggml_cast(ctx0,
                        ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                Vcur,
                                n_state_head, hparams.n_audio_head, Vcur->ne[1]),
                            1, 2, 0, 3),
                        GGML_TYPE_F16);
            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_inplace(ctx0, KQ);
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            // printf("KQV_merge: %d %d %d %d\n", KQV_merged->ne[0], KQV_merged->ne[1], KQV_merged->ne[2], KQV_merged->ne[3]);
            // printf("n_state_head = %d, n_audio_head = %d\n", n_state_head, hparams.n_audio_state);
            cur = ggml_cont_2d(ctx0, KQV_merged, hparams.n_audio_state, KQV_merged->ne[2]);
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0,
                    layer.self_attn_out_proj_w,
                    cur);

            cur = ggml_add(ctx0, cur, layer.self_attn_out_proj_b);
        }
 
        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_add(ctx0,
                        ggml_mul(ctx0, cur, layer.final_layer_norm_w),
                        layer.final_layer_norm_b);
            }

            // fully connected
            cur = ggml_mul_mat(ctx0,
                    layer.fc1_w,
                    cur);

            cur = ggml_add(ctx0, cur, layer.fc1_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0,
                    layer.fc2_w,
                    cur);

            cur = ggml_add(ctx0, cur, layer.fc2_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);

    }


    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);

        // cur = ln_f_g*cur + ln_f_b
        cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.layer_norm_w),
                model.layer_norm_b);
    }

    // ggml_set_name(cur, "hidden_states");

    // Add projection layers
    {
        // First linear layer
        cur = ggml_mul_mat(ctx0, 
                model.audio_proj_layer_linear1_w,
                ggml_cont(ctx0, cur));
        cur = ggml_add(ctx0, cur, model.audio_proj_layer_linear1_b);

        // ReLU activation
        cur = ggml_relu(ctx0, cur);

        // Second linear layer
        cur = ggml_mul_mat(ctx0,
                model.audio_proj_layer_linear2_w,
                ggml_cont(ctx0, cur));
        cur = ggml_add(ctx0, cur, model.audio_proj_layer_linear2_b);

        // now tensor shape is [2560, 51]

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        // now tensor shape is [51, 2560]

        // Apply average pooling using ggml_pool_1d
        cur = ggml_cont(ctx0, ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0));
        // now tensor shape is [25, 2560]

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        // now tensor shape is [2560, 25]
    }
      
    /*
    */

    ggml_set_name(cur, "audio_embeds");

    // build the graph
    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    return gf;
}

static bool audio_batch_encode(struct audio_ctx * ctx, const int n_threads, audio_f32_batch * audio_batch, audio_f32& ret) {
    if (audio_batch->size != 1) {
        std::cerr << __func__ << ": only support batch size 1." << std::endl;
        return false;
    }
    audio_f32 * audio = audio_batch->data;

    auto & encoder_config = ctx->encoder_config;
    ctx->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
    encoder_config.compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(encoder_config.backend));

    auto & hparams = ctx->model.hparams;
    hparams.n_audio_ctx = audio->n_len; //audio->n_len / 2;
    hparams.n_mels = audio->n_mel;
    hparams.n_audio_head = 16;
    hparams.n_audio_state = 1024;
    hparams.eps = 1e-05f;

    ggml_cgraph * gf = whisper_build_graph(ctx);
    ggml_gallocr_reserve(encoder_config.compute_alloc, gf);
    ggml_gallocr_alloc_graph(encoder_config.compute_alloc, gf);

   
    // int mel_size = 2*ctx.n_ctx * ctx.n_mels;
    struct ggml_tensor * inp_mel = ggml_graph_get_tensor(gf, "mel");
    // std::ifstream file("/Users/hankyz/mel_result.bin", std::ios::binary);
    // float *mel_data = (float *) malloc(mel_size * sizeof(float));
    // file.read(reinterpret_cast<char*>(mel_data), mel_size * sizeof(float));

    // ggml_backend_tensor_set(inp_mel, mel_data, 0, ggml_nbytes(inp_mel));
    ggml_backend_tensor_set(inp_mel, reinterpret_cast<float*>(audio->buf.data()), 0, ggml_nbytes(inp_mel));
    //print_tensor(inp_mel);
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }
    ggml_backend_graph_compute(encoder_config.backend, gf);
    // printf("%s: graph compute done\n", __func__);


    // struct ggml_tensor * hidden_states = ggml_graph_get_tensor(gf, "hidden_states");
    // print_tensor_2d(hidden_states);
    // std::ofstream outfile("cpp_hidden_states.bin", std::ios::binary);
    // outfile.write(reinterpret_cast<const char*>(hidden_states->data), ggml_nbytes(hidden_states));
    // outfile.close();
    // ggml_backend_tensor_get(hidden_states, vec, 0, ggml_nbytes(hidden_states));

    struct ggml_tensor * audio_embeds = ggml_graph_get_tensor(gf, "audio_embeds");
    ret.n_len = audio_embeds->ne[1];
    // printf("audio_embeds:");
    // print_tensor(audio_embeds);
    ret.buf.resize(ggml_nelements(audio_embeds));
    ggml_backend_tensor_get(audio_embeds, ret.buf.data(), 0, ggml_nbytes(audio_embeds));
    return true;
}

bool audio_encode(struct audio_ctx * ctx, const int n_threads, audio_f32 * aud, audio_f32& ret) {
    if (ctx->npu_use_ane) {
        float * aud_embedding1 = aud->buf.data();
        // audio_encode_ane(aud_embedding1, ret.buf.data());
        free(aud_embedding1);
        return true;
    }
    audio_f32_batch auds{};
    auds.size = 1;
    auds.data = aud;
    bool status = audio_batch_encode(ctx, n_threads, &auds, ret);
    return status;
}
