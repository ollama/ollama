#define _USE_MATH_DEFINES // for M_PI
#include "mtmd-audio.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <fstream>
#include <algorithm>

// most of the code here is copied from whisper.cpp

constexpr bool DEBUG = false;

struct mtmd_audio_mel_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

// note: this global cache is shared among all preprocessors
//       if we want to use multiple preprocessors at the same time,
//       we will need to enclose it in the preprocessor class in the future
static struct mtmd_audio_global_cache {
    // precomputed sin/cos table for FFT
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    // hann window
    std::vector<float> hann_window;

    // mel filter bank
    mtmd_audio_mel_filters filters;

    void fill_sin_cos_table(int n) {
        sin_vals.resize(n);
        cos_vals.resize(n);
        for (int i = 0; i < n; i++) {
            double theta = (2 * M_PI * i) / n;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic) {
        hann_window.resize(length);
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            hann_window[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }

    // Build mel filterbank matrix [n_mel × n_fft_bins] at runtime.
    // n_fft_bins must be (N_fft / 2 + 1). Example: if N_fft=512 -> n_fft_bins=257.
    void fill_mel_filterbank_matrix(
        int n_mel,
        int n_fft,
        int sample_rate,            // e.g. 16000
        float fmin = 0.0f,          // e.g. 0.0
        float fmax = -1.0f,         // e.g. sr/2; pass -1 for auto
        bool slaney_area_norm = true,
        float scale = 1.0f          // optional extra scaling; use 1.0f/1000.0f to mimic your code
    ) {
        GGML_ASSERT(n_mel > 0 && n_fft > 1);
        if (fmax <= 0.0f) {
            fmax = 0.5f * sample_rate;
        }

        // Slaney scale (matches librosa default)
        const double min_log_hz = 1000.0;
        const double lin_slope = 3 / 200.;
        const double min_log_mel = min_log_hz * lin_slope;
        const double log_step = log(6.4) / 27.0;
        auto hz_to_mel = [min_log_hz, lin_slope, log_step, min_log_mel](const double f_hz) -> double {
            return (f_hz < min_log_hz) ? f_hz * lin_slope : min_log_mel + log(f_hz / min_log_hz) / log_step;
        };
        auto mel_to_hz = [min_log_hz, lin_slope, log_step, min_log_mel](const double m) -> double {
            return (m < min_log_mel) ? m / lin_slope : min_log_hz * exp((m - min_log_mel) * log_step);
        };

        // infer N_fft from n_fft_bins
        const double bin_hz_step = double(sample_rate) / double(n_fft);

        // mel grid: n_mel + 2 edges
        const double m_lo = hz_to_mel(fmin);
        const double m_hi = hz_to_mel(fmax);
        std::vector<double> mel_pts(n_mel + 2);
        for (int i = 0; i < n_mel + 2; ++i) {
            mel_pts[i] = m_lo + (m_hi - m_lo) * (double(i) / (n_mel + 1));
        }

        // convert to Hz
        std::vector<double> hz_pts(n_mel + 2);
        for (int i = 0; i < n_mel + 2; ++i) {
            hz_pts[i] = mel_to_hz(mel_pts[i]);
        }

        const int n_fft_bins = n_fft / 2 + 1;

        // filterbank
        std::vector<float> out(n_mel * n_fft_bins, 0);
        for (int m = 0; m < n_mel; ++m) {
            const double f_left   = hz_pts[m];
            const double f_center = hz_pts[m + 1];
            const double f_right  = hz_pts[m + 2];

            const double denom_l = std::max(1e-30, f_center - f_left);
            const double denom_r = std::max(1e-30, f_right  - f_center);
            const double enorm   = slaney_area_norm ? (2.0 / std::max(1e-30, f_right - f_left)) : 1.0;

            for (int k = 0; k < n_fft_bins; ++k) {
                const double f = k * bin_hz_step;
                double w = 0.0;
                if (f >= f_left && f <= f_center) {
                    w = (f - f_left) / denom_l;
                } else if (f > f_center && f <= f_right) {
                    w = (f_right - f) / denom_r;
                }
                out[size_t(m) * size_t(n_fft_bins) + size_t(k)] = float(w * enorm * scale);
            }
        }

        filters.n_mel = n_mel;
        filters.n_fft = n_fft;
        filters.data  = std::move(out);

        if (DEBUG) { // debug
            for (size_t i = 0; i < filters.data.size(); ++i) {
                if (filters.data[i] != 0.0f) {
                    printf("filters[%zu] = %f\n", i, filters.data[i] * 1000.0f);
                }
            }
        }
    }
} g_cache;

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const float * in, int N, float * out) {
    const int n_sin_cos_vals = g_cache.sin_vals.size();
    const int sin_cos_step = n_sin_cos_vals / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (n_sin_cos_vals); // t = 2*M_PI*k*n/N
            re += in[n] * g_cache.cos_vals[idx]; // cos(t)
            im -= in[n] * g_cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(float * in, int N, float * out) {
    const int n_sin_cos_vals = g_cache.sin_vals.size();
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

    const int sin_cos_step = n_sin_cos_vals / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re =  g_cache.cos_vals[idx]; // cos(t)
        float im = -g_cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

struct filter_params {
    int32_t n_mel;
    int32_t n_fft_bins;
    int32_t hann_window_size;
    int32_t hop_length;
    int32_t sample_rate;
    bool    center_padding = false;
    float   preemph = 0.f;
    bool    use_natural_log = false;
    bool    norm_per_feature = false;
};

static void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const filter_params & params, mtmd_audio_mel & out) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft_bins = params.n_fft_bins;
    int i = ith;

    const auto & filters = g_cache.filters;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    GGML_ASSERT(n_fft_bins == 1 + (frame_size / 2));
    GGML_ASSERT(g_cache.sin_vals.size() == g_cache.cos_vals.size());
    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, out.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft_bins; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < out.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft_bins - 3; k += 4) {
                size_t idx = size_t(j) * size_t(n_fft_bins) + size_t(k);
                sum +=
                        fft_out[k + 0] * filters.data[idx + 0] +
                        fft_out[k + 1] * filters.data[idx + 1] +
                        fft_out[k + 2] * filters.data[idx + 2] +
                        fft_out[k + 3] * filters.data[idx + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft_bins; k++) {
                sum += fft_out[k] * filters.data[j * n_fft_bins + k];
            }
            sum = params.use_natural_log
                ? log(sum + 5.960464477539063e-08)
                : log10(std::max(sum, 1e-10));
            out.data[j * out.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = params.use_natural_log ? log(1e-10) : log10(1e-10);
    for (; i < out.n_len; i += n_threads) {
        for (int j = 0; j < out.n_mel; j++) {
            out.data[j * out.n_len + i] = sum;
        }
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
        const float * samples,
        const int     n_samples_in,
        const int     n_threads,
        const filter_params & params,
        mtmd_audio_mel & out) {
    //const int64_t t_start_us = ggml_time_us();

    out.n_len_org = n_samples_in;
    int n_samples = n_samples_in;

    // Hann window
    const float * hann = g_cache.hann_window.data();
    const int frame_size = (params.n_fft_bins - 1) * 2;
    const int frame_step = params.hop_length;

    // Padding
    std::vector<float> samples_padded;
    if (params.center_padding) {
        const auto pad_amount = frame_size / 2;
        samples_padded = std::vector<float>(n_samples + 2 * pad_amount, 0);
        std::copy(samples, samples + n_samples, samples_padded.data() + pad_amount);
        samples = samples_padded.data();
        n_samples = samples_padded.size();
    } else {
        // existing padding logic
        int64_t stage_1_pad = params.sample_rate * 30;
        int64_t stage_2_pad = frame_size / 2;
        samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
        std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);
        // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
        std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);
        // reflective pad 200 samples at the beginning of audio
        if (n_samples < stage_2_pad + 1) {
            // TODO: Handle short audio differently or return error
            return false;
        }
        std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());
    }

    // preemphasis
    if (params.preemph) {
        const int pad_amount = frame_size / 2;
        const float preemph = 0.97f;
        float prev = samples_padded[pad_amount];
        for (int i = pad_amount + 1; i + pad_amount < n_samples; ++i) {
            float cur = samples_padded[i];
            samples_padded[i] = cur - preemph * prev;
            prev = cur;
        }
    }

    // pad hann window if it's smaller than frame_size
    // TODO: probably unnecessary here? (or better doing it in g_cache?)
    std::vector<float> hann_window_padded;
    if (params.hann_window_size < frame_size) {
        hann_window_padded.resize(frame_size);
        const int padding = (frame_size - params.hann_window_size) / 2;
        std::copy(hann, hann + params.hann_window_size, &hann_window_padded[padding]);
        hann = hann_window_padded.data();
    }


    out.n_mel = params.n_mel;
    out.n_len = (n_samples - frame_size) / frame_step + 1;
    // TODO: handle these checks better
    if (out.n_mel > 0 && (unsigned long)out.n_len > SIZE_MAX / out.n_mel) {
        LOG_ERR("%s: size overflow\n", __func__);
        return false;
    }
    if (n_samples < frame_size) {
        LOG_ERR("%s: not enough samples after padding\n", __func__);
        return false;
    }
    out.data.resize(out.n_mel * out.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                    log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded),
                    n_samples, frame_size, frame_step, n_threads,
                    std::cref(params), std::ref(out));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples, frame_size, frame_step, n_threads, params, out);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    const int effective_n_len = n_samples_in / frame_step;
    if (params.norm_per_feature) {
        for (int i = 0; i < out.n_mel; i++) {
            double mean = 0;
            for (int j = 0; j < effective_n_len; ++j) {
                mean += out.data[i * out.n_len + j];
            }
            mean /= effective_n_len;

            double var = 0.0;
            for (int j = 0; j < effective_n_len; ++j) {
                const double value = out.data[i * out.n_len + j] - mean;
                var += value * value;
            }
            var /= effective_n_len - 1;  // unbiased
            const double mstd = std::sqrt(var + 1e-5);

            for (int j = 0; j < effective_n_len; ++j) {
                auto &value = out.data[i * out.n_len + j];
                value = (value - mean) / mstd;
            }

            // pad the rest with zeros
            for (int j = effective_n_len; j < out.n_len; ++j) {
                out.data[i * out.n_len + j] = 0.0;
            }
        }
    } else {
        // clamping and normalization
        double mmax = -1e20;
        for (int i = 0; i < out.n_mel*out.n_len; i++) {
            if (out.data[i] > mmax) {
                mmax = out.data[i];
            }
        }

        mmax -= 8.0;

        for (int i = 0; i < out.n_mel*out.n_len; i++) {
            if (out.data[i] < mmax) {
                out.data[i] = mmax;
            }
            out.data[i] = (out.data[i] + 4.0)/4.0;
        }
    }

    // Dump log_mel_spectrogram
    if (DEBUG) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < out.data.size() - 1; i++) {
            outFile << out.data[i] << ", ";
        }
        outFile << out.data[out.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

//
// mtmd_audio_preprocessor_whisper
//

void mtmd_audio_preprocessor_whisper::initialize() {
    g_cache.fill_sin_cos_table(hparams.audio_n_fft);
    g_cache.fill_hann_window(hparams.audio_window_len, true);
    g_cache.fill_mel_filterbank_matrix(
        hparams.n_mel_bins,
        hparams.audio_n_fft,
        hparams.audio_sample_rate);
}

bool mtmd_audio_preprocessor_whisper::preprocess(
        const float * samples,
        size_t n_samples,
        std::vector<mtmd_audio_mel> & output) {
    if (n_samples == 0) {
        // empty audio
        return false;
    }

    std::vector<float> smpl;
    // if input is too short, pad with zeros
    // this is to avoid potential issues with stage1/2 padding in log_mel_spectrogram
    // TODO: maybe handle this better
    size_t min_samples = (size_t)hparams.audio_sample_rate * (hparams.audio_chunk_len + 1); // +1 second margin
    if (n_samples < min_samples) {
        smpl.resize(min_samples, 0.0f);
        std::memcpy(smpl.data(), samples, n_samples * sizeof(float));
        samples   = smpl.data();
        n_samples = smpl.size();
    }

    filter_params params;
    params.n_mel            = hparams.n_mel_bins;
    params.n_fft_bins       = 1 + (hparams.audio_n_fft / 2);
    params.hann_window_size = hparams.audio_window_len;
    params.hop_length       = hparams.audio_hop_len;
    params.sample_rate      = hparams.audio_sample_rate;
    params.center_padding   = false;
    params.preemph          = 0.0f; // disabled
    params.use_natural_log  = false;
    params.norm_per_feature = false;

    // make sure the global cache is initialized
    GGML_ASSERT(!g_cache.sin_vals.empty());
    GGML_ASSERT(!g_cache.cos_vals.empty());
    GGML_ASSERT(!g_cache.filters.data.empty());

    mtmd_audio_mel out_full;
    bool ok = log_mel_spectrogram(
                samples,
                n_samples,
                4, // n_threads
                params,
                out_full);
    if (!ok) {
        return false;
    }

    // because the cgraph in clip.cpp only accepts 3000 frames each, we need to split the mel
    // we always expect the mel to have 3000 silent frames at the end
    if (DEBUG) {
        printf("output: n_mel = %d, n_len = %d\n", out_full.n_mel, out_full.n_len);
    }
    const size_t frames_per_chunk = 3000;
    GGML_ASSERT((size_t)out_full.n_len > frames_per_chunk);
    for (size_t off = 0; off < (size_t)out_full.n_len; off += frames_per_chunk) {
        int n_len = std::min(frames_per_chunk, (size_t)out_full.n_len - off);
        if ((size_t)n_len < frames_per_chunk) {
            break; // last uncomplete chunk will always be a padded chunk, safe to ignore
        }

        mtmd_audio_mel out_chunk;
        out_chunk.n_len     = n_len;
        out_chunk.n_mel     = out_full.n_mel;
        out_chunk.n_len_org = out_full.n_mel; // unused
        out_chunk.data.reserve(out_chunk.n_mel * out_chunk.n_len);

        for (int i = 0; i < out_full.n_mel; i++) {
            auto src = out_full.data.begin() + i*out_full.n_len + off;
            out_chunk.data.insert(out_chunk.data.end(), src, src + frames_per_chunk);
        }

        output.push_back(std::move(out_chunk));
    }

    return true;
}
