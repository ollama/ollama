#include "mtmd-audio.h"

#define _USE_MATH_DEFINES // for M_PI
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <fstream>
#include <algorithm>

// most of the code here is copied from whisper.cpp

// align x to upper multiple of n
#define _ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

namespace whisper_preprocessor {

#define SIN_COS_N_COUNT WHISPER_N_FFT
namespace {
struct whisper_global_cache {
    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    float hann_window[WHISPER_N_FFT];

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

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
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

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

static void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters & filters, whisper_mel & mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft = filters.n_fft;
    int i = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    WHISPER_ASSERT(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
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
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

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

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
        const float * samples,
        const int   n_samples,
        const int   /*sample_rate*/,
        const int   frame_size,
        const int   frame_step,
        const int   n_mel,
        const int   n_threads,
        const whisper_filters & filters,
        const bool   debug,
        whisper_mel & mel) {
    //const int64_t t_start_us = ggml_time_us();

    // Hann window
    WHISPER_ASSERT(frame_size == WHISPER_N_FFT && "Unsupported frame_size");
    const float * hann = global_cache.hann_window;

    // Calculate the length of padding
    int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel     = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len     = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                    log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded),
                    n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                    std::cref(filters), std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel);

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

    // Dump log_mel_spectrogram
    if (debug) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

bool preprocess_audio(
        const float * samples,
        size_t n_samples,
        const whisper_filters & filters,
        std::vector<whisper_mel> & output) {

    if (n_samples == 0) {
        // empty audio
        return false;
    }

    whisper_mel out_full;
    bool ok = log_mel_spectrogram(
                samples,
                n_samples,
                COMMON_SAMPLE_RATE,
                WHISPER_N_FFT,
                WHISPER_HOP_LENGTH,
                filters.n_mel,
                4, // n_threads
                filters,
                false, // debug
                out_full);
    if (!ok) {
        return false;
    }

    // because the cgraph in clip.cpp only accepts 3000 frames each, we need to split the mel
    // we always expect the mel to have 3000 silent frames at the end
    // printf("n_len %d\n", out_full.n_len);
    const size_t frames_per_chunk = 3000;
    GGML_ASSERT((size_t)out_full.n_len > frames_per_chunk);
    for (size_t off = 0; off < (size_t)out_full.n_len; off += frames_per_chunk) {
        int n_len = std::min(frames_per_chunk, (size_t)out_full.n_len - off);
        if ((size_t)n_len < frames_per_chunk) {
            break; // last uncomplete chunk will always be a padded chunk, safe to ignore
        }

        whisper_mel out_chunk;
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

} // namespace whisper_preprocessor


// precalculated mel filter banks
// values are multiplied by 1000.0 to save space, and will be divided by 1000.0 in the end of the function
//
// generated from python code:
//
// from numpy import load
// data = load('mel_filters.npz')
// lst = data.files
// for item in lst:
//   print(item)
//   print(data[item].shape)
//   n_mel = data[item].shape[0]
//   n_fft = data[item].shape[1]
//   for i, row in enumerate(data[item]):
//     for j, val in enumerate(row):
//       val = val * 1000.0
//       if val != 0:
//         print(f"data[{i*n_fft + j}] = {val:.6f};")

namespace whisper_precalc_filters {

whisper_preprocessor::whisper_filters get_128_bins() {
    whisper_preprocessor::whisper_filters filters;
    filters.n_mel = 128;
    filters.n_fft = 201;
    std::vector data(filters.n_mel * filters.n_fft, 0.0f);

    data[1] = 12.37398665;
    data[202] = 30.39256483;
    data[404] = 24.74797331;
    data[605] = 18.01857911;
    data[807] = 37.12195903;
    data[1008] = 5.64459199;
    data[1009] = 6.72939420;
    data[1210] = 36.03715822;
    data[1412] = 19.10337992;
    data[1613] = 23.66316877;
    data[1815] = 31.47736564;
    data[2016] = 11.28918398;
    data[2017] = 1.08480197;
    data[2218] = 41.68175161;
    data[2420] = 13.45878839;
    data[2621] = 29.30776216;
    data[2823] = 25.83277412;
    data[3024] = 16.93377644;
    data[3226] = 38.20675984;
    data[3427] = 4.55979025;
    data[3428] = 7.81419594;
    data[3629] = 34.95235741;
    data[3831] = 20.18818259;
    data[4032] = 22.57836796;
    data[4234] = 32.56217018;
    data[4435] = 10.20438317;
    data[4436] = 2.16960395;
    data[4637] = 40.59694707;
    data[4839] = 14.54358920;
    data[5040] = 28.22295949;
    data[5242] = 26.91757679;
    data[5443] = 15.84897563;
    data[5645] = 39.29156065;
    data[5846] = 3.47498828;
    data[5847] = 8.89899861;
    data[6048] = 33.86755288;
    data[6250] = 21.27298526;
    data[6451] = 21.49356715;
    data[6653] = 33.64697099;
    data[6854] = 9.11958050;
    data[6855] = 3.25440569;
    data[7056] = 39.51214626;
    data[7258] = 15.62839188;
    data[7459] = 27.13815868;
    data[7661] = 28.00237760;
    data[7862] = 14.76417296;
    data[8064] = 40.37636518;
    data[8265] = 2.38068704;
    data[8266] = 10.20263787;
    data[8467] = 31.61146119;
    data[8669] = 24.54700135;
    data[8870] = 15.32919332;
    data[8871] = 1.66583748;
    data[9072] = 36.72905266;
    data[9274] = 20.09709924;
    data[9475] = 16.93102531;
    data[9476] = 2.90265540;
    data[9677] = 32.84499049;
    data[9879] = 23.52004871;
    data[10080] = 11.03894413;
    data[10081] = 10.72582975;
    data[10282] = 22.71829173;
    data[10484] = 32.27872774;
    data[10685] = 0.11626833;
    data[10686] = 22.85348251;
    data[10887] = 8.56344029;
    data[10888] = 14.97978810;
    data[11089] = 15.51398356;
    data[11090] = 8.51490628;
    data[11291] = 21.10680379;
    data[11292] = 3.32652032;
    data[11493] = 25.47064796;
    data[11695] = 27.35907957;
    data[11896] = 0.65853616;
    data[11897] = 23.83812517;
    data[12098] = 3.44359246;
    data[12099] = 21.22455277;
    data[12300] = 5.35842171;
    data[12301] = 19.42555793;
    data[12502] = 6.49324711;
    data[12503] = 18.35542172;
    data[12704] = 6.93138083;
    data[12705] = 17.93504693;
    data[12906] = 6.74968259;
    data[12907] = 18.09151843;
    data[13108] = 6.01899112;
    data[13109] = 18.75767298;
    data[13310] = 4.80452832;
    data[13311] = 19.87172849;
    data[13512] = 3.16627859;
    data[13513] = 21.37690969;
    data[13514] = 1.25317345;
    data[13714] = 1.15934468;
    data[13715] = 20.80361731;
    data[13716] = 4.04486805;
    data[13917] = 17.55363122;
    data[13918] = 7.08320038;
    data[14119] = 14.07538634;
    data[14120] = 10.32655034;
    data[14321] = 10.40921453;
    data[14322] = 13.73696327;
    data[14523] = 6.59187697;
    data[14524] = 17.27988198;
    data[14525] = 1.46804214;
    data[14725] = 2.65681883;
    data[14726] = 18.09193194;
    data[14727] = 5.85655728;
    data[14928] = 13.34277913;
    data[14929] = 10.28267574;
    data[15130] = 8.56800377;
    data[15131] = 14.72230814;
    data[15132] = 1.04039861;
    data[15332] = 3.79085587;
    data[15333] = 17.14678481;
    data[15334] = 6.11609267;
    data[15535] = 11.75929047;
    data[15536] = 11.13393717;
    data[15737] = 6.43857848;
    data[15738] = 16.07806236;
    data[15739] = 4.23917221;
    data[15939] = 1.19989377;
    data[15940] = 12.75671553;
    data[15941] = 9.65298992;
    data[16142] = 7.06935255;
    data[16143] = 14.94054683;
    data[16144] = 4.19024844;
    data[16344] = 1.51483389;
    data[16345] = 12.00899947;
    data[16346] = 9.84823331;
    data[16547] = 6.10224018;
    data[16548] = 15.33857174;
    data[16549] = 5.57676842;
    data[16749] = 0.36827257;
    data[16750] = 9.89749376;
    data[16751] = 11.35340426;
    data[16752] = 2.05122307;
    data[16952] = 3.89297144;
    data[16953] = 12.97352277;
    data[16954] = 8.06631614;
    data[17155] = 6.74493238;
    data[17156] = 13.85874674;
    data[17157] = 5.41190524;
    data[17357] = 0.74220158;
    data[17358] = 8.98779090;
    data[17359] = 11.37871388;
    data[17360] = 3.32958088;
    data[17560] = 2.82313535;
    data[17561] = 10.68049297;
    data[17562] = 9.43340641;
    data[17563] = 1.76325557;
    data[17763] = 4.39018616;
    data[17764] = 11.87758986;
    data[17765] = 7.97005836;
    data[17766] = 0.66104700;
    data[17966] = 5.49466675;
    data[17967] = 12.62953598;
    data[17968] = 6.93987962;
    data[18169] = 6.18401915;
    data[18170] = 12.93473132;
    data[18171] = 6.29778765;
    data[18371] = 0.02325210;
    data[18372] = 6.50206627;
    data[18373] = 12.32661773;
    data[18374] = 6.00216538;
    data[18574] = 0.31548753;
    data[18575] = 6.48925547;
    data[18576] = 12.04130240;
    data[18577] = 6.01462880;
    data[18777] = 0.29979556;
    data[18778] = 6.18288014;
    data[18779] = 12.04272825;
    data[18780] = 6.29981188;
    data[18781] = 0.55689598;
    data[18980] = 0.01120471;
    data[18981] = 5.61729167;
    data[18982] = 11.22337859;
    data[18983] = 6.82516303;
    data[18984] = 1.35264499;
    data[19184] = 4.82410006;
    data[19185] = 10.16623247;
    data[19186] = 7.56075513;
    data[19187] = 2.34590308;
    data[19387] = 3.83235747;
    data[19388] = 8.92296247;
    data[19389] = 8.47910438;
    data[19390] = 3.50978645;
    data[19590] = 2.66873185;
    data[19591] = 7.51965167;
    data[19592] = 9.55500547;
    data[19593] = 4.81966138;
    data[19594] = 0.08431751;
    data[19793] = 1.35767367;
    data[19794] = 5.98019501;
    data[19795] = 10.60271543;
    data[19796] = 6.25298498;
    data[19797] = 1.74059917;
    data[19997] = 4.32644226;
    data[19998] = 8.73131864;
    data[19999] = 7.78916525;
    data[20000] = 3.48923868;
    data[20200] = 2.57835095;
    data[20201] = 6.77582854;
    data[20202] = 9.40941647;
    data[20203] = 5.31194592;
    data[20204] = 1.21447595;
    data[20403] = 0.75411191;
    data[20404] = 4.75395704;
    data[20405] = 8.75380263;
    data[20406] = 7.19209015;
    data[20407] = 3.28754401;
    data[20607] = 2.68179690;
    data[20608] = 6.49331464;
    data[20609] = 9.11457930;
    data[20610] = 5.39387390;
    data[20611] = 1.67316827;
    data[20810] = 0.57394296;
    data[20811] = 4.20600036;
    data[20812] = 7.83805829;
    data[20813] = 7.52023002;
    data[20814] = 3.97470826;
    data[20815] = 0.42918732;
    data[21014] = 1.90464477;
    data[21015] = 5.36569161;
    data[21016] = 8.82673822;
    data[21017] = 6.27609482;
    data[21018] = 2.89750961;
    data[21218] = 2.89885257;
    data[21219] = 6.19694078;
    data[21220] = 8.56699049;
    data[21221] = 5.34748193;
    data[21222] = 2.12797290;
    data[21421] = 0.44750227;
    data[21422] = 3.59030394;
    data[21423] = 6.73310598;
    data[21424] = 7.77023612;
    data[21425] = 4.70231380;
    data[21426] = 1.63439126;
    data[21625] = 1.01536023;
    data[21626] = 4.01018746;
    data[21627] = 7.00501446;
    data[21628] = 7.23442994;
    data[21629] = 4.31095669;
    data[21630] = 1.38748321;
    data[21829] = 1.33348850;
    data[21830] = 4.18730825;
    data[21831] = 7.04112789;
    data[21832] = 6.93188375;
    data[21833] = 4.14605811;
    data[21834] = 1.36023236;
    data[22033] = 1.42879714;
    data[22034] = 4.14824858;
    data[22035] = 6.86769979;
    data[22036] = 6.83705276;
    data[22037] = 4.18239459;
    data[22038] = 1.52773573;
    data[22237] = 1.32610439;
    data[22238] = 3.91751388;
    data[22239] = 6.50892360;
    data[22240] = 6.92639686;
    data[22241] = 4.39672917;
    data[22242] = 1.86706171;
    data[22441] = 1.04827771;
    data[22442] = 3.51767405;
    data[22443] = 5.98707050;
    data[22444] = 7.17824046;
    data[22445] = 4.76767914;
    data[22446] = 2.35711760;
    data[22645] = 0.61636406;
    data[22646] = 2.96949223;
    data[22647] = 5.32262027;
    data[22648] = 7.57265091;
    data[22649] = 5.27558755;
    data[22650] = 2.97852419;
    data[22651] = 0.68146095;
    data[22849] = 0.04971400;
    data[22850] = 2.29204819;
    data[22851] = 4.53438237;
    data[22852] = 6.77671656;
    data[22853] = 5.90240723;
    data[22854] = 3.71349836;
    data[22855] = 1.52458926;
    data[23054] = 1.50285335;
    data[23055] = 3.63961048;
    data[23056] = 5.77636715;
    data[23057] = 6.63159089;
    data[23058] = 4.54574358;
    data[23059] = 2.45989650;
    data[23060] = 0.37404924;
    data[23258] = 0.61795861;
    data[23259] = 2.65410915;
    data[23260] = 4.69025923;
    data[23261] = 6.72641024;
    data[23262] = 5.46034705;
    data[23263] = 3.47270933;
    data[23264] = 1.48507138;
    data[23463] = 1.59233576;
    data[23464] = 3.53261665;
    data[23465] = 5.47289755;
    data[23466] = 6.44368259;
    data[23467] = 4.54962999;
    data[23468] = 2.65557761;
    data[23469] = 0.76152512;
    data[23667] = 0.46749352;
    data[23668] = 2.31641904;
    data[23669] = 4.16534441;
    data[23670] = 6.01426978;
    data[23671] = 5.67844696;
    data[23672] = 3.87357362;
    data[23673] = 2.06870004;
    data[23674] = 0.26382666;
    data[23872] = 1.05349103;
    data[23873] = 2.81536230;
    data[23874] = 4.57723346;
    data[23875] = 6.33910485;
    data[23876] = 5.12815686;
    data[23877] = 3.40826320;
    data[23878] = 1.68837002;
    data[24077] = 1.43350090;
    data[24078] = 3.11241671;
    data[24079] = 4.79133241;
    data[24080] = 6.40943693;
    data[24081] = 4.77052201;
    data[24082] = 3.13160778;
    data[24083] = 1.49269309;
    data[24281] = 0.02932359;
    data[24282] = 1.62918994;
    data[24283] = 3.22905602;
    data[24284] = 4.82892245;
    data[24285] = 6.14671456;
    data[24286] = 4.58496623;
    data[24287] = 3.02321767;
    data[24288] = 1.46146910;
    data[24486] = 0.13601698;
    data[24487] = 1.66055572;
    data[24488] = 3.18509457;
    data[24489] = 4.70963307;
    data[24490] = 6.04072399;
    data[24491] = 4.55250870;
    data[24492] = 3.06429295;
    data[24493] = 1.57607743;
    data[24494] = 0.08786193;
    data[24691] = 0.09328097;
    data[24692] = 1.54603878;
    data[24693] = 2.99879676;
    data[24694] = 4.45155473;
    data[24695] = 5.90431225;
    data[24696] = 4.65566106;
    data[24697] = 3.23751615;
    data[24698] = 1.81937125;
    data[24699] = 0.40122634;
    data[24897] = 1.30262633;
    data[24898] = 2.68698297;
    data[24899] = 4.07133950;
    data[24900] = 5.45569602;
    data[24901] = 4.87832492;
    data[24902] = 3.52695142;
    data[24903] = 2.17557792;
    data[24904] = 0.82420459;
    data[25102] = 0.94595028;
    data[25103] = 2.26512621;
    data[25104] = 3.58430226;
    data[25105] = 4.90347855;
    data[25106] = 5.20569785;
    data[25107] = 3.91795207;
    data[25108] = 2.63020652;
    data[25109] = 1.34246063;
    data[25110] = 0.05471494;
    data[25307] = 0.49037894;
    data[25308] = 1.74744334;
    data[25309] = 3.00450763;
    data[25310] = 4.26157191;
    data[25311] = 5.51863620;
    data[25312] = 4.39707236;
    data[25313] = 3.16995848;
    data[25314] = 1.94284460;
    data[25315] = 0.71573065;
    data[25513] = 1.14698056;
    data[25514] = 2.34485767;
    data[25515] = 3.54273478;
    data[25516] = 4.74061165;
    data[25517] = 4.95198462;
    data[25518] = 3.78264743;
    data[25519] = 2.61331047;
    data[25520] = 1.44397374;
    data[25521] = 0.27463681;
    data[25718] = 0.47569509;
    data[25719] = 1.61717169;
    data[25720] = 2.75864848;
    data[25721] = 3.90012516;
    data[25722] = 5.04160160;
    data[25723] = 4.45712078;
    data[25724] = 3.34284059;
    data[25725] = 2.22856039;
    data[25726] = 1.11428020;

    for (auto & val : data) {
        val /= 1000.0f;
    }

    filters.data = std::move(data);
    return filters;
}

} // namespace whisper_precalc_filters
