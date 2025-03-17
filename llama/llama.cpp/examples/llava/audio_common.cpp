#define _USE_MATH_DEFINES // for M_PI

#include "audio_common.h"

// third-party utilities
// use your favorite implementations
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <cmath>
#include <codecvt>
#include <cstring>
#include <fstream>
#include <regex>
#include <locale>
#include <iostream>
#include <sstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifdef WHISPER_FFMPEG
// as implemented in ffmpeg_trancode.cpp only embedded in common lib if whisper built with ffmpeg support
extern bool ffmpeg_decode_audio(const std::string & ifname, std::vector<uint8_t> & wav_data);
#endif

bool is_wav_buffer(const std::string& buf) {
    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    if (buf.size() < 12 || buf.substr(0, 4) != "RIFF" || buf.substr(8, 4) != "WAVE") {
        return false;
    }

    uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(buf.data() + 4);
    if (chunk_size + 8 != buf.size()) {
        return false;
    }

    return true;
}

bool read_binary_file(const std::string file_path, std::vector<uint8_t>* buf) {
    // printf("read_binary_file 1 :%s\n", file_path.c_str());
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return false;
    }
    // printf("read_binary_file 2 :%s\n", file_path.c_str());
    // 获取文件大小
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // printf("read_binary_file 3 :%s\n", file_path.c_str());
    // 读取文件内容到缓冲区
    buf->resize(file_size);
    if (!file.read(reinterpret_cast<char*>(buf->data()), file_size)) {
        std::cerr << "Failed to read file: " << file_path << std::endl;
        return false;
    }

    // printf("read_binary_file 4 :%s\n", file_path.c_str());
    // 关闭文件
    file.close();
    return true;
}

bool read_wav(const std::string & fname, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo) {
    drwav wav;
    std::vector<uint8_t> wav_data; // used for pipe input from stdin or ffmpeg decoding output

    if (fname == "-") {
        {
            #ifdef _WIN32
            _setmode(_fileno(stdin), _O_BINARY);
            #endif

            uint8_t buf[1024];
            while (true)
            {
                const size_t n = fread(buf, 1, sizeof(buf), stdin);
                if (n == 0) {
                    break;
                }
                wav_data.insert(wav_data.end(), buf, buf + n);
            }
        }

        if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false) {
            fprintf(stderr, "error: failed to open WAV file from stdin\n");
            return false;
        }

        fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, wav_data.size());
    }
    else if (is_wav_buffer(fname)) {
        if (drwav_init_memory(&wav, fname.c_str(), fname.size(), nullptr) == false) {
            fprintf(stderr, "error: failed to open WAV file from fname buffer\n");
            return false;
        }
    }
    else if (drwav_init_file(&wav, fname.c_str(), nullptr) == false) {
#if defined(WHISPER_FFMPEG)
        if (ffmpeg_decode_audio(fname, wav_data) != 0) {
            fprintf(stderr, "error: failed to ffmpeg decode '%s' \n", fname.c_str());
            return false;
        }
        if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false) {
            fprintf(stderr, "error: failed to read wav data as wav \n");
            return false;
        }
#else
        fprintf(stderr, "error: failed to open '%s' as WAV file\n", fname.c_str());
        return false;
#endif
    }

    if (wav.channels != 1 && wav.channels != 2) {
        fprintf(stderr, "%s: WAV file '%s' must be mono or stereo\n", __func__, fname.c_str());
        drwav_uninit(&wav);
        return false;
    }

    if (stereo && wav.channels != 2) {
        fprintf(stderr, "%s: WAV file '%s' must be stereo for diarization\n", __func__, fname.c_str());
        drwav_uninit(&wav);
        return false;
    }

    if (wav.sampleRate != COMMON_SAMPLE_RATE) {
        fprintf(stderr, "%s: WAV file '%s' must be %i kHz\n", __func__, fname.c_str(), COMMON_SAMPLE_RATE/1000);
        drwav_uninit(&wav);
        return false;
    }

    if (wav.bitsPerSample != 16) {
        fprintf(stderr, "%s: WAV file '%s' must be 16-bit\n", __func__, fname.c_str());
        drwav_uninit(&wav);
        return false;
    }

    const uint64_t n = wav_data.empty() ? wav.totalPCMFrameCount : wav_data.size()/(wav.channels*wav.bitsPerSample/8);

    std::vector<int16_t> pcm16;
    pcm16.resize(n*wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    // convert to mono, float
    pcmf32.resize(n);
    if (wav.channels == 1) {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[i])/32768.0f;
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
        }
    }

    if (stereo) {
        // convert to stereo, float
        pcmf32s.resize(2);

        pcmf32s[0].resize(n);
        pcmf32s[1].resize(n);
        for (uint64_t i = 0; i < n; i++) {
            pcmf32s[0][i] = float(pcm16[2*i])/32768.0f;
            pcmf32s[1][i] = float(pcm16[2*i + 1])/32768.0f;
        }
    }

    return true;
}

std::vector<audio_segment> slice_audio(const std::vector<float>& pcmf32,
                                    const std::vector<std::vector<float>>& pcmf32s,
                                    int sample_rate,
                                    float segment_duration,
                                    float overlap) {
    std::vector<audio_segment> segments;
    bool is_stereo = !pcmf32s.empty();

    // Calculate samples per segment and overlap
    int samples_per_segment = static_cast<int>(sample_rate * segment_duration);
    int overlap_samples = static_cast<int>(sample_rate * overlap);
    int step_size = samples_per_segment - overlap_samples;

    if (is_stereo) {
        // Process stereo audio
        size_t total_samples = pcmf32s[0].size();
        size_t num_segments = (total_samples + step_size - 1) / step_size;

        for (size_t i = 0; i < num_segments; ++i) {
            size_t start_idx = i * step_size;
            size_t end_idx = std::min(start_idx + samples_per_segment, total_samples);

            audio_segment segment;
            segment.is_stereo = true;
            segment.start_time = static_cast<double>(start_idx) / sample_rate;

            // Create stereo segment
            segment.stereo.resize(2);
            for (int ch = 0; ch < 2; ++ch) {
                segment.stereo[ch].resize(end_idx - start_idx);
                std::copy(pcmf32s[ch].begin() + start_idx,
                         pcmf32s[ch].begin() + end_idx,
                         segment.stereo[ch].begin());
            }

            segments.push_back(std::move(segment));
        }
    } else {
        // Process mono audio
        size_t total_samples = pcmf32.size();
        size_t num_segments = (total_samples + step_size - 1) / step_size;

        for (size_t i = 0; i < num_segments; ++i) {
            size_t start_idx = i * step_size;
            size_t end_idx = std::min(start_idx + samples_per_segment, total_samples);

            audio_segment segment;
            segment.is_stereo = false;
            segment.start_time = static_cast<double>(start_idx) / sample_rate;

            // Create mono segment
            segment.mono.resize(end_idx - start_idx);
            std::copy(pcmf32.begin() + start_idx,
                     pcmf32.begin() + end_idx,
                     segment.mono.begin());

            segments.push_back(std::move(segment));
        }
    }
    return segments;
}

void high_pass_filter(std::vector<float> & data, float cutoff, float sample_rate) {
    const float rc = 1.0f / (2.0f * M_PI * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];

    for (size_t i = 1; i < data.size(); i++) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

bool vad_simple(std::vector<float> & pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
    const int n_samples      = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        // not enough samples - assume no speech
        return false;
    }

    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }

    float energy_all  = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        energy_all += fabsf(pcmf32[i]);

        if (i >= n_samples - n_samples_last) {
            energy_last += fabsf(pcmf32[i]);
        }
    }

    energy_all  /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        fprintf(stderr, "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n", __func__, energy_all, energy_last, vad_thold, freq_thold);
    }

    if (energy_last > vad_thold*energy_all) {
        return false;
    }

    return true;
}

float similarity(const std::string & s0, const std::string & s1) {
    const size_t len0 = s0.size() + 1;
    const size_t len1 = s1.size() + 1;

    std::vector<int> col(len1, 0);
    std::vector<int> prevCol(len1, 0);

    for (size_t i = 0; i < len1; i++) {
        prevCol[i] = i;
    }

    for (size_t i = 0; i < len0; i++) {
        col[0] = i;
        for (size_t j = 1; j < len1; j++) {
            col[j] = std::min(std::min(1 + col[j - 1], 1 + prevCol[j]), prevCol[j - 1] + (i > 0 && s0[i - 1] == s1[j - 1] ? 0 : 1));
        }
        col.swap(prevCol);
    }

    const float dist = prevCol[len1 - 1];

    return 1.0f - (dist / std::max(s0.size(), s1.size()));
}

bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

// bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id)
// {
//     std::ofstream speak_file(path.c_str());
//     if (speak_file.fail()) {
//         fprintf(stderr, "%s: failed to open speak_file\n", __func__);
//         return false;
//     } else {
//         speak_file.write(text.c_str(), text.size());
//         speak_file.close();
//         int ret = system((command + " " + std::to_string(voice_id) + " " + path).c_str());
//         if (ret != 0) {
//             fprintf(stderr, "%s: failed to speak\n", __func__);
//             return false;
//         }
//     }
//     return true;
// }
