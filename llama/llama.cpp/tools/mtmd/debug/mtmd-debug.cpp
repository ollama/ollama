#include "mtmd-debug.h"

#include "arg.h"
#include "debug.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <cmath>
#include <limits.h>
#include <cinttypes>
#include <clocale>

// INTERNAL TOOL FOR DEBUGGING PURPOSES ONLY
// NOT INTENDED FOR PUBLIC USE

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Internal debugging tool for mtmd; See mtmd-debug.md for the pytorch equivalent code\n"
        "Note: we repurpose some args from other examples, they will have different meaning here\n"
        "\n"
        "Usage: %s -m <model> --mmproj <mmproj> -p <mode> -n <size> --image <image> --audio <audio>\n"
        "\n"
        "    -n <size>: number of pixels per edge for image (always square image), or number of samples for audio\n"
        "\n"
        "    -p \"encode\" (debugging encode pass, default case):\n"
        "        --image can be:\n"
        "          \"white\", \"black\", \"gray\": filled 1.0f, 0.0f and 0.5f respectively\n"
        "          \"cb\": checkerboard pattern, alternate 1.0f and 0.0f\n"
        "        --audio can be:\n"
        "          \"one\", \"zero\", \"half\": filled 1.0f, 0.0f and 0.5f respectively\n"
        "          \"1010\": checkerboard pattern, alternate 1.0f and 0.0f\n"
        "\n"
        "    -p \"preproc\" (debugging preprocessing pass):\n"
        "        --image can be:\n"
        "          \"white\", \"black\", \"gray\": filled image with respective colors\n"
        "          \"cb\": checkerboard pattern\n"
        "        --audio can be:\n"
        "          \"one\", \"zero\", \"half\": filled 1.0f, 0.0f and 0.5f respectively\n"
        "          \"440\": sine wave with 440 Hz frequency\n"
        "\n",
        argv[0]
    );
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    LOG_INF("%s: loading model: %s\n", __func__, params.model.path.c_str());

    mtmd::context_ptr ctx_mtmd;
    common_init_result_ptr llama_init;
    base_callback_data cb_data;

    llama_init = common_init_from_params(params);
    {
        auto * model = llama_init->model();
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.print_timings    = true;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        {
            // always enable debug callback
            mparams.cb_eval_user_data = &cb_data;
            mparams.cb_eval = common_debug_cb_eval<false>;
        }
        ctx_mtmd.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_mtmd.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }
    }

    std::string input;
    int32_t inp_size = params.n_predict;
    if (params.image.empty()) {
        LOG_ERR("ERR: At least one of --image or --audio must be specified\n");
        return 1;
    }
    if (inp_size <= 0) {
        LOG_ERR("ERR: Invalid size specified with -n, must be greater than 0\n");
        return 1;
    }
    input = params.image[0];

    if (params.prompt.empty() || params.prompt == "encode") {
        std::vector<std::vector<float>> image;
        std::vector<float> samples;

        if (input == "black") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 0.0f);
                image.push_back(row);
            }
        } else if (input == "white") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 1.0f);
                image.push_back(row);
            }
        } else if (input == "gray") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 0.5f);
                image.push_back(row);
            }
        } else if (input == "cb") {
            for (int i = 0; i < inp_size; ++i) {
                auto row = std::vector<float>(inp_size * 3, 0.0f);
                image.push_back(row);
            }
            for (int y = 0; y < inp_size; ++y) {
                for (int x = 0; x < inp_size; ++x) {
                    float v = ((x + y) % 2) ? 0.0f : 1.0f;
                    image[y][x * 3 + 0] = v;
                    image[y][x * 3 + 1] = v;
                    image[y][x * 3 + 2] = v;
                }
            }
        } else if (input == "one") {
            samples = std::vector<float>(inp_size, 1.0f);
        } else if (input == "zero") {
            samples = std::vector<float>(inp_size, 0.0f);
        } else if (input == "half") {
            samples = std::vector<float>(inp_size, 0.5f);
        } else if (input == "1010") {
            samples.resize(inp_size);
            for (int i = 0; i < inp_size; ++i) {
                samples[i] = (i % 2) ? 0.0f : 1.0f;
            }
        } else {
            LOG_ERR("ERR: Invalid input specified with --image/--audio\n");
            show_additional_info(argc, argv);
            return 1;
        }

        // run encode pass
        LOG_INF("Running encode pass for input type: %s\n", input.c_str());
        if (samples.size() > 0) {
            LOG_INF("Input audio with %zu samples, type: %s\n", samples.size(), input.c_str());
            mtmd_debug_encode_audio(ctx_mtmd.get(), samples);
        } else {
            LOG_INF("Input image with dimensions %d x %d, type: %s\n", inp_size, inp_size, input.c_str());
            mtmd_debug_encode_image(ctx_mtmd.get(), image);
        }

    } else if (params.prompt == "preproc") {
        std::vector<uint8_t> rgb_values;
        std::vector<float> pcm_samples;

        if (input == "black") {
            rgb_values = std::vector<uint8_t>(inp_size * inp_size * 3, 0);
        } else if (input == "white") {
            rgb_values = std::vector<uint8_t>(inp_size * inp_size * 3, 255);
        } else if (input == "gray") {
            rgb_values = std::vector<uint8_t>(inp_size * inp_size * 3, 128);
        } else if (input == "cb") {
            rgb_values.resize(inp_size * inp_size * 3);
            for (int y = 0; y < inp_size; ++y) {
                for (int x = 0; x < inp_size; ++x) {
                    uint8_t v = ((x + y) % 2) ? 0 : 255;
                    rgb_values[(y * inp_size + x) * 3 + 0] = v;
                    rgb_values[(y * inp_size + x) * 3 + 1] = v;
                    rgb_values[(y * inp_size + x) * 3 + 2] = v;
                }
            }
        } else if (input == "one") {
            pcm_samples = std::vector<float>(inp_size, 1.0f);
        } else if (input == "zero") {
            pcm_samples = std::vector<float>(inp_size, 0.0f);
        } else if (input == "half") {
            pcm_samples = std::vector<float>(inp_size, 0.5f);
        } else if (input == "440") {
            pcm_samples.resize(inp_size);
            float freq = 440.0f;
            float sample_rate = mtmd_get_audio_sample_rate(ctx_mtmd.get());
            float pi = 3.14159265f;
            for (int i = 0; i < inp_size; ++i) {
                pcm_samples[i] = sinf(2 * pi * freq * i / sample_rate);
            }
        } else {
            LOG_ERR("ERR: Invalid input specified with --image/--audio\n");
            show_additional_info(argc, argv);
            return 1;
        }

        // run preprocessing pass
        LOG_INF("Running preprocessing pass for input type: %s\n", input.c_str());
        if (pcm_samples.size() > 0) {
            LOG_INF("Input audio with %zu samples, type: %s\n", pcm_samples.size(), input.c_str());
            mtmd_debug_preprocess_audio(ctx_mtmd.get(), pcm_samples);
        } else {
            LOG_INF("Input image with dimensions %d x %d, type: %s\n", inp_size, inp_size, input.c_str());
            mtmd_debug_preprocess_image(ctx_mtmd.get(), rgb_values, inp_size, inp_size);
        }

    } else {
        LOG_ERR("ERR: Invalid mode specified with -p\n");
        show_additional_info(argc, argv);
        return 1;
    }

    return 0;
}

