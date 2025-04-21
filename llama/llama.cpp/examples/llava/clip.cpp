// NOTE: This is modified from clip.cpp only for LLaVA,
// so there might be still unnecessary artifacts hanging around
// I'll gradually clean and extend it
// Note: Even when using identical normalized image inputs (see normalize_image_u8_to_f32()) we have a significant difference in resulting embeddings compared to pytorch
#include "clip.h"
#include "clip-impl.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <cinttypes>
#include <limits>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>
#if __GLIBCXX__
#include <cstdio>
#include <ext/stdio_filebuf.h>
#include <fcntl.h>
#endif
#endif

struct clip_logger_state g_logger_state = {GGML_LOG_LEVEL_CONT, clip_log_callback_default, NULL};

//#define CLIP_DEBUG_FUNCTIONS

#ifdef CLIP_DEBUG_FUNCTIONS
static void clip_image_write_image_to_ppm(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    // PPM header: P6 format, width, height, and max color value
    file << "P6\n" << img.nx << " " << img.ny << "\n255\n";

    // Write pixel data
    for (size_t i = 0; i < img.buf.size(); i += 3) {
        // PPM expects binary data in RGB format, which matches our image buffer
        file.write(reinterpret_cast<const char*>(&img.buf[i]), 3);
    }

    file.close();
}

static void clip_image_save_to_bmp(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    int fileSize = 54 + 3 * img.nx * img.ny; // File header + info header + pixel data
    int bytesPerPixel = 3;
    int widthInBytes = img.nx * bytesPerPixel;
    int paddingAmount = (4 - (widthInBytes % 4)) % 4;
    int stride = widthInBytes + paddingAmount;

    // Bitmap file header
    unsigned char fileHeader[14] = {
        'B','M',     // Signature
        0,0,0,0,    // Image file size in bytes
        0,0,0,0,    // Reserved
        54,0,0,0    // Start of pixel array
    };

    // Total file size
    fileSize = 54 + (stride * img.ny);
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    // Bitmap information header (BITMAPINFOHEADER)
    unsigned char infoHeader[40] = {
        40,0,0,0,   // Size of this header (40 bytes)
        0,0,0,0,    // Image width
        0,0,0,0,    // Image height
        1,0,        // Number of color planes
        24,0,       // Bits per pixel
        0,0,0,0,    // No compression
        0,0,0,0,    // Image size (can be 0 for no compression)
        0,0,0,0,    // X pixels per meter (not specified)
        0,0,0,0,    // Y pixels per meter (not specified)
        0,0,0,0,    // Total colors (color table not used)
        0,0,0,0     // Important colors (all are important)
    };

    // Width and height in the information header
    infoHeader[4] = (unsigned char)(img.nx);
    infoHeader[5] = (unsigned char)(img.nx >> 8);
    infoHeader[6] = (unsigned char)(img.nx >> 16);
    infoHeader[7] = (unsigned char)(img.nx >> 24);
    infoHeader[8] = (unsigned char)(img.ny);
    infoHeader[9] = (unsigned char)(img.ny >> 8);
    infoHeader[10] = (unsigned char)(img.ny >> 16);
    infoHeader[11] = (unsigned char)(img.ny >> 24);

    // Write file headers
    file.write(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char*>(infoHeader), sizeof(infoHeader));

    // Pixel data
    std::vector<unsigned char> padding(3, 0); // Max padding size to be added to each row
    for (int y = img.ny - 1; y >= 0; --y) { // BMP files are stored bottom-to-top
        for (int x = 0; x < img.nx; ++x) {
            // Each pixel
            size_t pixelIndex = (y * img.nx + x) * 3;
            unsigned char pixel[3] = {
                img.buf[pixelIndex + 2], // BMP stores pixels in BGR format
                img.buf[pixelIndex + 1],
                img.buf[pixelIndex]
            };
            file.write(reinterpret_cast<char*>(pixel), 3);
        }
        // Write padding for the row
        file.write(reinterpret_cast<char*>(padding.data()), paddingAmount);
    }

    file.close();
}

// debug function to convert f32 to u8
static void clip_image_convert_f32_to_u8(const clip_image_f32& src, clip_image_u8& dst) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(3 * src.nx * src.ny);
    for (size_t i = 0; i < src.buf.size(); ++i) {
        dst.buf[i] = static_cast<uint8_t>(std::min(std::max(int(src.buf[i] * 255.0f), 0), 255));
    }
}
#endif


//
// clip layers
//

enum patch_merge_type {
    PATCH_MERGE_FLAT,
    PATCH_MERGE_SPATIAL_UNPAD,
};

struct clip_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;

    patch_merge_type mm_patch_merge_type = PATCH_MERGE_FLAT;

    float eps;

    std::vector<int32_t> image_grid_pinpoints;
    int32_t image_crop_resolution;
    std::unordered_set<int32_t> vision_feature_layer;
};

struct clip_layer {
    // attention
    struct ggml_tensor * k_w = nullptr;
    struct ggml_tensor * k_b = nullptr;
    struct ggml_tensor * q_w = nullptr;
    struct ggml_tensor * q_b = nullptr;
    struct ggml_tensor * v_w = nullptr;
    struct ggml_tensor * v_b = nullptr;

    struct ggml_tensor * o_w = nullptr;
    struct ggml_tensor * o_b = nullptr;

    // layernorm 1
    struct ggml_tensor * ln_1_w = nullptr;
    struct ggml_tensor * ln_1_b = nullptr;

    // ff
    struct ggml_tensor * ff_i_w = nullptr;
    struct ggml_tensor * ff_i_b = nullptr;

    struct ggml_tensor * ff_o_w = nullptr;
    struct ggml_tensor * ff_o_b = nullptr;

    // layernorm 2
    struct ggml_tensor * ln_2_w = nullptr;
    struct ggml_tensor * ln_2_b = nullptr;
};

struct clip_vision_model {
    struct clip_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding = nullptr;
    struct ggml_tensor * patch_embeddings_0 = nullptr;
    struct ggml_tensor * patch_embeddings_1 = nullptr;  // second Conv2D kernel when we decouple Conv3D along temproal dimension (Qwen2VL)
    struct ggml_tensor * patch_bias = nullptr;
    struct ggml_tensor * position_embeddings = nullptr;

    struct ggml_tensor * pre_ln_w = nullptr;
    struct ggml_tensor * pre_ln_b = nullptr;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;

    // LLaVA projection
    struct ggml_tensor * mm_0_w = nullptr;
    struct ggml_tensor * mm_0_b = nullptr;
    struct ggml_tensor * mm_2_w = nullptr;
    struct ggml_tensor * mm_2_b = nullptr;

    struct ggml_tensor * image_newline = nullptr;

    // Yi type models with mlp+normalization projection
    struct ggml_tensor * mm_1_w = nullptr; // Yi type models have 0, 1, 3, 4
    struct ggml_tensor * mm_1_b = nullptr;
    struct ggml_tensor * mm_3_w = nullptr;
    struct ggml_tensor * mm_3_b = nullptr;
    struct ggml_tensor * mm_4_w = nullptr;
    struct ggml_tensor * mm_4_b = nullptr;

    //GLMV-Edge projection
    struct ggml_tensor * mm_model_adapter_conv_w = nullptr;
    struct ggml_tensor * mm_model_adapter_conv_b = nullptr;
    struct ggml_tensor * boi_w = nullptr;
    struct ggml_tensor * eoi_w = nullptr;

    // MobileVLM projection
    struct ggml_tensor * mm_model_mlp_1_w = nullptr;
    struct ggml_tensor * mm_model_mlp_1_b = nullptr;
    struct ggml_tensor * mm_model_mlp_3_w = nullptr;
    struct ggml_tensor * mm_model_mlp_3_b = nullptr;
    struct ggml_tensor * mm_model_block_1_block_0_0_w = nullptr;
    struct ggml_tensor * mm_model_block_1_block_0_1_w = nullptr;
    struct ggml_tensor * mm_model_block_1_block_0_1_b = nullptr;
    struct ggml_tensor * mm_model_block_1_block_1_fc1_w = nullptr;
    struct ggml_tensor * mm_model_block_1_block_1_fc1_b = nullptr;
    struct ggml_tensor * mm_model_block_1_block_1_fc2_w = nullptr;
    struct ggml_tensor * mm_model_block_1_block_1_fc2_b = nullptr;
    struct ggml_tensor * mm_model_block_1_block_2_0_w = nullptr;
    struct ggml_tensor * mm_model_block_1_block_2_1_w = nullptr;
    struct ggml_tensor * mm_model_block_1_block_2_1_b = nullptr;
    struct ggml_tensor * mm_model_block_2_block_0_0_w = nullptr;
    struct ggml_tensor * mm_model_block_2_block_0_1_w = nullptr;
    struct ggml_tensor * mm_model_block_2_block_0_1_b = nullptr;
    struct ggml_tensor * mm_model_block_2_block_1_fc1_w = nullptr;
    struct ggml_tensor * mm_model_block_2_block_1_fc1_b = nullptr;
    struct ggml_tensor * mm_model_block_2_block_1_fc2_w = nullptr;
    struct ggml_tensor * mm_model_block_2_block_1_fc2_b = nullptr;
    struct ggml_tensor * mm_model_block_2_block_2_0_w = nullptr;
    struct ggml_tensor * mm_model_block_2_block_2_1_w = nullptr;
    struct ggml_tensor * mm_model_block_2_block_2_1_b = nullptr;

    // MobileVLM_V2 projection
    struct ggml_tensor * mm_model_mlp_0_w = nullptr;
    struct ggml_tensor * mm_model_mlp_0_b = nullptr;
    struct ggml_tensor * mm_model_mlp_2_w = nullptr;
    struct ggml_tensor * mm_model_mlp_2_b = nullptr;
    struct ggml_tensor * mm_model_peg_0_w = nullptr;
    struct ggml_tensor * mm_model_peg_0_b = nullptr;

    // MINICPMV projection
    struct ggml_tensor * mm_model_pos_embed_k = nullptr;
    struct ggml_tensor * mm_model_query = nullptr;
    struct ggml_tensor * mm_model_proj = nullptr;
    struct ggml_tensor * mm_model_kv_proj = nullptr;
    struct ggml_tensor * mm_model_attn_q_w = nullptr;
    struct ggml_tensor * mm_model_attn_q_b = nullptr;
    struct ggml_tensor * mm_model_attn_k_w = nullptr;
    struct ggml_tensor * mm_model_attn_k_b = nullptr;
    struct ggml_tensor * mm_model_attn_v_w = nullptr;
    struct ggml_tensor * mm_model_attn_v_b = nullptr;
    struct ggml_tensor * mm_model_attn_o_w = nullptr;
    struct ggml_tensor * mm_model_attn_o_b = nullptr;
    struct ggml_tensor * mm_model_ln_q_w = nullptr;
    struct ggml_tensor * mm_model_ln_q_b = nullptr;
    struct ggml_tensor * mm_model_ln_kv_w = nullptr;
    struct ggml_tensor * mm_model_ln_kv_b = nullptr;
    struct ggml_tensor * mm_model_ln_post_w = nullptr;
    struct ggml_tensor * mm_model_ln_post_b = nullptr;

    // gemma3
    struct ggml_tensor * mm_input_proj_w = nullptr;
    struct ggml_tensor * mm_soft_emb_norm_w = nullptr;
};

struct clip_ctx {
    bool has_text_encoder    = false;
    bool has_vision_encoder  = false;
    bool has_llava_projector = false;
    bool has_minicpmv_projector = false;
    bool has_glm_projector = false;
    bool has_qwen2vl_merger = false;
    int minicpmv_version = 2;

    struct clip_vision_model vision_model;
    projector_type proj_type = PROJECTOR_TYPE_MLP;

    int32_t max_feature_layer; // unused in newer models like gemma3
    float image_mean[3];
    float image_std[3];
    bool use_gelu = false;
    bool use_silu = false;

    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend;
    ggml_backend_t backend_cpu;
    ggml_backend_buffer_ptr buf;

    ggml_backend_sched_ptr sched;

    clip_image_size load_image_size;

    clip_ctx(clip_context_params & ctx_params) {
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        backend     = ctx_params.use_gpu
                        ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
                        : nullptr;

        if (backend) {
            LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: CLIP using CPU backend\n", __func__);
        }

        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), 8192, false)
        );
    }

    ~clip_ctx() {
        ggml_backend_free(backend);
        if (backend != backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

static ggml_cgraph * clip_image_build_graph_siglip(clip_ctx * ctx, const clip_image_f32_batch & imgs) {
    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size = hparams.image_size;
    int image_size_width  = image_size;
    int image_size_height = image_size;

    const int patch_size           = hparams.patch_size;
    const int num_patches          = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int hidden_size          = hparams.hidden_size;
    const int n_head               = hparams.n_head;
    const int d_head               = hidden_size / n_head;
    const int n_layer              = hparams.n_layer;
    const float eps                = hparams.eps;

    GGML_ASSERT(imgs.entries.size() == 1); // batch_size == 1

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx0_ptr(ggml_init(params));
    auto ctx0 = ctx0_ptr.get();

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // input raw
    struct ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, image_size_width, image_size_height, 3);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_2d(ctx0, inp, num_patches, hidden_size);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    inp = ggml_add(ctx0, inp, model.patch_bias);

    // position embeddings
    struct ggml_tensor * embeddings = ggml_add(ctx0, inp, model.position_embeddings);

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        struct ggml_tensor * cur = embeddings; // embeddings = residual, cur = hidden_states

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_1_w), model.layers[il].ln_1_b);
        }

        // self-attention
        {

            struct ggml_tensor * Q =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].q_w, cur), model.layers[il].q_b);

            Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, num_patches);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

            struct ggml_tensor * K =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].k_w, cur), model.layers[il].k_b);

            K = ggml_reshape_3d(ctx0, K, d_head, n_head, num_patches);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));

            struct ggml_tensor * V =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].v_w, cur), model.layers[il].v_b);

            V = ggml_reshape_3d(ctx0, V, d_head, n_head, num_patches);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_ext(ctx0, KQ, nullptr, 1.0f / sqrtf((float)d_head), 0.0f);

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_3d(ctx0, KQV, d_head, num_patches, n_head);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cont_2d(ctx0, KQV, hidden_size, num_patches);
        }

        // attention output
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].o_w, cur), model.layers[il].o_b);

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_2_w), model.layers[il].ln_2_b);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_i_b);

        // siglip uses gelu
        cur = ggml_gelu(ctx0, cur);

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_o_b);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        embeddings = ggml_norm(ctx0, embeddings, eps);
        ggml_set_name(embeddings, "post_ln");

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.post_ln_w), model.post_ln_b);
    }

    if (ctx->proj_type == PROJECTOR_TYPE_GEMMA3) {
        const int batch_size = 1;
        const int mm_tokens_per_image = 256; // default value for gemma3
        const int tokens_per_side = sqrt(mm_tokens_per_image);
        const int patches_per_image = sqrt(num_patches);
        const int kernel_size = patches_per_image / tokens_per_side;

        embeddings = ggml_cont(ctx0, ggml_transpose(ctx0, embeddings));
        embeddings = ggml_reshape_4d(ctx0, embeddings, patches_per_image, patches_per_image, hidden_size, batch_size);

        // doing a pool2d to reduce the number of output tokens to 256
        embeddings = ggml_pool_2d(ctx0, embeddings, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        embeddings = ggml_reshape_3d(ctx0, embeddings, embeddings->ne[0] * embeddings->ne[0], hidden_size, batch_size);
        embeddings = ggml_cont(ctx0, ggml_transpose(ctx0, embeddings));

        // apply norm before projection
        embeddings = ggml_rms_norm(ctx0, embeddings, eps);
        embeddings = ggml_mul(ctx0, embeddings, model.mm_soft_emb_norm_w);

        // apply projection
        embeddings = ggml_mul_mat(ctx0,
            ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_input_proj_w)),
            embeddings);
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}

static ggml_cgraph * clip_image_build_graph_legacy(clip_ctx * ctx, const clip_image_f32_batch & imgs, struct clip_image_size load_image_size, bool is_inf = false) {
    if (!ctx->has_vision_encoder) {
        LOG_ERR("This gguf file seems to have no vision encoder\n");
        return nullptr;
    }

    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size = hparams.image_size;
    int image_size_width  = image_size;
    int image_size_height = image_size;
    if (ctx->has_minicpmv_projector) {
        LOG_DBG("%s: %d %d\n", __func__, load_image_size.width, load_image_size.height);
        image_size_width  = load_image_size.width;
        image_size_height = load_image_size.height;
        if (is_inf) {
            image_size_width  = imgs.entries[0]->nx;
            image_size_height = imgs.entries[0]->ny;
        }
    }
    else if (ctx->has_qwen2vl_merger) {
        // use the image's native resolution when image is avaible
        if (is_inf) {
        // if (imgs->data->nx && imgs->data->ny) {
            image_size_width  = imgs.entries[0]->nx;
            image_size_height = imgs.entries[0]->ny;
        }
    }
    const int patch_size           = hparams.patch_size;
    const int num_patches          = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int patches_w            = image_size_width / patch_size;
    const int patches_h            = image_size_height / patch_size;
    const int num_positions        = num_patches + (model.class_embedding ? 1 : 0);
    const int num_position_ids     = ctx->has_qwen2vl_merger ? num_positions * 4 : num_positions;
    const int hidden_size          = hparams.hidden_size;
    const int n_head               = hparams.n_head;
    const int d_head               = hidden_size / n_head;
    const float eps                = hparams.eps;
    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    const int batch_size = imgs.entries.size();

    if (ctx->has_llava_projector || ctx->has_minicpmv_projector || ctx->has_glm_projector) {
        GGML_ASSERT(batch_size == 1);
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx0_ptr(ggml_init(params));
    auto ctx0 = ctx0_ptr.get();

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size_width, image_size_height, 3, batch_size);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    if (ctx->has_qwen2vl_merger) {
        GGML_ASSERT(image_size_width % (patch_size * 2) == 0);
        GGML_ASSERT(image_size_height % (patch_size * 2) == 0);

        auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_add(ctx0, inp, inp_1);
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 2, 0, 3));  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_reshape_4d(
            ctx0, inp,
            hidden_size * 2, patches_w / 2, patches_h, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            hidden_size * 2, patches_w / 2, 2, batch_size * (patches_h / 2));
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 0, 2, 1, 3));
        inp = ggml_reshape_3d(
            ctx0, inp,
            hidden_size, patches_w * patches_h, batch_size);
    }
    else {
        inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));
    }

    if (model.patch_bias) {
        // inp = ggml_add(ctx0, inp, ggml_repeat(ctx0, model.patch_bias, inp));
        inp = ggml_add(ctx0, inp, model.patch_bias);
    }
    struct ggml_tensor * embeddings = inp;
    struct ggml_tensor * pos_embed = nullptr;

    if (ctx->has_llava_projector) {
        // concat class_embeddings and patch_embeddings
        if (model.class_embedding) {
            embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size);
            ggml_set_name(embeddings, "embeddings");
            ggml_set_input(embeddings);
            embeddings = ggml_acc(ctx0, embeddings, model.class_embedding,
                    embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], 0);
            embeddings = ggml_acc(ctx0, embeddings, inp,
                    embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], model.class_embedding->nb[1]);
        }
    }

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    if (!ctx->has_qwen2vl_merger) { // qwen2vl use rope position embedding
        embeddings =
            ggml_add(ctx0, embeddings, ggml_get_rows(ctx0, model.position_embeddings, positions));
    }

    if (ctx->has_minicpmv_projector) {
        int pos_w = image_size_width/patch_size;
        int pos_h = image_size_height/patch_size;
        if (ctx->minicpmv_version == 2) {
            pos_embed = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 4096, pos_w * pos_h, 1);
        }
        else if (ctx->minicpmv_version == 3) {
            pos_embed = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 3584, pos_w * pos_h, 1);
        }
        else if (ctx->minicpmv_version == 4) {
            pos_embed = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 3584, pos_w * pos_h, 1);
        }
        ggml_set_name(pos_embed, "pos_embed");
        ggml_set_input(pos_embed);
    }

    // pre-layernorm
    if (model.pre_ln_w) {
        embeddings = ggml_norm(ctx0, embeddings, eps);
        ggml_set_name(embeddings, "pre_ln");

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.pre_ln_w), model.pre_ln_b);
    }

    std::vector<struct ggml_tensor *> embedding_stack;
    const auto & vision_feature_layer = hparams.vision_feature_layer;

    // loop over layers
    for (int il = 0; il < ctx->max_feature_layer; il++) {
        struct ggml_tensor * cur = embeddings; // embeddings = residual, cur = hidden_states

        // If this is an embedding feature layer, save the output.
        // NOTE: 0 index here refers to the input to the encoder.
        if (vision_feature_layer.find(il) != vision_feature_layer.end()) {
            embedding_stack.push_back(embeddings);
        }

        //const size_t nb_q_w = model.layers[il].q_w->nb[0];

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_1_w),
                           model.layers[il].ln_1_b);
        }

        // self-attention
        {

            struct ggml_tensor * Q =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].q_w, cur), model.layers[il].q_b);

            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_positions, batch_size);
            if (ctx->has_qwen2vl_merger) {
                Q = ggml_rope_multi(
                    ctx0, Q, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            }
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * K =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].k_w, cur), model.layers[il].k_b);

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
            if (ctx->has_qwen2vl_merger) {
                K = ggml_rope_multi(
                    ctx0, K, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            }
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * V =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].v_w, cur), model.layers[il].v_b);

            V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_ext(ctx0, KQ, nullptr, 1.0f / sqrtf((float)d_head), 0.0f);
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_positions, n_head, batch_size);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cont_3d(ctx0, KQV, hidden_size, num_positions, batch_size);
        }

        // attention output
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].o_w, cur), model.layers[il].o_b);

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_2_w), model.layers[il].ln_2_b);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_i_b);

        if (ctx->use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else if (ctx->use_silu) {
            cur = ggml_silu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_o_b);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        embeddings = ggml_norm(ctx0, embeddings, eps);
        ggml_set_name(embeddings, "post_ln");

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.post_ln_w), model.post_ln_b);
    }

    // final layer is a vision feature layer
    if (vision_feature_layer.find(ctx->max_feature_layer) != vision_feature_layer.end()) {
        embedding_stack.push_back(embeddings);
    }

    // If feature layers are explicitly set, stack them (if we have multiple)
    if (!embedding_stack.empty()) {
        embeddings = embedding_stack[0];
        for (size_t i = 1; i < embedding_stack.size(); i++) {
            embeddings = ggml_concat(ctx0, embeddings, embedding_stack[i], 0);
        }
    }

    // llava projector
    if (ctx->has_llava_projector) {
        embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

        struct ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_patches);
        ggml_set_name(patches, "patches");
        ggml_set_input(patches);

        // shape [1, 576, 1024]
        // ne is whcn, ne = [1024, 576, 1, 1]
        embeddings = ggml_get_rows(ctx0, embeddings, patches);

        // print_tensor_info(embeddings, "embeddings");

        // llava projector
        if (ctx->proj_type == PROJECTOR_TYPE_MLP) {
            embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

            embeddings = ggml_gelu(ctx0, embeddings);
            if (model.mm_2_w) {
                embeddings = ggml_mul_mat(ctx0, model.mm_2_w, embeddings);
                embeddings = ggml_add(ctx0, embeddings, model.mm_2_b);
            }
        }
        else if (ctx->proj_type == PROJECTOR_TYPE_MLP_NORM) {
            embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);
            // ggml_tensor_printf(embeddings, "mm_0_w",0,true,false);
            // First LayerNorm
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_1_w),
                                model.mm_1_b);

            // GELU activation
            embeddings = ggml_gelu(ctx0, embeddings);

            // Second linear layer
            embeddings = ggml_mul_mat(ctx0, model.mm_3_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_3_b);

            // Second LayerNorm
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_4_w),
                                model.mm_4_b);
        }
        else if (ctx->proj_type == PROJECTOR_TYPE_LDP) {
            // MobileVLM projector
            int n_patch = 24;
            struct ggml_tensor * mlp_1 = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w, embeddings);
            mlp_1 = ggml_add(ctx0, mlp_1, model.mm_model_mlp_1_b);
            mlp_1 = ggml_gelu(ctx0, mlp_1);
            struct ggml_tensor * mlp_3 = ggml_mul_mat(ctx0, model.mm_model_mlp_3_w, mlp_1);
            mlp_3 = ggml_add(ctx0, mlp_3, model.mm_model_mlp_3_b);
            // mlp_3 shape = [1, 576, 2048], ne = [2048, 576, 1, 1]

            // block 1
            struct ggml_tensor * block_1 = nullptr;
            {
                // transpose from [1, 576, 2048] --> [1, 2048, 576] --> [1, 2048, 24, 24]
                mlp_3 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_3, 1, 0, 2, 3));
                mlp_3 = ggml_reshape_4d(ctx0, mlp_3, n_patch, n_patch, mlp_3->ne[1], mlp_3->ne[2]);
                // stride = 1, padding = 1, bias is nullptr
                block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_1_block_0_0_w, mlp_3, 1, 1, 1, 1, 1, 1);

                // layer norm
                // // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_0_1_w), model.mm_model_block_1_block_0_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));

                // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                // hardswish
                struct ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                // pointwise conv
                block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc1_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc1_b);
                block_1 = ggml_relu(ctx0, block_1);
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc2_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc2_b);
                block_1 = ggml_hardsigmoid(ctx0, block_1);
                // block_1_hw shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1], block_1 shape = [1, 2048], ne = [2048, 1, 1, 1]
                block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                int w = block_1->ne[0], h = block_1->ne[1];
                block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));

                // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_2_0_w, block_1);
                block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);

                // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_2_1_w), model.mm_model_block_1_block_2_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // block1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                // residual
                block_1 = ggml_add(ctx0, mlp_3, block_1);
            }

            // block_2
            {
                // stride = 2
                block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_2_block_0_0_w, block_1, 2, 2, 1, 1, 1, 1);

                // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                // layer norm
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_0_1_w), model.mm_model_block_2_block_0_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                // hardswish
                struct ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                // not sure the parameters is right for globalAvgPooling
                block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                // pointwise conv
                block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc1_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc1_b);
                block_1 = ggml_relu(ctx0, block_1);
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc2_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc2_b);
                block_1 = ggml_hardsigmoid(ctx0, block_1);

                // block_1_hw shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1], block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                int w = block_1->ne[0], h = block_1->ne[1];
                block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));
                // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_2_0_w, block_1);
                block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);


                // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_2_1_w), model.mm_model_block_2_block_2_1_b);
                block_1 = ggml_reshape_3d(ctx0, block_1, block_1->ne[0], block_1->ne[1] * block_1->ne[2], block_1->ne[3]);
                // block_1 shape = [1, 144, 2048], ne = [2048, 144, 1]
            }
            embeddings = block_1;
        }
        else if (ctx->proj_type == PROJECTOR_TYPE_LDPV2)
        {
            int n_patch = 24;
            struct ggml_tensor * mlp_0 = ggml_mul_mat(ctx0, model.mm_model_mlp_0_w, embeddings);
            mlp_0 = ggml_add(ctx0, mlp_0, model.mm_model_mlp_0_b);
            mlp_0 = ggml_gelu(ctx0, mlp_0);
            struct ggml_tensor * mlp_2 = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, mlp_0);
            mlp_2 = ggml_add(ctx0, mlp_2, model.mm_model_mlp_2_b);
            // mlp_2 ne = [2048, 576, 1, 1]
            // // AVG Pool Layer 2*2, strides = 2
            mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 0, 2, 3));
            // mlp_2 ne = [576, 2048, 1, 1]
            mlp_2 = ggml_reshape_4d(ctx0, mlp_2, n_patch, n_patch, mlp_2->ne[1], mlp_2->ne[2]);
            // mlp_2 ne [24, 24, 2048, 1]
            mlp_2 = ggml_pool_2d(ctx0, mlp_2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
            // weight ne = [3, 3, 2048, 1]
            struct ggml_tensor * peg_0 = ggml_conv_2d_dw(ctx0, model.mm_model_peg_0_w, mlp_2, 1, 1, 1, 1, 1, 1);
            peg_0 = ggml_cont(ctx0, ggml_permute(ctx0, peg_0, 1, 2, 0, 3));
            peg_0 = ggml_add(ctx0, peg_0, model.mm_model_peg_0_b);
            mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 2, 0, 3));
            peg_0 = ggml_add(ctx0, peg_0, mlp_2);
            peg_0 = ggml_reshape_3d(ctx0, peg_0, peg_0->ne[0], peg_0->ne[1] * peg_0->ne[2], peg_0->ne[3]);
            embeddings = peg_0;
        }
        else {
            GGML_ABORT("fatal error");
        }
    }
    // minicpmv projector
    else if (ctx->has_minicpmv_projector)
    {
        if (ctx->proj_type == PROJECTOR_TYPE_RESAMPLER) {
            struct ggml_tensor * q = model.mm_model_query;
            { // layernorm
                q = ggml_norm(ctx0, q, eps);
                q = ggml_add(ctx0, ggml_mul(ctx0, q, model.mm_model_ln_q_w), model.mm_model_ln_q_b);
            }
            struct ggml_tensor * v = ggml_mul_mat(ctx0, model.mm_model_kv_proj, embeddings);
            { // layernorm
                v = ggml_norm(ctx0, v, eps);
                v = ggml_add(ctx0, ggml_mul(ctx0, v, model.mm_model_ln_kv_w), model.mm_model_ln_kv_b);
            }
            struct ggml_tensor * k;
            { // position
                // q = ggml_add(ctx0, q, model.mm_model_pos_embed);
                k = ggml_add(ctx0, v, pos_embed);
            }

            { // attention
                int hidden_size = 4096;
                const int d_head = 128;
                int n_head = hidden_size/d_head;
                int num_query = 96;
                if (ctx->minicpmv_version == 2) {
                    hidden_size = 4096;
                    n_head = hidden_size/d_head;
                    num_query = 96;
                }
                else if (ctx->minicpmv_version == 3) {
                    hidden_size = 3584;
                    n_head = hidden_size/d_head;
                    num_query = 64;
                }
                else if (ctx->minicpmv_version == 4) {
                    hidden_size = 3584;
                    n_head = hidden_size/d_head;
                    num_query = 64;
                }

                struct ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_q_w, q), model.mm_model_attn_q_b);
                struct ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_k_w, k), model.mm_model_attn_k_b);
                struct ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_v_w, v), model.mm_model_attn_v_b);
                // permute
                Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_query, batch_size);
                Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
                Q = ggml_reshape_3d(ctx0, Q, d_head, num_query, n_head * batch_size);
                K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
                K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
                K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);
                V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
                V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
                V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);
                struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
                KQ = ggml_soft_max_ext(ctx0, KQ, nullptr, 1.0f / sqrtf((float)d_head), 0.0f);
                struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
                KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_query, n_head, batch_size);
                KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
                KQV = ggml_cont_3d(ctx0, KQV, hidden_size, num_query, batch_size);

                embeddings = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_o_w, KQV), model.mm_model_attn_o_b);
            }
            { // layernorm
                embeddings = ggml_norm(ctx0, embeddings, eps);
                embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_model_ln_post_w), model.mm_model_ln_post_b);
            }
            embeddings = ggml_mul_mat(ctx0, model.mm_model_proj, embeddings);
        }
        else {
            GGML_ASSERT(false);
        }
    }
    // glm projector
    else if (ctx->has_glm_projector) {
        if (ctx->proj_type == PROJECTOR_TYPE_GLM_EDGE) {
            size_t gridsz = (size_t)sqrt(embeddings->ne[1]);
            embeddings = ggml_cont(ctx0, ggml_permute(ctx0,embeddings,1,0,2,3));
            embeddings = ggml_reshape_3d(ctx0, embeddings, gridsz, gridsz, embeddings->ne[1]);
            embeddings = ggml_conv_2d(ctx0, model.mm_model_adapter_conv_w, embeddings, 2, 2, 0, 0, 1, 1);
            embeddings = ggml_reshape_3d(ctx0, embeddings,embeddings->ne[0]*embeddings->ne[1] , embeddings->ne[2], batch_size);
            embeddings = ggml_cont(ctx0, ggml_permute(ctx0,embeddings, 1, 0, 2, 3));
            embeddings = ggml_add(ctx0, embeddings, model.mm_model_adapter_conv_b);
            //GLU
            {
                embeddings = ggml_mul_mat(ctx0, model.mm_model_mlp_0_w, embeddings);
                embeddings = ggml_norm(ctx0, embeddings, eps);
                embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_model_ln_q_w), model.mm_model_ln_q_b);
                embeddings = ggml_gelu_inplace(ctx0, embeddings);
                struct ggml_tensor * x = embeddings;
                embeddings = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, embeddings);
                x = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w,x);
                embeddings = ggml_silu_inplace(ctx0, embeddings);
                embeddings = ggml_mul(ctx0, embeddings,x);
                embeddings = ggml_mul_mat(ctx0, model.mm_model_mlp_3_w, embeddings);
            }
        } else {
            GGML_ABORT("fatal error");
        }
    }
    else if (ctx->proj_type == PROJECTOR_TYPE_MERGER) {
        embeddings = ggml_reshape_3d(ctx0, embeddings, hidden_size * 4, num_positions / 4, batch_size);

        embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

        // GELU activation
        embeddings = ggml_gelu(ctx0, embeddings);

        // Second linear layer
        embeddings = ggml_mul_mat(ctx0, model.mm_1_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_1_b);
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}

static ggml_cgraph * clip_image_build_graph(clip_ctx * ctx, const clip_image_f32_batch & imgs, struct clip_image_size load_image_size, bool is_inf = false) {
    if (ctx->proj_type == PROJECTOR_TYPE_GEMMA3) {
        return clip_image_build_graph_siglip(ctx, imgs);
    } else {
        // TODO: we should have one build_* function per model
        return clip_image_build_graph_legacy(ctx, imgs, load_image_size, is_inf);
    }
}

struct clip_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    clip_ctx & ctx_clip;
    std::string fname;

    size_t model_size; // in bytes

    // TODO @ngxson : we should not pass clip_ctx here, it should be clip_vision_model
    clip_model_loader(const char * fname, clip_ctx & ctx_clip) : ctx_clip(ctx_clip), fname(fname) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(string_format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
        }

        ctx_meta.reset(meta);

        const int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

        // print gguf info
        {
            std::string name;
            get_string(KEY_NAME, name, false);
            std::string description;
            get_string(KEY_DESCRIPTION, description, false);
            LOG_INF("%s: model name:   %s\n",  __func__, name.c_str());
            LOG_INF("%s: description:  %s\n",  __func__, description.c_str());
            LOG_INF("%s: GGUF version: %d\n",  __func__, gguf_get_version(ctx_gguf.get()));
            LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf.get()));
            LOG_INF("%s: n_tensors:    %d\n",  __func__, n_tensors);
            LOG_INF("%s: n_kv:         %d\n",  __func__, (int)gguf_get_n_kv(ctx_gguf.get()));
            LOG_INF("\n");
        }

        // tensors
        {
            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t offset = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf.get(), i);
                struct ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                    __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams() {
        // projector type
        {
            std::string proj_type;
            get_string(KEY_PROJ_TYPE, proj_type, false);
            if (!proj_type.empty()) {
                ctx_clip.proj_type = clip_projector_type_from_string(proj_type);
            }
            if (ctx_clip.proj_type == PROJECTOR_TYPE_UNKNOWN) {
                throw std::runtime_error(string_format("%s: unknown projector type: %s\n", __func__, proj_type.c_str()));
            }
        }

        // other hparams
        {
            get_bool(KEY_HAS_TEXT_ENC, ctx_clip.has_text_encoder, false);
            get_bool(KEY_HAS_VIS_ENC, ctx_clip.has_vision_encoder, false);
            GGML_ASSERT(ctx_clip.has_vision_encoder);
            GGML_ASSERT(!ctx_clip.has_text_encoder);

            // legacy keys, use KEY_PROJ_TYPE instead
            get_bool(KEY_HAS_LLAVA_PROJ, ctx_clip.has_llava_projector, false);
            get_bool(KEY_HAS_MINICPMV_PROJ, ctx_clip.has_minicpmv_projector, false);
            get_i32(KEY_MINICPMV_VERSION, ctx_clip.minicpmv_version, false);
            get_bool(KEY_HAS_GLM_PROJ, ctx_clip.has_glm_projector, false);
            get_bool(KEY_HAS_QWEN2VL_MERGER, ctx_clip.has_qwen2vl_merger, false);
            // !!! do NOT extend the list above, use KEY_PROJ_TYPE instead

            get_bool(KEY_USE_GELU, ctx_clip.use_gelu, false);
            get_bool(KEY_USE_SILU, ctx_clip.use_silu, false);

            auto & hparams = ctx_clip.vision_model.hparams;
            get_u32(string_format(KEY_N_EMBD,         "vision"), hparams.hidden_size);
            get_u32(string_format(KEY_N_HEAD,         "vision"), hparams.n_head);
            get_u32(string_format(KEY_N_FF,           "vision"), hparams.n_intermediate);
            get_u32(string_format(KEY_N_BLOCK,        "vision"), hparams.n_layer);
            get_u32(string_format(KEY_PROJ_DIM,       "vision"), hparams.projection_dim);
            get_f32(string_format(KEY_LAYER_NORM_EPS, "vision"), hparams.eps);
            get_u32(KEY_IMAGE_SIZE, hparams.image_size);
            get_u32(KEY_PATCH_SIZE, hparams.patch_size);
            get_u32(KEY_IMAGE_CROP_RESOLUTION, hparams.image_crop_resolution, false);
            get_arr_int(KEY_IMAGE_GRID_PINPOINTS, hparams.image_grid_pinpoints, false);

            {
                std::string mm_patch_merge_type;
                get_string(KEY_MM_PATCH_MERGE_TYPE, mm_patch_merge_type, false);
                if (mm_patch_merge_type == "spatial_unpad") {
                    hparams.mm_patch_merge_type = PATCH_MERGE_SPATIAL_UNPAD;
                }
            }

            {
                int idx_mean = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_MEAN);
                int idx_std  = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_STD);
                GGML_ASSERT(idx_mean >= 0 && "image_mean not found");
                GGML_ASSERT(idx_std >= 0  && "image_std not found");
                const float * mean_data = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_mean);
                const float * std_data  = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_std);
                for (int i = 0; i < 3; ++i) {
                    ctx_clip.image_mean[i] = mean_data[i];
                    ctx_clip.image_std[i]  = std_data[i];
                }
            }

            // Load the vision feature layer indices if they are explicitly provided;
            // if multiple vision feature layers are present, the values will be concatenated
            // to form the final visual features.
            // NOTE: gguf conversions should standardize the values of the vision feature layer to
            // be non-negative, since we use -1 to mark values as unset here.
            std::vector<int> vision_feature_layer;
            get_arr_int(KEY_FEATURE_LAYER, vision_feature_layer, false);
            // convert std::vector to std::unordered_set
            for (auto & layer : vision_feature_layer) {
                hparams.vision_feature_layer.insert(layer);
            }
            // Calculate the deepest feature layer based on hparams and projector type
            ctx_clip.max_feature_layer = get_deepest_feature_layer(&ctx_clip);

            LOG_INF("%s: text_encoder:       %d\n", __func__, ctx_clip.has_text_encoder);
            LOG_INF("%s: vision_encoder:     %d\n", __func__, ctx_clip.has_vision_encoder);
            LOG_INF("%s: llava_projector:    %d\n", __func__, ctx_clip.has_llava_projector);
            LOG_INF("%s: minicpmv_projector: %d\n", __func__, ctx_clip.has_minicpmv_projector);
            LOG_INF("%s: minicpmv_version:   %d\n", __func__, ctx_clip.minicpmv_version);
            LOG_INF("%s: glm_projector:      %d\n", __func__, ctx_clip.has_glm_projector);
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors() {
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        struct ggml_init_params params = {
            /*.mem_size =*/ (gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        ctx_clip.ctx_data.reset(ggml_init(params));
        if (!ctx_clip.ctx_data) {
            throw std::runtime_error(string_format("%s: failed to init ggml context\n", __func__));
        }

        // helper function
        auto get_tensor = [&](const std::string & name, bool required = true) {
            struct ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                // add tensors to context
                struct ggml_tensor * data_tensor = ggml_dup_tensor(ctx_clip.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                cur = data_tensor;
            }
            return cur;
        };

        auto & vision_model = ctx_clip.vision_model;

        vision_model.class_embedding = get_tensor(TN_CLASS_EMBD, false);

        vision_model.pre_ln_w = get_tensor(string_format(TN_LN_PRE, "v", "weight"), false);
        vision_model.pre_ln_b = get_tensor(string_format(TN_LN_PRE, "v", "bias"),   false);

        vision_model.post_ln_w = get_tensor(string_format(TN_LN_POST, "v", "weight"), false);
        vision_model.post_ln_b = get_tensor(string_format(TN_LN_POST, "v", "bias"),   false);

        vision_model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        vision_model.patch_embeddings_0 = get_tensor(TN_PATCH_EMBD,   false);
        vision_model.patch_embeddings_1 = get_tensor(TN_PATCH_EMBD_1, false);
        if (vision_model.patch_embeddings_1 == nullptr) {
            ctx_clip.has_qwen2vl_merger = false;
        }

        vision_model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, "v"), false);

        // layers
        vision_model.layers.resize(vision_model.hparams.n_layer);
        for (int il = 0; il < vision_model.hparams.n_layer; ++il) {
            auto & layer = vision_model.layers[il];
            layer.k_w    = get_tensor(string_format(TN_ATTN_K,      "v", il, "weight"));
            layer.q_w    = get_tensor(string_format(TN_ATTN_Q,      "v", il, "weight"));
            layer.v_w    = get_tensor(string_format(TN_ATTN_V,      "v", il, "weight"));
            layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, "v", il, "weight"));
            layer.ln_1_w = get_tensor(string_format(TN_LN_1,        "v", il, "weight"), false);
            layer.ln_2_w = get_tensor(string_format(TN_LN_2,        "v", il, "weight"), false);
            layer.ff_i_w = get_tensor(string_format(TN_FFN_DOWN,    "v", il, "weight"));
            layer.ff_o_w = get_tensor(string_format(TN_FFN_UP,      "v", il, "weight"));
            layer.k_b    = get_tensor(string_format(TN_ATTN_K,      "v", il, "bias"), false);
            layer.q_b    = get_tensor(string_format(TN_ATTN_Q,      "v", il, "bias"), false);
            layer.v_b    = get_tensor(string_format(TN_ATTN_V,      "v", il, "bias"), false);
            layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, "v", il, "bias"), false);
            layer.ln_1_b = get_tensor(string_format(TN_LN_1,        "v", il, "bias"), false);
            layer.ln_2_b = get_tensor(string_format(TN_LN_2,        "v", il, "bias"), false);
            layer.ff_i_b = get_tensor(string_format(TN_FFN_DOWN,    "v", il, "bias"), false);
            layer.ff_o_b = get_tensor(string_format(TN_FFN_UP,      "v", il, "bias"), false);
        }

        switch (ctx_clip.proj_type) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
                {
                    // LLaVA projection
                    vision_model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"), false);
                    vision_model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    // Yi-type llava
                    vision_model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"), false);
                    vision_model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    // missing in Yi-type llava
                    vision_model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"), false);
                    vision_model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // Yi-type llava
                    vision_model.mm_3_w = get_tensor(string_format(TN_LLAVA_PROJ, 3, "weight"), false);
                    vision_model.mm_3_b = get_tensor(string_format(TN_LLAVA_PROJ, 3, "bias"), false);
                    vision_model.mm_4_w = get_tensor(string_format(TN_LLAVA_PROJ, 4, "weight"), false);
                    vision_model.mm_4_b = get_tensor(string_format(TN_LLAVA_PROJ, 4, "bias"), false);
                    if (vision_model.mm_3_w) {
                        // TODO: this is a hack to support Yi-type llava
                        ctx_clip.proj_type = PROJECTOR_TYPE_MLP_NORM;
                    }
                    vision_model.image_newline = get_tensor(TN_IMAGE_NEWLINE, false);
                } break;
            case PROJECTOR_TYPE_LDP:
                {
                    // MobileVLM projection
                    vision_model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    vision_model.mm_model_mlp_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    vision_model.mm_model_mlp_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    vision_model.mm_model_mlp_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                    vision_model.mm_model_block_1_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "0.weight"));
                    vision_model.mm_model_block_1_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.weight"));
                    vision_model.mm_model_block_1_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.bias"));
                    vision_model.mm_model_block_1_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.weight"));
                    vision_model.mm_model_block_1_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.bias"));
                    vision_model.mm_model_block_1_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.weight"));
                    vision_model.mm_model_block_1_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.bias"));
                    vision_model.mm_model_block_1_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "0.weight"));
                    vision_model.mm_model_block_1_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.weight"));
                    vision_model.mm_model_block_1_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.bias"));
                    vision_model.mm_model_block_2_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "0.weight"));
                    vision_model.mm_model_block_2_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.weight"));
                    vision_model.mm_model_block_2_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.bias"));
                    vision_model.mm_model_block_2_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.weight"));
                    vision_model.mm_model_block_2_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.bias"));
                    vision_model.mm_model_block_2_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.weight"));
                    vision_model.mm_model_block_2_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.bias"));
                    vision_model.mm_model_block_2_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "0.weight"));
                    vision_model.mm_model_block_2_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.weight"));
                    vision_model.mm_model_block_2_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.bias"));
                } break;
            case PROJECTOR_TYPE_LDPV2:
                {
                    // MobilVLM_V2 projection
                    vision_model.mm_model_mlp_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    vision_model.mm_model_mlp_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    vision_model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                    vision_model.mm_model_mlp_2_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "bias"));
                    vision_model.mm_model_peg_0_w = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "weight"));
                    vision_model.mm_model_peg_0_b = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "bias"));
                } break;
            case PROJECTOR_TYPE_RESAMPLER:
                {
                    // vision_model.mm_model_pos_embed = get_tensor(new_clip->ctx_data, TN_MINICPMV_POS_EMBD);
                    vision_model.mm_model_pos_embed_k = get_tensor(TN_MINICPMV_POS_EMBD_K);
                    vision_model.mm_model_query = get_tensor(TN_MINICPMV_QUERY);
                    vision_model.mm_model_proj = get_tensor(TN_MINICPMV_PROJ);
                    vision_model.mm_model_kv_proj = get_tensor(TN_MINICPMV_KV_PROJ);
                    vision_model.mm_model_attn_q_w = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "weight"));
                    vision_model.mm_model_attn_k_w = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "weight"));
                    vision_model.mm_model_attn_v_w = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "weight"));
                    vision_model.mm_model_attn_q_b = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "bias"));
                    vision_model.mm_model_attn_k_b = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "bias"));
                    vision_model.mm_model_attn_v_b = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "bias"));
                    vision_model.mm_model_attn_o_w = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "weight"));
                    vision_model.mm_model_attn_o_b = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "bias"));
                    vision_model.mm_model_ln_q_w = get_tensor(string_format(TN_MINICPMV_LN, "q", "weight"));
                    vision_model.mm_model_ln_q_b = get_tensor(string_format(TN_MINICPMV_LN, "q", "bias"));
                    vision_model.mm_model_ln_kv_w = get_tensor(string_format(TN_MINICPMV_LN, "kv", "weight"));
                    vision_model.mm_model_ln_kv_b = get_tensor(string_format(TN_MINICPMV_LN, "kv", "bias"));
                    vision_model.mm_model_ln_post_w = get_tensor(string_format(TN_MINICPMV_LN, "post", "weight"));
                    vision_model.mm_model_ln_post_b = get_tensor(string_format(TN_MINICPMV_LN, "post", "bias"));
                } break;
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    vision_model.mm_model_adapter_conv_w = get_tensor(string_format(TN_GLM_ADAPER_CONV, "weight"));
                    vision_model.mm_model_adapter_conv_b = get_tensor(string_format(TN_GLM_ADAPER_CONV, "bias"));
                    vision_model.mm_model_mlp_0_w = get_tensor(string_format(TN_GLM_ADAPTER_LINEAR,"weight"));
                    vision_model.mm_model_ln_q_w = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1,"weight"));
                    vision_model.mm_model_ln_q_b = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1,"bias"));
                    vision_model.mm_model_mlp_1_w = get_tensor(string_format(TN_GLM_ADAPTER_D_H_2_4H,"weight"));
                    vision_model.mm_model_mlp_2_w = get_tensor(string_format(TN_GLM_ADAPTER_GATE,"weight"));
                    vision_model.mm_model_mlp_3_w = get_tensor(string_format(TN_GLM_ADAPTER_D_4H_2_H,"weight"));
                    vision_model.boi_w = get_tensor(TN_GLM_BOI_W);
                    vision_model.eoi_w = get_tensor(TN_GLM_EOI_W);
                } break;
            case PROJECTOR_TYPE_MERGER:
                {
                    vision_model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    vision_model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    vision_model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    vision_model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_GEMMA3:
                {
                    vision_model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    vision_model.mm_soft_emb_norm_w = get_tensor(TN_MM_SOFT_EMB_N);
                } break;
            default:
                GGML_ASSERT(false && "unknown projector type");
        }

        // load data
        {
            std::vector<uint8_t> read_buf;

#ifdef _WIN32
            int wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, NULL, 0);
            if (!wlen) {
                throw std::runtime_error(string_format("%s: failed to convert filename to wide string\n", __func__));
            }
            wchar_t * wbuf = (wchar_t *) malloc(wlen * sizeof(wchar_t));
            wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wbuf, wlen);
            if (!wlen) {
                free(wbuf);
                throw std::runtime_error(string_format("%s: failed to convert filename to wide string\n", __func__));
            }
#if __GLIBCXX__
            int fd = _wopen(wbuf, _O_RDONLY | _O_BINARY);
            __gnu_cxx::stdio_filebuf<char> buffer(fd, std::ios_base::in);
            std::istream fin(&buffer);
#else // MSVC
            // unused in our current build
            auto fin = std::ifstream(wbuf, std::ios::binary);
#endif
            free(wbuf);
#else
            auto fin = std::ifstream(fname, std::ios::binary);
#endif
            if (!fin) {
                throw std::runtime_error(string_format("%s: failed to open %s\n", __func__, fname.c_str()));
            }

            // alloc memory and offload data
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx_clip.backend);
            ctx_clip.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_clip.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(ctx_clip.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (auto & t : tensors_to_load) {
                struct ggml_tensor * cur = ggml_get_tensor(ctx_clip.ctx_data.get(), t->name);
                const size_t offset = tensor_offset[t->name];
                fin.seekg(offset, std::ios::beg);
                if (!fin) {
                    throw std::runtime_error(string_format("%s: failed to seek for tensor %s\n", __func__, t->name));
                }
                size_t num_bytes = ggml_nbytes(cur);
                if (ggml_backend_buft_is_host(buft)) {
                    // for the CPU and Metal backend, we can read directly into the tensor
                    fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
                } else {
                    // read into a temporary buffer first, then copy to device memory
                    read_buf.resize(num_bytes);
                    fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
                }
            }
#if defined(_WIN32) && defined(__GLIBCXX__)
            close(fd);
#else
            fin.close();
#endif

            LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(), fname.c_str());
        }
    }

    void alloc_compute_meta() {
        ctx_clip.buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());

        // create a fake batch
        clip_image_f32_batch batch;
        clip_image_f32_ptr img(clip_image_f32_init());
        clip_image_size image_size;
        image_size.width  = clip_get_image_size(&ctx_clip);
        image_size.height = clip_get_image_size(&ctx_clip);
        int n_patches = clip_get_image_size(&ctx_clip) / image_size.width;
        img->nx = n_patches;
        img->ny = n_patches;
        img->buf.resize(n_patches * image_size.width * image_size.height * 3);
        batch.entries.push_back(std::move(img));

        ggml_cgraph * gf = clip_image_build_graph(&ctx_clip, batch, image_size, false);
        ggml_backend_sched_reserve(ctx_clip.sched.get(), gf);
        for (size_t i = 0; i < ctx_clip.backend_ptrs.size(); ++i) {
            ggml_backend_t backend = ctx_clip.backend_ptrs[i];
            ggml_backend_buffer_type_t buft = ctx_clip.backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(ctx_clip.sched.get(), backend);
            if (size > 1) {
                LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }
    }

    void get_bool(const std::string & key, bool & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_i32(const std::string & key, int & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_i32(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_u32(ctx_gguf.get(), i);
    }

    void get_f32(const std::string & key, float & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_string(const std::string & key, std::string & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }

    void get_arr_int(const std::string & key, std::vector<int> & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        int n = gguf_get_arr_n(ctx_gguf.get(), i);
        output.resize(n);
        const int32_t * values = (const int32_t *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (int i = 0; i < n; ++i) {
            output[i] = values[i];
        }
    }
};

// read and create ggml_context containing the tensors and their data
struct clip_ctx * clip_model_load(const char * fname, const int verbosity) {
    return clip_init(fname, clip_context_params{
        /* use_gpu */   true,
        /* verbosity */ static_cast<ggml_log_level>(verbosity),
    });
}

struct clip_ctx * clip_init(const char * fname, struct clip_context_params ctx_params) {
    g_logger_state.verbosity_thold = ctx_params.verbosity;
    clip_ctx * ctx_clip = new clip_ctx(ctx_params);

    try {
        clip_model_loader loader(fname, *ctx_clip);
        loader.load_hparams();
        loader.load_tensors();
        loader.alloc_compute_meta();
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load model '%s': %s\n", __func__, fname, e.what());
        delete ctx_clip;
        return nullptr;
    }

    return ctx_clip;
}

void clip_add_load_image_size(struct clip_ctx * ctx_clip, struct clip_image_size * load_image_size) {
    ctx_clip->load_image_size = *load_image_size; // copy
}

struct clip_image_size * clip_get_load_image_size(struct clip_ctx * ctx_clip) {
    return &ctx_clip->load_image_size;
}

struct clip_image_size * clip_image_size_init() {
    struct clip_image_size * load_image_size = new struct clip_image_size();
    load_image_size->width = 448;
    load_image_size->height = 448;
    return load_image_size;
}

struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

struct clip_image_f32_batch * clip_image_f32_batch_init() {
    return new clip_image_f32_batch();
}

unsigned char * clip_image_u8_get_data(struct clip_image_u8 * img, uint32_t * nx, uint32_t * ny) {
    if (nx) *nx = img->nx;
    if (ny) *ny = img->ny;
    return img->buf.data();
}

void clip_image_size_free(struct clip_image_size * load_image_size) {
    if (load_image_size == nullptr) {
        return;
    }
    delete load_image_size;
}
void clip_image_u8_free(struct clip_image_u8  * img) { if (img) delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { if (img) delete img; }
void clip_image_u8_batch_free(struct clip_image_u8_batch * batch) { if (batch) delete batch; }
void clip_image_f32_batch_free(struct clip_image_f32_batch * batch) { if (batch) delete batch; }

size_t clip_image_f32_batch_n_images(const struct clip_image_f32_batch * batch) {
    return batch->entries.size();
}

size_t clip_image_f32_batch_nx(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->nx;
}

size_t clip_image_f32_batch_ny(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->ny;
}

clip_image_f32 * clip_image_f32_get_img(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return nullptr;
    }
    return batch->entries[idx].get();
}

void clip_build_img_from_pixels(const unsigned char * rgb_pixels, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), rgb_pixels, img->buf.size());
}

bool clip_image_load_from_file(const char * fname, clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load(fname, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to load image '%s'\n", __func__, fname);
        return false;
    }
    clip_build_img_from_pixels(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOG_ERR("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    clip_build_img_from_pixels(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

// Linear interpolation between two points
inline float clip_lerp(float s, float e, float t) {
    return s + (e - s) * t;
}
// Bilinear resize function
static void bilinear_resize(const clip_image_u8& src, clip_image_u8& dst, int target_width, int target_height) {
    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float x_ratio = static_cast<float>(src.nx - 1) / target_width;
    float y_ratio = static_cast<float>(src.ny - 1) / target_height;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            float px = x_ratio * x;
            float py = y_ratio * y;
            int x_floor = static_cast<int>(px);
            int y_floor = static_cast<int>(py);
            float x_lerp = px - x_floor;
            float y_lerp = py - y_floor;

            for (int c = 0; c < 3; c++) {
                float top = clip_lerp(
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                float bottom = clip_lerp(
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(clip_lerp(top, bottom, y_lerp));
            }
        }
    }
}

// Normalize image to float32 - careful with pytorch .to(model.device, dtype=torch.float16) - this sometimes reduces precision (32>16>32), sometimes not
static void normalize_image_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst, const float mean[3], const float std[3]) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    // TODO @ngxson : seems like this could be done more efficiently on cgraph
    for (size_t i = 0; i < src.buf.size(); ++i) {
        int c = i % 3; // rgb
        dst.buf[i] = (static_cast<float>(src.buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

inline int clip(int x, int lower, int upper) {
    return std::max(lower, std::min(x, upper));
}

static bool bicubic_resize(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++) {
        for (j = 0; j < target_width; j++) {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                }
            }
        }
    }

    return true;
}

// llava-1.6 type of resize_and_pad (black)
static void resize_and_pad_image(const clip_image_u8& image, clip_image_u8 &image_output, const std::pair<int, int>& target_resolution) {
    int target_width = target_resolution.first;
    int target_height = target_resolution.second;

    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;

    int new_width, new_height;

    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
    } else {
        new_height = target_height;
        new_width = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
    }

    clip_image_u8 resized_image;
    // bilinear_resize(image, resized_image, new_width, new_height);
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height, 0); // Initialize with black

    // Calculate padding offsets
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Copy the resized image into the center of the padded buffer
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    image_output = std::move(padded_image);
}

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static std::pair<int, int> select_best_resolution(const std::pair<int, int> & original_size, const std::vector<std::pair<int, int>> & possible_resolutions) {
    int original_width = original_size.first;
    int original_height = original_size.second;
    std::pair<int, int> best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width = resolution.first;
        int height = resolution.second;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // LOG_INF("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

static std::vector<clip_image_u8_ptr> divide_to_patches_u8(const clip_image_u8 & image, int patch_size) {
    std::vector<clip_image_u8_ptr> patches;
    int width = image.nx;
    int height = image.ny;
    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            clip_image_u8_ptr patch(clip_image_u8_init());
            patch->nx = std::min(patch_size, width - j);
            patch->ny = std::min(patch_size, height - i);
            patch->buf.resize(3 * patch->nx * patch->ny);
            for (int y = 0; y < patch->ny; ++y) {
                for (int x = 0; x < patch->nx; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        patch->buf[3 * (y * patch->nx + x) + c] = image.buf[3 * ((i + y) * width + (j + x)) + c];
                    }
                }
            }
            patches.push_back(std::move(patch));
        }
    }
    return patches;
}

static int ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

static std::pair<int, int> uhd_find_best_resize(std::pair<int, int> original_size, int scale_resolution, int patch_size, bool allow_upscale = false) {
    int width = original_size.first;
    int height = original_size.second;
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
        float r = static_cast<float>(width) / height;
        height = static_cast<int>(scale_resolution / std::sqrt(r));
        width = static_cast<int>(height * r);
    }
    int best_width = ensure_divide(width, patch_size);
    int best_height = ensure_divide(height, patch_size);
    return std::make_pair(best_width, best_height);
}

static std::pair<int, int> uhd_get_refine_size(std::pair<int, int> original_size, std::pair<int, int> grid, int scale_resolution, int patch_size, bool allow_upscale = false) {
    int width, height;
    std::tie(width, height) = original_size;
    int grid_x, grid_y;
    std::tie(grid_x, grid_y) = grid;

    int refine_width = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    int grid_width = refine_width / grid_x;
    int grid_height = refine_height / grid_y;

   // auto best_grid_size = find_best_resize(std::make_tuple(grid_width, grid_height), scale_resolution, patch_size, allow_upscale); (old line)
    auto best_grid_size = uhd_find_best_resize(std::make_pair(grid_width, grid_height), scale_resolution, patch_size, allow_upscale); // (new line) => fixes conversion for make_tuple to make_pair
    int best_grid_width, best_grid_height;
    std::tie(best_grid_width, best_grid_height) = best_grid_size;

  //  std::pair<int, int> refine_size = std::make_tuple(best_grid_width * grid_x, best_grid_height * grid_y); (old line)
    std::pair<int, int> refine_size = std::make_pair(best_grid_width * grid_x, best_grid_height * grid_y); // (new line)
    return refine_size;
}

static std::pair<int, int> uhd_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
    std::vector<int> candidate_split_grids_nums;
    for (int i : {multiple - 1, multiple, multiple + 1}) {
        if (i == 1 || i > max_slice_nums) {
            continue;
        }
        candidate_split_grids_nums.push_back(i);
    }

    std::vector<std::pair<int, int>> candidate_grids;
    for (int split_grids_nums : candidate_split_grids_nums) {
        int m = 1;
        while (m <= split_grids_nums) {
            if (split_grids_nums % m == 0) {
                candidate_grids.emplace_back(m, split_grids_nums / m);
            }
            ++m;
        }
    }

    std::pair<int, int> best_grid{1, 1};
    float min_error = std::numeric_limits<float>::infinity();
    for (const auto& grid : candidate_grids) {
        float error = std::abs(log_ratio - std::log(1.0 * grid.first / grid.second));
        if (error < min_error) {
            best_grid = grid;
            min_error = error;
        }
    }
    return best_grid;
}

// inspired from LLaVA-UHD:
//    -> https://arxiv.org/pdf/2403.11703
//    -> https://github.com/thunlp/LLaVA-UHD
//    -> https://github.com/thunlp/LLaVA-UHD/blob/302301bc2175f7e717fb8548516188e89f649753/llava_uhd/train/llava-uhd/slice_logic.py#L118
static std::vector<std::vector<clip_image_u8_ptr>> uhd_slice_image(const clip_image_u8 * img, const int max_slice_nums=9, const int scale_resolution=448, const int patch_size=14) {
    const std::pair<int, int> original_size={img->nx,img->ny};
    const int original_width = img->nx;
    const int original_height = img->ny;
    const float log_ratio = log(1.0*original_width/original_height);
    const float ratio = 1.0 * original_width * original_height/ (scale_resolution * scale_resolution);
    const int multiple = fmin(ceil(ratio), max_slice_nums);

    std::vector<std::vector<clip_image_u8_ptr>> images;
    LOG_DBG("%s: multiple %d\n", __func__, multiple);
    images.push_back(std::vector<clip_image_u8_ptr>());

    if (multiple <= 1) {
        auto best_size = uhd_find_best_resize(original_size, scale_resolution, patch_size, true);
        clip_image_u8_ptr source_image(clip_image_u8_init());
        bicubic_resize(*img, *source_image, best_size.first, best_size.second);
        // source_image = image.resize(best_size, Image.Resampling.BICUBIC)
        images.back().push_back(std::move(source_image));
    }
    else if (multiple > 1) {
        auto best_size = uhd_find_best_resize(original_size, scale_resolution, patch_size);
        clip_image_u8_ptr source_image(clip_image_u8_init());
        bicubic_resize(*img, *source_image, best_size.first, best_size.second);
        // source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        LOG_DBG("%s: image_size: %d %d; source_image size: %d %d\n", __func__, img->nx, img->ny, best_size.first, best_size.second);
        images.back().push_back(std::move(source_image));

        std::pair<int, int> best_grid = uhd_best_grid(max_slice_nums, multiple, log_ratio);
        LOG_DBG("%s: image_size: %d %d; best_grid: %d %d\n", __func__, img->nx, img->ny, best_grid.first, best_grid.second);

        auto refine_size = uhd_get_refine_size(original_size, best_grid, scale_resolution, patch_size, true);
        clip_image_u8_ptr refine_image(clip_image_u8_init());
        bicubic_resize(*img, *refine_image, refine_size.first, refine_size.second);

        LOG_DBG("%s: refine_image_size: %d %d; refine_size: %d %d\n", __func__, refine_image->nx, refine_image->ny, refine_size.first, refine_size.second);

        // split_to_patches
        int width = refine_image->nx;
        int height = refine_image->ny;
        int grid_x = int(width / best_grid.first);
        int grid_y = int(height / best_grid.second);
        for (int patches_i = 0, ic = 0; patches_i < height && ic < best_grid.second; patches_i += grid_y, ic += 1){
            images.push_back(std::vector<clip_image_u8_ptr>());
            for(int patches_j = 0, jc = 0; patches_j < width && jc < best_grid.first; patches_j += grid_x, jc += 1){
                clip_image_u8_ptr patch(clip_image_u8_init());
                patch->nx = grid_x;
                patch->ny = grid_y;
                patch->buf.resize(3 * patch->nx * patch->ny);
                for (int y = patches_i; y < patches_i + grid_y; ++y) {
                    for (int x = patches_j; x < patches_j + grid_x; ++x) {
                        const int i = 3 * (y * refine_image->nx + x);
                        const int j = 3 * ((y-patches_i) * patch->nx + (x-patches_j));
                        patch->buf[j]   = refine_image->buf[i];
                        patch->buf[j+1] = refine_image->buf[i+1];
                        patch->buf[j+2] = refine_image->buf[i+2];
                    }
                }
                images.back().push_back(std::move(patch));
            }
        }
    }
    return images;
}

int clip_uhd_num_image_embeds_col(struct clip_ctx * ctx_clip) {
    const int max_slice_nums=9;
    const int scale_resolution=448;
    const int original_width = ctx_clip->load_image_size.width;
    const int original_height = ctx_clip->load_image_size.height;
    const float log_ratio = log(1.0*original_width/original_height);
    const float ratio = 1.0 * original_width * original_height/ (scale_resolution * scale_resolution);
    const int multiple = fmin(ceil(ratio), max_slice_nums);
    std::pair<int, int> best_grid = uhd_best_grid(max_slice_nums, multiple, log_ratio);
    return best_grid.first;
}

// returns the normalized float tensor for llava-1.5, for spatial_unpad with anyres processing for llava-1.6 it returns the normalized image patch tensors as a vector
// res_imgs memory is being allocated here, previous allocations will be freed if found
bool clip_image_preprocess(struct clip_ctx * ctx, const clip_image_u8 * img, struct clip_image_f32_batch * res_imgs) {

    if (clip_is_minicpmv(ctx)) {
        int max_slice_nums = 9;
        std::vector<std::vector<clip_image_u8_ptr>> imgs = uhd_slice_image(img, max_slice_nums);
        for (size_t i = 0; i < imgs.size(); ++i) {
            for (size_t j = 0; j < imgs[i].size(); ++j) {
                LOG_DBG("%s: %d %d\n", __func__,imgs[i][j]->nx,imgs[i][j]->ny);
                clip_image_f32_ptr res(clip_image_f32_init());
                normalize_image_u8_to_f32(*imgs[i][j], *res, ctx->image_mean, ctx->image_std);
                res_imgs->entries.push_back(std::move(res));
            }
        }
        return true;
    }
    else if (ctx->has_qwen2vl_merger) {
        clip_image_u8 resized;
        auto patch_size = clip_get_patch_size(ctx) * 2;
        int nx = ceil((float)img->nx / patch_size) * patch_size;
        int ny = ceil((float)img->ny / patch_size) * patch_size;
        bicubic_resize(*img, resized, nx, ny);

        clip_image_f32_ptr img_f32(clip_image_f32_init());
        // clip_image_f32_ptr res(clip_image_f32_init());
        normalize_image_u8_to_f32(resized, *img_f32, ctx->image_mean, ctx->image_std);
        // res_imgs->data[0] = *res;
        res_imgs->entries.push_back(std::move(img_f32));
        return true;
    }

    if (ctx->has_glm_projector || ctx->proj_type == PROJECTOR_TYPE_GEMMA3) {
        clip_image_u8 resized_image;
        int32_t sz=ctx->vision_model.hparams.image_size;
        bicubic_resize(*img, resized_image,sz,sz);
        clip_image_f32_ptr img_f32(clip_image_f32_init());
        //clip_image_save_to_bmp(resized_image, "resized.bmp");
        normalize_image_u8_to_f32(resized_image, *img_f32, ctx->image_mean, ctx->image_std);
        res_imgs->entries.push_back(std::move(img_f32));
        return true;
    }

    bool pad_to_square = true;
    if (!ctx->has_vision_encoder) {
        LOG_ERR("%s: This gguf file seems to have no vision encoder\n", __func__);
        return false;
    }
    auto & params = ctx->vision_model.hparams;
    // The model config actually contains all we need to decide on how to preprocess, here we automatically switch to the new llava-1.6 preprocessing
    if (params.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD) {
        pad_to_square = false;
    }
    // free the previous res_imgs if any set
    res_imgs->entries.clear();

    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8_ptr temp(clip_image_u8_init()); // we will keep the input image data here temporarily
    if (pad_to_square && img->nx != img->ny) {
        int longer_side = std::max(img->nx, img->ny);
        temp->nx = longer_side;
        temp->ny = longer_side;
        temp->buf.resize(3 * longer_side * longer_side);
        const uint8_t bc[3] = {122, 116, 104}; // background color in RGB from LLaVA (this is the mean rgb color * 255)

        // fill with background color
        for (size_t i = 0; i < temp->buf.size(); i++) {
            temp->buf[i] = bc[i % 3];
        }

        // copy from the input image
        for (int y = 0; y < img->ny; y++) {
            for (int x = 0; x < img->nx; x++) {
                const int i = 3 * (y * img->nx + x);
                const int j = 3 * (y * temp->nx + x);
                temp->buf[j]   = img->buf[i];
                temp->buf[j+1] = img->buf[i+1];
                temp->buf[j+2] = img->buf[i+2];
            }
        }
    } else {
        if (!params.image_grid_pinpoints.empty()) {
            // "spatial_unpad" with "anyres" processing for llava-1.6
            std::vector<std::pair<int, int>> possible_resolutions;
            for (size_t i = 0; i < params.image_grid_pinpoints.size(); i+=2) {
                possible_resolutions.push_back({params.image_grid_pinpoints[i], params.image_grid_pinpoints[i+1]});
            }
            std::pair<int, int> best_resolution = select_best_resolution({img->nx, img->ny}, possible_resolutions);
            // clip_image_save_to_bmp(*img, "input.bmp");
            resize_and_pad_image(*img, *temp, best_resolution);  // we do not pad with mean-bg color anymore in llava-1.6
            // clip_image_save_to_bmp(*temp, "resized.bmp");
            // visually verify normalized image:
            // normalize_image_u8_to_f32(*temp, *res, ctx->image_mean, ctx->image_std);
            // {
            //     clip_image_u8 * temp2 = clip_image_u8_init();
            //     clip_image_convert_f32_to_u8(*res, *temp2);
            //     clip_image_save_to_bmp(*temp2, "resized_normalized_f32.bmp");
            //     clip_image_u8_free(temp2);
            // }

            std::vector<clip_image_u8_ptr> patches = divide_to_patches_u8(*temp, params.image_size); // prepare spatial sorted main patches of image_size each (336 in llava-1.6)

            clip_image_u8_ptr image_original_resize(clip_image_u8_init());
            // bilinear_resize(*img, *image_original_resize, params.image_size, params.image_size); // in python this is "shortest_edge", but all CLIP are square
            bicubic_resize(*img, *image_original_resize, params.image_size, params.image_size); // in python this is "shortest_edge", but all CLIP are square
            patches.insert(patches.begin(), std::move(image_original_resize));
            for (auto & patch : patches) {
                clip_image_f32_ptr res(clip_image_f32_init());
                normalize_image_u8_to_f32(*patch, *res, ctx->image_mean, ctx->image_std);
                res_imgs->entries.push_back(std::move(res));
            }

            return true;
        } else {
            temp->nx = img->nx;
            temp->ny = img->ny;
            temp->buf.resize(img->buf.size());
            memcpy(temp->buf.data(), img->buf.data(), temp->buf.size());
        }
    }

    const int nx = temp->nx;
    const int ny = temp->ny;
    // clip_image_save_to_bmp(*temp, "resized_vanilla.bmp");

    const int nx2 = ctx->vision_model.hparams.image_size;
    const int ny2 = ctx->vision_model.hparams.image_size;
    clip_image_f32_ptr res(clip_image_f32_init());
    res->nx = nx2;
    res->ny = ny2;
    res->buf.resize(3 * nx2 * ny2);

    const float scale = std::max(nx, ny) / (float)ctx->vision_model.hparams.image_size;

    const int nx3 = int(nx / scale + 0.5f);
    const int ny3 = int(ny / scale + 0.5f);

    const auto & m3 = ctx->image_mean; // {0.48145466f, 0.4578275f, 0.40821073f};
    const auto & s3 = ctx->image_std;  // {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = (x + 0.5f) * scale - 0.5f;
                const float sy = (y + 0.5f) * scale - 0.5f;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = temp->buf[j00];
                const float v01 = temp->buf[j01];
                const float v10 = temp->buf[j10];
                const float v11 = temp->buf[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res->buf[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }

    // {
    //     clip_image_u8 * temp2 = clip_image_u8_init();
    //     clip_image_convert_f32_to_u8(*res, *temp2);
    //     clip_image_save_to_bmp(*temp2, "resized_normalized_f32_vanilla.bmp");
    //     clip_image_u8_free(temp2);
    // }
    // res_imgs.push_back(res);

    res_imgs->entries.push_back(std::move(res));

    return true;
}

ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx) {
    return ctx->vision_model.image_newline;
}

void clip_free(clip_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    int extra_tokens = ctx->has_glm_projector ? 2 : 0;
    return (clip_n_patches(ctx) + extra_tokens) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_h, int img_w) {
    clip_image_f32 img;
    img.nx = img_w;
    img.ny = img_h;
    return clip_n_patches_by_img(ctx, &img) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

int32_t clip_get_image_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.image_size;
}

int32_t clip_get_patch_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.patch_size;
}

int32_t clip_get_hidden_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.hidden_size;
}

const char * clip_patch_merge_type(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD ? "spatial_unpad" : "flat";
}

const int32_t * clip_image_grid(const struct clip_ctx * ctx) {
    if (ctx->vision_model.hparams.image_grid_pinpoints.size()) {
        return &ctx->vision_model.hparams.image_grid_pinpoints.front();
    }
    return nullptr;
}

size_t get_clip_image_grid_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.image_grid_pinpoints.size();
}

int clip_n_patches(const struct clip_ctx * ctx) {
    clip_image_f32 img;
    img.nx = ctx->vision_model.hparams.image_size;
    img.ny = ctx->vision_model.hparams.image_size;
    return clip_n_patches_by_img(ctx, &img);
}

int clip_n_patches_by_img(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->vision_model.hparams;

    int n_patches = (params.image_size / params.patch_size) * (params.image_size / params.patch_size);

    if (ctx->proj_type == PROJECTOR_TYPE_LDP || ctx->proj_type == PROJECTOR_TYPE_LDPV2 || ctx->proj_type == PROJECTOR_TYPE_GLM_EDGE) {
        n_patches /= 4;
    } else if (ctx->proj_type == PROJECTOR_TYPE_RESAMPLER) {
        if (ctx->minicpmv_version == 2) {
            n_patches = 96;
        }
        else if (ctx->minicpmv_version == 3) {
            n_patches = 64;
        }
        else if (ctx->minicpmv_version == 4) {
            n_patches = 64;
        }
    } else if (ctx->proj_type == PROJECTOR_TYPE_MERGER) {
        int patch_size = params.patch_size * 2;
        int x_patch = img->nx / patch_size + (int)(img->nx % patch_size > 0);
        int y_patch = img->ny / patch_size + (int)(img->ny % patch_size > 0);
        n_patches = x_patch * y_patch;
    } else if (ctx->proj_type == PROJECTOR_TYPE_GEMMA3) {
        n_patches = 256;
    }

    return n_patches;
}

static std::vector<std::vector<std::vector<float>>> get_1d_sincos_pos_embed_from_grid_new(int embed_dim, const std::vector<std::vector<float>> & pos) {
    assert(embed_dim % 2 == 0);
    int H = pos.size();
    int W = pos[0].size();

    std::vector<float> omega(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; ++i) {
        omega[i] = 1.0 / pow(10000.0, static_cast<float>(i) / (embed_dim / 2));
    }

    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                float out_value = pos[h][w] * omega[d];
                emb[h][w][d] = sin(out_value);
                emb[h][w][d + embed_dim / 2] = cos(out_value);
            }
        }
    }

    return emb;
}

static std::vector<std::vector<std::vector<float>>> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<std::vector<float>>> & grid) {
    assert(embed_dim % 2 == 0);
    std::vector<std::vector<std::vector<float>>> emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[0]); // (H, W, D/2)
    std::vector<std::vector<std::vector<float>>> emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[1]); // (H, W, D/2)

    int H = emb_h.size();
    int W = emb_h[0].size();
    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                emb[h][w][d] = emb_h[h][w][d];
                emb[h][w][d + embed_dim / 2] = emb_w[h][w][d];
            }
        }
    }
    return emb;
}

static std::vector<std::vector<float>> get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int> image_size) {
    int grid_h_size = image_size.first;
    int grid_w_size = image_size.second;

    std::vector<float> grid_h(grid_h_size);
    std::vector<float> grid_w(grid_w_size);

    for (int i = 0; i < grid_h_size; ++i) {
        grid_h[i] = static_cast<float>(i);
    }
    for (int i = 0; i < grid_w_size; ++i) {
        grid_w[i] = static_cast<float>(i);
    }

    std::vector<std::vector<float>> grid(grid_h_size, std::vector<float>(grid_w_size));
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid[h][w] = grid_w[w];
        }
    }
    std::vector<std::vector<std::vector<float>>> grid_2d = {grid, grid};
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid_2d[0][h][w] = grid_h[h];
            grid_2d[1][h][w] = grid_w[w];
        }
    }

    std::vector<std::vector<std::vector<float>>> pos_embed_3d = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d);

    int H = image_size.first;
    int W = image_size.second;
    std::vector<std::vector<float>> pos_embed_2d(H * W, std::vector<float>(embed_dim));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            pos_embed_2d[w * H + h] = pos_embed_3d[h][w];
        }
    }

    return pos_embed_2d;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    if (!ctx->has_vision_encoder) {
        LOG_ERR("%s: This gguf file seems to have no vision encoder\n", __func__);
        return false;
    }

    clip_image_f32_batch imgs;
    clip_image_f32_ptr img_copy(clip_image_f32_init());
    *img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs_c_ptr, float * vec) {
    const clip_image_f32_batch & imgs = *imgs_c_ptr;

    if (!ctx->has_vision_encoder) {
        LOG_ERR("%s: This gguf file seems to have no vision encoder\n", __func__);
        return false;
    }

    int batch_size = imgs.entries.size();
    if (ctx->has_llava_projector) {
        GGML_ASSERT(batch_size == 1); // TODO: support multiple images
    }
    if (ctx->has_minicpmv_projector) {
        GGML_ASSERT(batch_size == 1);
    }
    if (ctx->has_glm_projector) {
        GGML_ASSERT(batch_size == 1);
        ggml_tensor * boi = ctx->vision_model.boi_w;
        ggml_backend_tensor_get(boi,vec,0,ggml_nbytes(boi));
        vec = (float*)(vec+ggml_nelements(boi)); //offset for boi
    }

    // build the inference graph
    ggml_backend_sched_reset(ctx->sched.get());
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs, ctx->load_image_size, true);
    ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);

    // set inputs
    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size = hparams.image_size;
    int image_size_width  = image_size;
    int image_size_height = image_size;
    if (ctx->has_minicpmv_projector | ctx->has_qwen2vl_merger) {
        image_size_width  = imgs.entries[0]->nx;
        image_size_height = imgs.entries[0]->ny;
    }
    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int num_positions = num_patches + (model.class_embedding ? 1 : 0);
    const int pos_w = ctx->load_image_size.width / patch_size;
    const int pos_h = ctx->load_image_size.height / patch_size;

    {
        struct ggml_tensor * inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        float * data = (float *)malloc(ggml_nbytes(inp_raw));

        for (size_t i = 0; i < imgs.entries.size(); i++) {
            const int nx = imgs.entries[i]->nx;
            const int ny = imgs.entries[i]->ny;
            if (!(ctx->has_minicpmv_projector | ctx->has_qwen2vl_merger)) {
                GGML_ASSERT(nx == image_size && ny == image_size);
            }

            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < 3; k++) {
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            data[(b * 3 * n) + k * n + y * nx + x] = imgs.entries[b]->buf[3 * (y * nx + x) + k];
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(inp_raw, data, 0, ggml_nbytes(inp_raw));
        free(data);
    }
    if (ctx->has_minicpmv_projector) {
        {
            // inspired from siglip:
            //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
            //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
            struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "positions");
            int* positions_data = (int*)malloc(ggml_nbytes(positions));
            int bucket_coords_h[1024];
            int bucket_coords_w[1024];
            for (int i = 0; i < pos_h; i++){
                bucket_coords_h[i] = std::floor(70.0*i/pos_h);
            }
            for (int i = 0; i < pos_w; i++){
                bucket_coords_w[i] = std::floor(70.0*i/pos_w);
            }
            for (int i = 0, id = 0; i < pos_h; i++){
                for (int j = 0; j < pos_w; j++){
                    positions_data[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
                }
            }
            ggml_backend_tensor_set(positions, positions_data, 0, ggml_nbytes(positions));
            free(positions_data);
        }

        {
            // inspired from resampler of Qwen-VL:
            //    -> https://huggingface.co/Qwen/Qwen-VL/tree/main
            //    -> https://huggingface.co/Qwen/Qwen-VL/blob/0547ed36a86561e2e42fecec8fd0c4f6953e33c4/visual.py#L23
            struct ggml_tensor * pos_embed = ggml_graph_get_tensor(gf, "pos_embed");
            int embed_dim = 4096;
            if (ctx->minicpmv_version == 2) {
                embed_dim = 4096;
            }
            else if (ctx->minicpmv_version == 3) {
                embed_dim = 3584;
            }
            else if (ctx->minicpmv_version == 4) {
                embed_dim = 3584;
            }
            auto pos_embed_t = get_2d_sincos_pos_embed(embed_dim, std::make_pair(pos_w, pos_h));

            float * pos_embed_data = (float *)malloc(ggml_nbytes(pos_embed));
            for(int i=0;i < pos_w * pos_h; ++i){
                for(int j=0; j < embed_dim; ++j){
                    pos_embed_data[i * embed_dim + j] = pos_embed_t[i][j];
                }
            }

            ggml_backend_tensor_set(pos_embed, pos_embed_data, 0, ggml_nbytes(pos_embed));
            free(pos_embed_data);
        }
    }
    else {
        if (model.class_embedding) {
            struct ggml_tensor * embeddings = ggml_graph_get_tensor(gf, "embeddings");

            void* zero_mem = malloc(ggml_nbytes(embeddings));
            memset(zero_mem, 0, ggml_nbytes(embeddings));
            ggml_backend_tensor_set(embeddings, zero_mem, 0, ggml_nbytes(embeddings));
            free(zero_mem);
        }

        if (ctx->has_qwen2vl_merger) {
            struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "positions");

            const int pw = image_size_width / patch_size;
            const int ph = image_size_height / patch_size;
            int* positions_data = (int*)malloc(ggml_nbytes(positions));

            int ptr = 0;
            for (int y = 0; y < ph; y+=2)
            {
                for (int x = 0; x < pw; x+=2)
                {
                    for (int dy = 0; dy < 2; dy++) {
                        for (int dx = 0; dx < 2; dx++) {
                            positions_data[ptr]                 = y + dy;
                            positions_data[num_patches + ptr]     = x + dx;
                            positions_data[num_patches * 2 + ptr] = y + dy;
                            positions_data[num_patches * 3 + ptr] = x + dx;
                            ptr++;
                        }
                    }
                }
            }

            ggml_backend_tensor_set(positions, positions_data, 0, ggml_nbytes(positions));
            free(positions_data);
        }
        else if (ctx->proj_type == PROJECTOR_TYPE_GEMMA3) {
            // do nothing
        }
        else {
            struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "positions");

            int* positions_data = (int*)malloc(ggml_nbytes(positions));
            for (int i = 0; i < num_positions; i++) {
                positions_data[i] = i;
            }
            ggml_backend_tensor_set(positions, positions_data, 0, ggml_nbytes(positions));
            free(positions_data);

            if (!ctx->has_glm_projector) {
                struct ggml_tensor * patches = ggml_graph_get_tensor(gf, "patches");
                // The patches vector is used to get rows to index into the embeds with;
                // we should skip dim 0 only if we have CLS to avoid going out of bounds
                // when retrieving the rows.
                int patch_offset = model.class_embedding ? 1 : 0;
                int* patches_data = (int*)malloc(ggml_nbytes(patches));
                for (int i = 0; i < num_patches; i++) {
                    patches_data[i] = i + patch_offset;
                }
                ggml_backend_tensor_set(patches, patches_data, 0, ggml_nbytes(patches));
                free(patches_data);
            }
        }
    }

    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);

    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    // the last node is the embedding tensor
    struct ggml_tensor * embeddings = ggml_graph_node(gf, -1);

    // copy the embeddings to the location passed by the user
    ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));

    if (ctx->has_glm_projector) {
        //eoi
        ggml_tensor * eoi = ctx->vision_model.eoi_w;
        int offset = ggml_nelements(embeddings);
        ggml_backend_tensor_get(eoi, vec+offset, 0, ggml_nbytes(eoi));
    }

    return true;
}

bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype) {
    assert(itype < GGML_TYPE_COUNT);
    ggml_type type = static_cast<ggml_type>(itype);

    auto * ctx_clip = clip_init(fname_inp, clip_context_params{
        /* use_gpu */   false,
        /* verbosity */ GGML_LOG_LEVEL_ERROR,
    });

    const auto & ctx_src = ctx_clip->ctx_gguf.get();
    const auto & ctx_data = ctx_clip->ctx_data.get();

    auto * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_src);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", itype);

    auto fout = std::ofstream(fname_out, std::ios::binary);

    const int n_tensors = gguf_get_n_tensors(ctx_src);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
        gguf_add_tensor(ctx_out, cur);
    }

    const size_t meta_size = gguf_get_meta_size(ctx_out);
    for (size_t i = 0; i < meta_size; ++i) {
        fout.put(0);
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> k_names = {
        ".*weight",
    };

    std::vector<uint8_t> work(512);
    std::vector<float> conv_buf(512);
    size_t total_size_org = 0;
    size_t total_size_new = 0;

    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name.c_str());

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        bool quantize = false;
        for (const auto & s : k_names) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // quantize only 2D tensors and bigger than block size
        quantize &= (ggml_n_dims(cur) == 2) && cur->ne[0] > ggml_blck_size(type);

        if (quantize) {
            new_type = type;
            if (new_type >= GGML_TYPE_Q2_K && name.find("embd") != std::string::npos) {
                new_type = GGML_TYPE_Q8_0; // ggml_get_rows needs non K type
                // LOG_ERR("%s: quantizing %s to %s\n", __func__, name.c_str(), ggml_type_name(new_type));
            }
            const size_t n_elms = ggml_nelements(cur);
            float * f32_data;

            switch (cur->type) {
            case GGML_TYPE_F32:
                f32_data = (float *)cur->data;
                break;
            case GGML_TYPE_F16:
                if (conv_buf.size() < n_elms) {
                    conv_buf.resize(n_elms);
                }
                for (size_t j = 0; j < n_elms; ++j) {
                    conv_buf[j] = ggml_fp16_to_fp32(((ggml_fp16_t *)cur->data)[j]);
                }
                f32_data = (float *)conv_buf.data();
                break;
            default:
                LOG_ERR("%s: Please use an input file in f32 or f16\n", __func__);
                gguf_free(ctx_out);
                return false;
            }

            if (work.size() < n_elms * 4) {
                work.resize(n_elms * 4);
            }
            new_data = work.data();

            new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, n_elms/cur->ne[0], cur->ne[0], nullptr);
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }
        const size_t orig_size = ggml_nbytes(cur);
        total_size_org += orig_size;
        total_size_new += new_size;
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        GGML_ASSERT(gguf_get_tensor_size(ctx_out, gguf_find_tensor(ctx_out, name.c_str())) == new_size);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data);
        fout.write((const char *)new_data, new_size);
        size_t pad = GGML_PAD(new_size, gguf_get_alignment(ctx_out)) - new_size;
        for (size_t j = 0; j < pad; ++j) {
            fout.put(0);
        }

        LOG_INF("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), ggml_n_dims(cur), quantize,
               orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
    }

    // go back to beginning of file and write the updated metadata
    fout.seekp(0, std::ios::beg);
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(ctx_out, meta.data());
    fout.write((const char *)meta.data(), meta_size);

    fout.close();

    clip_free(ctx_clip);
    gguf_free(ctx_out);

    {
        LOG_INF("%s: original  size = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        LOG_INF("%s: quantized size = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);
    }

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    if (ctx->proj_type == PROJECTOR_TYPE_LDP) {
        return ctx->vision_model.mm_model_block_1_block_2_1_b->ne[0];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_LDPV2) {
        return ctx->vision_model.mm_model_peg_0_b->ne[0];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_MLP) {
        return ctx->vision_model.mm_2_b->ne[0];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_MLP_NORM) {
        return ctx->vision_model.mm_3_b->ne[0];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_RESAMPLER) {
        if (ctx->minicpmv_version == 2) {
            return 4096;
        }
        else if (ctx->minicpmv_version == 3) {
            return 3584;
        }
        else if (ctx->minicpmv_version == 4) {
            return 3584;
        }
    }
    if (ctx->proj_type == PROJECTOR_TYPE_GLM_EDGE){
        return ctx->vision_model.mm_model_mlp_3_w->ne[1];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_MERGER) {
        return ctx->vision_model.mm_1_b->ne[0];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_GEMMA3) {
        return ctx->vision_model.mm_input_proj_w->ne[0];
    }

    std::string proj_type = PROJECTOR_TYPE_NAMES[ctx->proj_type];
    throw std::runtime_error(string_format("%s: don't support projector with: %s currently\n", __func__, proj_type.c_str()));
}

int clip_is_minicpmv(const struct clip_ctx * ctx) {
    if (ctx->has_minicpmv_projector) {
        return ctx->minicpmv_version;
    }
    return 0;
}

bool clip_is_glm(const struct clip_ctx * ctx) {
    return ctx->has_glm_projector;
}

bool clip_is_qwen2vl(const struct clip_ctx * ctx) {
    return ctx->has_qwen2vl_merger;
}

bool clip_is_llava(const struct clip_ctx * ctx) {
    return ctx->has_llava_projector;
}

bool clip_is_gemma3(const struct clip_ctx * ctx) {
    return ctx->proj_type == PROJECTOR_TYPE_GEMMA3;
}

// Determine the number of encoder layers to iterate over
int get_deepest_feature_layer(const struct clip_ctx * ctx) {
    // Get the index of the second to last layer; this is the
    // default for models that have a llava projector
    const auto & hparams = ctx->vision_model.hparams;
    int n_layer = hparams.n_layer - 1;
    int deepest_feature_layer = -1;

    // Handle other projectors; incrementing here indicates that we
    // should use the last encoder layer for the vision features.
    if (ctx->has_minicpmv_projector || ctx->has_glm_projector || ctx->has_qwen2vl_merger) {
        n_layer += 1;
    }

    // If we set explicit vision feature layers, only go up to the deepest one
    for (const auto & feature_layer : hparams.vision_feature_layer) {
        if (feature_layer > deepest_feature_layer) {
            deepest_feature_layer = feature_layer;
        }
    }
    return deepest_feature_layer < 0 ? n_layer : deepest_feature_layer;
}

bool clip_encode_float_image (struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec) {
    clip_image_f32 clip_img;
    clip_img.buf.resize(h * w * 3);
    for (int i = 0; i < h*w*3; i++)
    {
        clip_img.buf[i] = img[i];
    }
    clip_img.nx = w;
    clip_img.ny = h;
    clip_image_encode(ctx, n_threads, &clip_img, vec);
    return true;
}

//
// API used internally with mtmd
//

projector_type clip_get_projector_type(const struct clip_ctx * ctx) {
    return ctx->proj_type;
}
