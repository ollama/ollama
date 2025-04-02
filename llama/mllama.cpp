// NOTE: This is modified from clip.cpp for Mllama only
#include "mllama.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

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

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#define REQUIRE(x)                                           \
    do {                                                     \
        if (!(x)) {                                          \
            throw std::runtime_error("REQUIRE failed: " #x); \
        }                                                    \
    } while (0)

#define LOG(fmt, ...) fprintf(stderr, "%s: " fmt "\n", __func__, ##__VA_ARGS__)

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

struct mllama_image {
    int width;
    int height;

    int num_channels = 3;
    int num_tiles = 4;

    int aspect_ratio_id;

    std::vector<float> data;
};

static std::string format(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vector<char> b(128);
    int n = vsnprintf(b.data(), b.size(), fmt, args);
    REQUIRE(n >= 0 && n < b.size());
    va_end(args);
    return std::string(b.data(), b.size());
}

//
// utilities to get data from a gguf file
//

static int get_key_index(const gguf_context *ctx, const char *key) {
    int key_index = gguf_find_key(ctx, key);
    REQUIRE(key_index != -1);
    return key_index;
}

static std::vector<uint32_t> get_u32_array(const gguf_context *ctx, const std::string &key) {
    const int i = get_key_index(ctx, key.c_str());
    const int n = gguf_get_arr_n(ctx, i);
    const uint32_t *data = (uint32_t *)gguf_get_arr_data(ctx, i);

    std::vector<uint32_t> s(n);
    for (size_t j = 0; j < s.size(); j++) {
        s[j] = data[j];
    }

    return s;
}

static uint32_t get_u32(const gguf_context *ctx, const std::string &key) {
    return gguf_get_val_u32(ctx, get_key_index(ctx, key.c_str()));
}

static float get_f32(const gguf_context *ctx, const std::string &key) {
    return gguf_get_val_f32(ctx, get_key_index(ctx, key.c_str()));
}

static std::string get_ftype(int ftype) {
    return ggml_type_name(static_cast<ggml_type>(ftype));
}

//
// mllama layers
//

struct mllama_hparams {
    uint32_t image_size;
    uint32_t patch_size;
    uint32_t hidden_size;
    uint32_t n_intermediate;
    uint32_t projection_dim;
    uint32_t n_head;
    uint32_t n_layer;
    uint32_t n_global_layer;
    uint32_t n_tiles;

    float eps;

    std::vector<bool> intermediate_layers;
};

struct mllama_layer {
    // attention
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    struct ggml_tensor *attn_gate;

    // layernorm 1
    struct ggml_tensor *ln_1_w;
    struct ggml_tensor *ln_1_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;

    struct ggml_tensor *ff_gate;

    // layernorm 2
    struct ggml_tensor *ln_2_w;
    struct ggml_tensor *ln_2_b;
};

struct mllama_vision_model {
    struct mllama_hparams hparams;

    // embeddings
    struct ggml_tensor *class_embedding;
    struct ggml_tensor *patch_embeddings;
    struct ggml_tensor *position_embeddings;
    struct ggml_tensor *position_embeddings_gate;
    struct ggml_tensor *tile_position_embeddings;
    struct ggml_tensor *tile_position_embeddings_gate;
    struct ggml_tensor *pre_tile_position_embeddings;
    struct ggml_tensor *pre_tile_position_embeddings_gate;
    struct ggml_tensor *post_tile_position_embeddings;
    struct ggml_tensor *post_tile_position_embeddings_gate;

    struct ggml_tensor *pre_ln_w;
    struct ggml_tensor *pre_ln_b;

    std::vector<mllama_layer> layers;
    std::vector<mllama_layer> global_layers;

    struct ggml_tensor *post_ln_w;
    struct ggml_tensor *post_ln_b;

    struct ggml_tensor *mm_0_w;
    struct ggml_tensor *mm_0_b;
};

struct mllama_ctx {
    struct mllama_vision_model vision_model;

    uint32_t ftype = 1;

    struct gguf_context *ctx_gguf;
    struct ggml_context *ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer = nullptr;

    ggml_backend_t backend = nullptr;
    ggml_gallocr_t compute_alloc = nullptr;
};

static ggml_tensor *mllama_image_build_encoder_layer(
    struct ggml_context *ctx0, const size_t il, const struct mllama_layer &layer, struct ggml_tensor *embeddings,
    const float eps, const int hidden_size, const int batch_size, const int n_head, const int d_head) {
    struct ggml_tensor *cur = embeddings;

    {
        // layernorm1
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ln_1_w), layer.ln_1_b);
        ggml_set_name(cur, format("%d pre layernorm", il).c_str());
    }

    {
        // self-attention
        struct ggml_tensor *Q = ggml_mul_mat(ctx0, layer.q_w, cur);
        if (layer.q_b != nullptr) {
            Q = ggml_add(ctx0, Q, layer.q_b);
        }

        Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, Q->ne[1], batch_size);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        ggml_set_name(Q, format("%d query", il).c_str());

        struct ggml_tensor *K = ggml_mul_mat(ctx0, layer.k_w, cur);
        if (layer.k_b != nullptr) {
            K = ggml_add(ctx0, K, layer.k_b);
        }

        K = ggml_reshape_4d(ctx0, K, d_head, n_head, K->ne[1], batch_size);
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        ggml_set_name(K, format("%d key", il).c_str());

        struct ggml_tensor *V = ggml_mul_mat(ctx0, layer.v_w, cur);
        if (layer.v_b != nullptr) {
            V = ggml_add(ctx0, V, layer.v_b);
        }

        V = ggml_reshape_4d(ctx0, V, d_head, n_head, V->ne[1], batch_size);
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
        ggml_set_name(V, format("%d value", il).c_str());

        struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale_inplace(ctx0, KQ, 1.0f / sqrtf((float)d_head));
        KQ = ggml_soft_max_inplace(ctx0, KQ);
        ggml_set_name(KQ, format("%d KQ", il).c_str());

        struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_reshape_4d(ctx0, KQV, d_head, KQV->ne[1], n_head, batch_size);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        KQV = ggml_cont_3d(ctx0, KQV, hidden_size, KQV->ne[2], batch_size);
        ggml_set_name(KQV, format("%d KQV", il).c_str());

        cur = ggml_mul_mat(ctx0, layer.o_w, KQV);
        if (layer.o_b != nullptr) {
            cur = ggml_add(ctx0, cur, layer.o_b);
        }
        ggml_set_name(cur, format("%d self attention", il).c_str());

        if (layer.attn_gate != nullptr) {
            cur = ggml_mul_inplace(ctx0, cur, layer.attn_gate);
            ggml_set_name(cur, format("%d self attention gate", il).c_str());
        }
    }

    cur = ggml_add(ctx0, cur, embeddings);
    ggml_set_name(cur, format("%d residual", il).c_str());

    embeddings = cur;

    {
        // layernorm2
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ln_2_w), layer.ln_2_b);
        ggml_set_name(cur, format("%d post layernorm", il).c_str());
    }

    {
        // feed forward
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff_i_w, cur), layer.ff_i_b);
        cur = ggml_gelu_inplace(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff_o_w, cur), layer.ff_o_b);
        ggml_set_name(cur, format("%d feed forward", il).c_str());

        if (layer.ff_gate != nullptr) {
            cur = ggml_mul_inplace(ctx0, cur, layer.ff_gate);
            ggml_set_name(cur, format("%d feed forward gate", il).c_str());
        }
    }

    // residual 2
    cur = ggml_add(ctx0, cur, embeddings);
    ggml_set_name(cur, format("%d residual", il).c_str());

    embeddings = cur;

    return embeddings;
}

static ggml_cgraph *mllama_image_build_graph(mllama_ctx *ctx, const mllama_image_batch *imgs) {
    const auto &model = ctx->vision_model;
    const auto &hparams = model.hparams;

    const int image_size = hparams.image_size;
    const int image_size_width = image_size;
    const int image_size_height = image_size;

    const int patch_size = hparams.patch_size;
    const int num_patches = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int num_positions = num_patches + (model.class_embedding == nullptr ? 0 : 1);
    const int hidden_size = hparams.hidden_size;
    const int n_head = hparams.n_head;
    const int d_head = hidden_size / n_head;

    const int batch_size = imgs->size;
    REQUIRE(batch_size == 1);

    int num_tiles = 4;
    int num_channels = 3;
    if (imgs->data != nullptr) {
        num_tiles = imgs->data[0].num_tiles > 0 ? imgs->data[0].num_tiles : num_tiles;
        num_channels = imgs->data[0].num_channels > 0 ? imgs->data[0].num_channels : num_channels;
    }

    struct ggml_init_params params = {
        ctx->buf_compute_meta.size(), // mem_size
        ctx->buf_compute_meta.data(), // mem_buffer
        true,                         // no_alloc
    };

    struct ggml_context *ctx0 = ggml_init(params);
    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size_width, image_size_height, num_channels, num_tiles);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    struct ggml_tensor *inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, num_tiles);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

    struct ggml_tensor *aspect_ratios = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, imgs->size);
    ggml_set_name(aspect_ratios, "aspect_ratios");
    ggml_set_input(aspect_ratios);

    if (model.pre_tile_position_embeddings != nullptr) {
        struct ggml_tensor *pre_tile_position_embeddings = ggml_get_rows(ctx0, model.pre_tile_position_embeddings, aspect_ratios);
        ggml_set_name(pre_tile_position_embeddings, "pre_tile_position_embeddings");

        pre_tile_position_embeddings = ggml_reshape_3d(ctx0, pre_tile_position_embeddings, hidden_size, 1, num_tiles);
        if (model.pre_tile_position_embeddings_gate != nullptr) {
            pre_tile_position_embeddings = ggml_mul_inplace(ctx0, pre_tile_position_embeddings, model.pre_tile_position_embeddings_gate);
        }

        inp = ggml_add(ctx0, inp, pre_tile_position_embeddings);
    }

    struct ggml_tensor *embeddings = inp;

    if (model.class_embedding != nullptr) {
        // concat class_embeddings and patch_embeddings
        embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, num_tiles);
        ggml_set_name(embeddings, "embeddings");
        ggml_set_input(embeddings);
        for (int i = 0; i < num_tiles; ++i) {
            // repeat class embeddings for each tile
            embeddings = ggml_acc_inplace(ctx0, embeddings, model.class_embedding, embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], i * embeddings->nb[2]);
        }

        embeddings = ggml_acc_inplace(ctx0, embeddings, inp, embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], model.class_embedding->nb[1]);
    }

    struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    struct ggml_tensor *position_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);
    if (model.position_embeddings_gate != nullptr) {
        position_embd = ggml_mul_inplace(ctx0, position_embd, model.position_embeddings_gate);
    }

    embeddings = ggml_add(ctx0, embeddings, position_embd);

    if (model.tile_position_embeddings != nullptr) {
        struct ggml_tensor *tile_position_embeddings = ggml_get_rows(ctx0, model.tile_position_embeddings, aspect_ratios);
        ggml_set_name(tile_position_embeddings, "tile_position_embeddings");

        tile_position_embeddings = ggml_reshape_3d(ctx0, tile_position_embeddings, hidden_size, num_positions, num_tiles);
        if (model.tile_position_embeddings_gate != nullptr) {
            tile_position_embeddings = ggml_mul_inplace(ctx0, tile_position_embeddings, model.tile_position_embeddings_gate);
        }

        embeddings = ggml_add(ctx0, embeddings, tile_position_embeddings);
    }

    // pre-layernorm
    if (model.pre_ln_w != nullptr) {
        embeddings = ggml_mul(ctx0, ggml_norm(ctx0, embeddings, hparams.eps), model.pre_ln_w);
        if (model.pre_ln_b != nullptr) {
            embeddings = ggml_add(ctx0, embeddings, model.pre_ln_b);
        }

        ggml_set_name(embeddings, "pre layernorm");
    }

    const int num_padding_patches = 8 - (embeddings->ne[1] % 8) % 8;

    embeddings = ggml_pad(ctx0, embeddings, 0, num_padding_patches, 0, 0);
    embeddings = ggml_view_3d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1] * embeddings->ne[2], batch_size, embeddings->nb[1], embeddings->nb[2] * embeddings->ne[3], 0);

    std::vector<struct ggml_tensor *> intermediate_embeddings;

    // encoder
    for (size_t il = 0; il < model.layers.size(); il++) {
        if (hparams.intermediate_layers[il]) {
            intermediate_embeddings.push_back(embeddings);
        }

        embeddings = mllama_image_build_encoder_layer(
            ctx0, il, model.layers[il], embeddings,
            hparams.eps, hidden_size, batch_size, n_head, d_head);
    }

    // post-layernorm
    if (model.post_ln_w != nullptr) {
        embeddings = ggml_mul(ctx0, ggml_norm(ctx0, embeddings, hparams.eps), model.post_ln_w);
        if (model.post_ln_b != nullptr) {
            embeddings = ggml_add(ctx0, embeddings, model.post_ln_b);
        }

        ggml_set_name(embeddings, "post layernorm");
    }

    embeddings = ggml_reshape_3d(ctx0, embeddings, hidden_size, num_positions + num_padding_patches, num_tiles);

    if (model.post_tile_position_embeddings != nullptr) {
        struct ggml_tensor *post_tile_position_embeddings = ggml_get_rows(ctx0, model.post_tile_position_embeddings, aspect_ratios);
        ggml_set_name(post_tile_position_embeddings, "post_tile_position_embeddings");

        post_tile_position_embeddings = ggml_reshape_3d(ctx0, post_tile_position_embeddings, hidden_size, 1, num_tiles);
        if (model.post_tile_position_embeddings_gate != nullptr) {
            post_tile_position_embeddings = ggml_mul(ctx0, post_tile_position_embeddings, model.post_tile_position_embeddings_gate);
        }

        embeddings = ggml_add(ctx0, embeddings, post_tile_position_embeddings);
    }

    embeddings = ggml_reshape_3d(ctx0, embeddings, hidden_size, num_tiles * (num_positions + num_padding_patches), 1);

    // global encoder
    for (size_t il = 0; il < model.global_layers.size(); il++) {
        embeddings = mllama_image_build_encoder_layer(
            ctx0, il, model.global_layers[il], embeddings,
            hparams.eps, hidden_size, batch_size, n_head, d_head);
    }

    struct ggml_tensor *stacked_embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 0, hidden_size, (num_positions + num_padding_patches) * num_tiles);
    for (size_t i = 0; i < intermediate_embeddings.size(); ++i) {
        stacked_embeddings = ggml_concat(ctx0, stacked_embeddings, ggml_reshape_3d(ctx0, intermediate_embeddings[i], 1, intermediate_embeddings[i]->ne[0], intermediate_embeddings[i]->ne[1]), 0);
    }

    stacked_embeddings = ggml_reshape_4d(ctx0, stacked_embeddings, intermediate_embeddings.size() * hidden_size, num_positions + num_padding_patches, num_tiles, batch_size);
    stacked_embeddings = ggml_unpad(ctx0, stacked_embeddings, 0, num_padding_patches, 0, 0);

    embeddings = ggml_reshape_3d(ctx0, embeddings, hidden_size, num_positions + num_padding_patches, num_tiles);
    embeddings = ggml_unpad(ctx0, embeddings, 0, num_padding_patches, 0, 0);
    embeddings = ggml_concat(ctx0, embeddings, stacked_embeddings, 0);

    // mllama projector
    embeddings = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_0_w, embeddings), model.mm_0_b);
    ggml_set_name(embeddings, "multi modal projector");

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_tensor *mllama_tensor_load(struct ggml_context *ctx, const char *name, const bool optional) {
    struct ggml_tensor *cur = ggml_get_tensor(ctx, name);
    REQUIRE(cur != nullptr || optional);
    return cur;
}

static std::vector<struct mllama_layer> mllama_layers_load(struct ggml_context *ctx, const char *prefix, const int n) {
    std::vector<struct mllama_layer> layers(n);
    for (size_t i = 0; i < layers.size(); i++) {
        auto &layer = layers[i];
        layer.ln_1_w = mllama_tensor_load(ctx, format("%s.blk.%d.ln1.weight", prefix, i).c_str(), false);
        layer.ln_1_b = mllama_tensor_load(ctx, format("%s.blk.%d.ln1.bias", prefix, i).c_str(), false);
        layer.ln_2_w = mllama_tensor_load(ctx, format("%s.blk.%d.ln2.weight", prefix, i).c_str(), false);
        layer.ln_2_b = mllama_tensor_load(ctx, format("%s.blk.%d.ln2.bias", prefix, i).c_str(), false);

        layer.k_w = mllama_tensor_load(ctx, format("%s.blk.%d.attn_k.weight", prefix, i).c_str(), false);
        layer.k_b = mllama_tensor_load(ctx, format("%s.blk.%d.attn_k.bias", prefix, i).c_str(), true);
        layer.q_w = mllama_tensor_load(ctx, format("%s.blk.%d.attn_q.weight", prefix, i).c_str(), false);
        layer.q_b = mllama_tensor_load(ctx, format("%s.blk.%d.attn_q.bias", prefix, i).c_str(), true);
        layer.v_w = mllama_tensor_load(ctx, format("%s.blk.%d.attn_v.weight", prefix, i).c_str(), false);
        layer.v_b = mllama_tensor_load(ctx, format("%s.blk.%d.attn_v.bias", prefix, i).c_str(), true);
        layer.o_w = mllama_tensor_load(ctx, format("%s.blk.%d.attn_out.weight", prefix, i).c_str(), false);
        layer.o_b = mllama_tensor_load(ctx, format("%s.blk.%d.attn_out.bias", prefix, i).c_str(), true);

        layer.ff_i_w = mllama_tensor_load(ctx, format("%s.blk.%d.ffn_down.weight", prefix, i).c_str(), false);
        layer.ff_i_b = mllama_tensor_load(ctx, format("%s.blk.%d.ffn_down.bias", prefix, i).c_str(), false);
        layer.ff_o_w = mllama_tensor_load(ctx, format("%s.blk.%d.ffn_up.weight", prefix, i).c_str(), false);
        layer.ff_o_b = mllama_tensor_load(ctx, format("%s.blk.%d.ffn_up.bias", prefix, i).c_str(), false);

        layer.attn_gate = mllama_tensor_load(ctx, format("%s.blk.%d.attn_gate", prefix, i).c_str(), true);
        layer.ff_gate = mllama_tensor_load(ctx, format("%s.blk.%d.ffn_gate", prefix, i).c_str(), true);
    }

    return layers;
}

// read and create ggml_context containing the tensors and their data
struct mllama_ctx *mllama_model_load(const char *fname, const int verbosity = 1) {
    struct ggml_context *meta = nullptr;

    struct gguf_init_params params = {
        true,  // no_alloc
        &meta, // ctx
    };

    struct gguf_context *ctx = gguf_init_from_file(fname, params);
    REQUIRE(ctx != nullptr);

    if (verbosity >= 1) {
        const int n_tensors = gguf_get_n_tensors(ctx);
        const int n_kv = gguf_get_n_kv(ctx);
        const std::string ftype = get_ftype(get_u32(ctx, "general.file_type"));
        const int idx_desc = get_key_index(ctx, "general.description");
        const std::string description = gguf_get_val_str(ctx, idx_desc);
        const int idx_name = gguf_find_key(ctx, "general.name");
        if (idx_name != -1) { // make name optional temporarily as some of the uploaded models missing it due to a bug
            const std::string name = gguf_get_val_str(ctx, idx_name);
            LOG("model name:   %s", name.c_str());
        }
        LOG("description:  %s", description.c_str());
        LOG("GGUF version: %d", gguf_get_version(ctx));
        LOG("alignment:    %zu", gguf_get_alignment(ctx));
        LOG("n_tensors:    %d", n_tensors);
        LOG("n_kv:         %d", n_kv);
        LOG("ftype:        %s", ftype.c_str());
        LOG("");
    }
    const int n_tensors = gguf_get_n_tensors(ctx);

    mllama_ctx *new_mllama = new mllama_ctx{};

    ggml_backend_t backend = ggml_backend_init_best();
    if (backend == nullptr) {
        LOG("%s: failed to initialize backend\n", __func__);
        mllama_free(new_mllama);
        gguf_free(ctx);
        return nullptr;
    }
    LOG("%s: using %s backend\n", __func__, ggml_backend_name(backend));
    new_mllama->backend = backend;

    // load tensors
    {
        std::vector<uint8_t> read_buf;
        struct ggml_init_params params = {
            (n_tensors + 1) * ggml_tensor_overhead(), // mem_size
            nullptr,                                  // mem_buffer
            true,                                     // no_alloc
        };

        new_mllama->ctx_data = ggml_init(params);
        if (!new_mllama->ctx_data) {
            LOG("ggml_init() failed");
            mllama_free(new_mllama);
            gguf_free(ctx);
            return nullptr;
        }

#ifdef _WIN32
        int wlen = MultiByteToWideChar(CP_UTF8, 0, fname, -1, NULL, 0);
        if (!wlen) {
            return NULL;
        }
        wchar_t * wbuf = (wchar_t *) malloc(wlen * sizeof(wchar_t));
        wlen = MultiByteToWideChar(CP_UTF8, 0, fname, -1, wbuf, wlen);
        if (!wlen) {
            free(wbuf);
            return NULL;
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
            LOG("cannot open model file for loading tensors\n");
            mllama_free(new_mllama);
            gguf_free(ctx);
            return nullptr;
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char *name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor *t = ggml_get_tensor(meta, name);
            struct ggml_tensor *cur = ggml_dup_tensor(new_mllama->ctx_data, t);
            ggml_set_name(cur, name);
        }

        // alloc memory and offload data
        new_mllama->params_buffer = ggml_backend_alloc_ctx_tensors(new_mllama->ctx_data, new_mllama->backend);
        for (int i = 0; i < n_tensors; ++i) {
            const char *name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor *cur = ggml_get_tensor(new_mllama->ctx_data, name);
            const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                LOG("failed to seek for tensor %s\n", name);
                mllama_free(new_mllama);
                gguf_free(ctx);
                return nullptr;
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(new_mllama->params_buffer)) {
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
    }

    // vision model
    // load vision model
    auto &vision_model = new_mllama->vision_model;
    auto &hparams = vision_model.hparams;
    hparams.hidden_size = get_u32(ctx, "mllama.vision.embedding_length");
    hparams.n_head = get_u32(ctx, "mllama.vision.attention.head_count");
    hparams.n_intermediate = get_u32(ctx, "mllama.vision.feed_forward_length");
    hparams.n_layer = get_u32(ctx, "mllama.vision.block_count");
    hparams.n_global_layer = get_u32(ctx, "mllama.vision.global.block_count");
    hparams.n_tiles = get_u32(ctx, "mllama.vision.max_num_tiles");
    hparams.image_size = get_u32(ctx, "mllama.vision.image_size");
    hparams.patch_size = get_u32(ctx, "mllama.vision.patch_size");
    hparams.projection_dim = get_u32(ctx, "mllama.vision.projection_dim");
    hparams.eps = get_f32(ctx, "mllama.vision.attention.layer_norm_epsilon");

    std::vector<uint32_t> intermediate_layers_indices = get_u32_array(ctx, "mllama.vision.intermediate_layers_indices");
    hparams.intermediate_layers.resize(hparams.n_layer);
    for (size_t i = 0; i < intermediate_layers_indices.size(); i++) {
        hparams.intermediate_layers[intermediate_layers_indices[i]] = true;
    }

    if (verbosity >= 2) {
        LOG("");
        LOG("vision model hparams");
        LOG("image_size         %d", hparams.image_size);
        LOG("patch_size         %d", hparams.patch_size);
        LOG("v_hidden_size      %d", hparams.hidden_size);
        LOG("v_n_intermediate   %d", hparams.n_intermediate);
        LOG("v_projection_dim   %d", hparams.projection_dim);
        LOG("v_n_head           %d", hparams.n_head);
        LOG("v_n_layer          %d", hparams.n_layer);
        LOG("v_n_global_layer   %d", hparams.n_global_layer);
        LOG("v_eps              %f", hparams.eps);
    }

    vision_model.class_embedding = mllama_tensor_load(new_mllama->ctx_data, "v.class_embd", true);
    vision_model.patch_embeddings = mllama_tensor_load(new_mllama->ctx_data, "v.patch_embd.weight", true);

    vision_model.position_embeddings = mllama_tensor_load(new_mllama->ctx_data, "v.position_embd.weight", true);
    vision_model.position_embeddings_gate = mllama_tensor_load(new_mllama->ctx_data, "v.position_embd.gate", true);

    vision_model.pre_ln_w = mllama_tensor_load(new_mllama->ctx_data, "v.pre_ln.weight", true);
    vision_model.pre_ln_b = mllama_tensor_load(new_mllama->ctx_data, "v.pre_ln.bias", true);
    vision_model.post_ln_w = mllama_tensor_load(new_mllama->ctx_data, "v.post_ln.weight", true);
    vision_model.post_ln_b = mllama_tensor_load(new_mllama->ctx_data, "v.post_ln.bias", true);

    vision_model.tile_position_embeddings = mllama_tensor_load(new_mllama->ctx_data, "v.tile_position_embd.weight", true);
    vision_model.tile_position_embeddings_gate = mllama_tensor_load(new_mllama->ctx_data, "v.tile_position_embd.gate", true);

    vision_model.pre_tile_position_embeddings = mllama_tensor_load(new_mllama->ctx_data, "v.pre_tile_position_embd.weight", true);
    vision_model.pre_tile_position_embeddings_gate = mllama_tensor_load(new_mllama->ctx_data, "v.pre_tile_position_embd.gate", true);

    vision_model.post_tile_position_embeddings = mllama_tensor_load(new_mllama->ctx_data, "v.post_tile_position_embd.weight", true);
    vision_model.post_tile_position_embeddings_gate = mllama_tensor_load(new_mllama->ctx_data, "v.post_tile_position_embd.gate", true);

    vision_model.mm_0_w = mllama_tensor_load(new_mllama->ctx_data, "mm.0.weight", false);
    vision_model.mm_0_b = mllama_tensor_load(new_mllama->ctx_data, "mm.0.bias", false);

    vision_model.layers = mllama_layers_load(new_mllama->ctx_data, "v", hparams.n_layer);
    vision_model.global_layers = mllama_layers_load(new_mllama->ctx_data, "v.global", hparams.n_global_layer);

    ggml_free(meta);

    new_mllama->ctx_gguf = ctx;

    {
        // measure mem requirement and allocate
        new_mllama->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
        new_mllama->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(new_mllama->backend));
        struct mllama_image_batch batch;
        batch.size = 1;
        ggml_cgraph *gf = mllama_image_build_graph(new_mllama, &batch);
        ggml_gallocr_reserve(new_mllama->compute_alloc, gf);
        size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(new_mllama->compute_alloc, 0);
        LOG("compute allocated memory: %.2f MB", compute_memory_buffer_size / 1024.0 / 1024.0);
    }

    return new_mllama;
}

struct mllama_image *mllama_image_init() {
    return new mllama_image();
}

void mllama_image_free(struct mllama_image *img) { delete img; }
void mllama_image_batch_free(struct mllama_image_batch *batch) {
    if (batch->size > 0) {
        delete[] batch->data;
        batch->size = 0;
    }
}

bool mllama_image_load_from_data(const void *data, const int n, const int width, const int height, const int num_channels, const int num_tiles, const int aspect_ratio_id, struct mllama_image *img) {
    img->width = width;
    img->height = height;
    img->num_channels = num_channels;
    img->num_tiles = num_tiles;
    img->aspect_ratio_id = aspect_ratio_id;
    img->data.resize(n);

    memcpy(img->data.data(), data, n);
    return true;
}

inline int mllama(int x, int lower, int upper) {
    return std::max(lower, std::min(x, upper));
}

void mllama_free(mllama_ctx *ctx) {
    ggml_free(ctx->ctx_data);
    gguf_free(ctx->ctx_gguf);

    ggml_backend_buffer_free(ctx->params_buffer);
    ggml_backend_free(ctx->backend);
    ggml_gallocr_free(ctx->compute_alloc);
    delete ctx;
}

bool mllama_image_encode(struct mllama_ctx *ctx, const int n_threads, mllama_image *img, float *vec) {
    mllama_image_batch imgs{};
    imgs.size = 1;
    imgs.data = img;
    return mllama_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool mllama_image_batch_encode(mllama_ctx *ctx, const int n_threads, const mllama_image_batch *imgs, float *vec) {
    int batch_size = imgs->size;
    REQUIRE(batch_size == 1);

    // build the inference graph
    ggml_cgraph *gf = mllama_image_build_graph(ctx, imgs);
    ggml_gallocr_alloc_graph(ctx->compute_alloc, gf);

    // set inputs
    const auto &model = ctx->vision_model;
    const auto &hparams = model.hparams;

    const int image_size = hparams.image_size;
    int image_size_width = image_size;
    int image_size_height = image_size;

    const int patch_size = hparams.patch_size;
    const int num_patches = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int num_positions = num_patches + (model.class_embedding == nullptr ? 0 : 1);

    {
        struct ggml_tensor *inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        ggml_backend_tensor_set(inp_raw, imgs->data[0].data.data(), 0, ggml_nbytes(inp_raw));
    }

    {
        struct ggml_tensor *embeddings = ggml_graph_get_tensor(gf, "embeddings");
        if (embeddings != nullptr) {
            void *zeros = malloc(ggml_nbytes(embeddings));
            memset(zeros, 0, ggml_nbytes(embeddings));
            ggml_backend_tensor_set(embeddings, zeros, 0, ggml_nbytes(embeddings));
            free(zeros);
        }
    }

    {
        struct ggml_tensor *positions = ggml_graph_get_tensor(gf, "positions");
        if (positions != nullptr) {
            int *positions_data = (int *)malloc(ggml_nbytes(positions));
            for (int i = 0; i < num_positions; i++) {
                positions_data[i] = i;
            }
            ggml_backend_tensor_set(positions, positions_data, 0, ggml_nbytes(positions));
            free(positions_data);
        }
    }

    {
        struct ggml_tensor *aspect_ratios = ggml_graph_get_tensor(gf, "aspect_ratios");
        if (aspect_ratios != nullptr) {
            int *aspect_ratios_data = (int *)malloc(ggml_nbytes(aspect_ratios));
            aspect_ratios_data[0] = imgs->data[0].aspect_ratio_id;
            ggml_backend_tensor_set(aspect_ratios, aspect_ratios_data, 0, ggml_nbytes(aspect_ratios));
            free(aspect_ratios_data);
        }
    }

    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }

    ggml_backend_graph_compute(ctx->backend, gf);

    // the last node is the embedding tensor
    struct ggml_tensor *embeddings = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);

    // copy the embeddings to the location passed by the user
    ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));

    return true;
}

int32_t mllama_image_size(const struct mllama_ctx *ctx) {
    return ctx->vision_model.hparams.image_size;
}

int32_t mllama_patch_size(const struct mllama_ctx *ctx) {
    return ctx->vision_model.hparams.patch_size;
}

int32_t mllama_hidden_size(const struct mllama_ctx *ctx) {
    return ctx->vision_model.hparams.hidden_size;
}

int mllama_n_patches(const struct mllama_ctx *ctx) {
    const auto &hparams = ctx->vision_model.hparams;
    return (hparams.image_size / hparams.patch_size) * (hparams.image_size / hparams.patch_size);
}

int mllama_n_positions(const struct mllama_ctx *ctx) {
    return mllama_n_patches(ctx) + (ctx->vision_model.class_embedding == nullptr ? 0 : 1);
}

int mllama_n_tiles(const struct mllama_ctx *ctx) {
    return ctx->vision_model.hparams.n_tiles;
}

int mllama_n_embd(const struct mllama_ctx *ctx) {
    return ctx->vision_model.hparams.projection_dim;
}

size_t mllama_n_embd_bytes(const struct mllama_ctx *ctx) {
    return mllama_n_positions(ctx) * mllama_n_embd(ctx) * mllama_n_tiles(ctx) * sizeof(float);
}
