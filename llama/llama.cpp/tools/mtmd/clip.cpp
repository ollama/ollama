#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "clip-graph.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <cinttypes>
#include <limits>
#include <array>
#include <functional>

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

struct clip_logger_state g_logger_state = {clip_log_callback_default, NULL};

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


struct clip_ctx {
    clip_model model;

    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_ptr buf;

    int max_nodes = 8192;
    ggml_backend_sched_ptr sched;
    clip_flash_attn_type flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    bool is_allocated = false;

    // for debugging
    bool debug_graph = false;
    std::vector<ggml_tensor *> debug_print_tensors;

    clip_ctx(clip_context_params & ctx_params) {
        flash_attn_type = ctx_params.flash_attn_type;
        debug_graph = std::getenv("MTMD_DEBUG_GRAPH") != nullptr;
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        if (ctx_params.use_gpu) {
            auto backend_name = std::getenv("MTMD_BACKEND_DEVICE");
            if (backend_name != nullptr) {
                backend = ggml_backend_init_by_name(backend_name, nullptr);
                if (!backend) {
                    LOG_WRN("%s: Warning: Failed to initialize \"%s\" backend, falling back to default GPU backend\n", __func__, backend_name);
                }
            }
            if (!backend) {
                backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
                backend = backend ? backend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
            }
        }

        if (backend) {
            LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: CLIP using CPU backend\n", __func__);
        }

        if (ctx_params.image_min_tokens > 0) {
            model.hparams.custom_image_min_tokens = ctx_params.image_min_tokens;
        }
        if (ctx_params.image_max_tokens > 0) {
            model.hparams.custom_image_max_tokens = ctx_params.image_max_tokens;
        }

        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), 8192, false, true)
        );
    }

    ~clip_ctx() {
        ggml_backend_free(backend);
        if (backend != backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }

    // this function is added so that we don't change too much of the existing code
    projector_type proj_type() const {
        return model.proj_type;
    }
};

//
// clip_graph
//

clip_graph::clip_graph(clip_ctx * ctx, const clip_image_f32 & img) :
        model(ctx->model),
        hparams(model.hparams),
        proj_type(ctx->proj_type()),
        img(img),
        patch_size(hparams.patch_size),
        n_patches_x(img.nx / patch_size),
        n_patches_y(img.ny / patch_size),
        n_patches(n_patches_x * n_patches_y),
        n_embd(hparams.n_embd),
        n_head(hparams.n_head),
        d_head(n_embd / n_head),
        n_layer(hparams.n_layer),
        n_mmproj_embd(clip_n_mmproj_embd(ctx)),
        eps(hparams.eps),
        kq_scale(1.0f / sqrtf((float)d_head)),
        flash_attn_type(ctx->flash_attn_type),
        debug_graph(ctx->debug_graph),
        debug_print_tensors(ctx->debug_print_tensors) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ctx0_ptr.reset(ggml_init(params));
    ctx0 = ctx0_ptr.get();
    gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
}

void clip_graph::cb(ggml_tensor * cur0, const char * name, int il) const {
    if (debug_graph) {
        ggml_tensor * cur = ggml_cpy(ctx0, cur0, ggml_dup_tensor(ctx0, cur0));
        std::string cur_name = il >= 0 ? std::string(name) + "_" + std::to_string(il) : name;
        ggml_set_name(cur, cur_name.c_str());
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        debug_print_tensors.push_back(cur);
    }
}

// siglip2 naflex
ggml_tensor * clip_graph::resize_position_embeddings(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny / patch_size;
    const int width        = img.nx / patch_size;
    const uint32_t mode    = interpolation_mode;
    const int n_per_side   = (int)std::sqrt(pos_embd->ne[1]);

    GGML_ASSERT(pos_embd);

    if (height == n_per_side && width == n_per_side) {
        return pos_embd;
    }

    pos_embd = ggml_reshape_3d(ctx0, pos_embd, n_embd, n_per_side, n_per_side);  // -> (n_embd, n_per_side, n_per_side)
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 0, 1, 3);                         // -> (n_per_side, n_per_side, n_embd)
    pos_embd = ggml_interpolate(ctx0, pos_embd, width, height, n_embd, 1, mode); // -> (width, height, n_embd)
    pos_embd = ggml_permute(ctx0, pos_embd, 1, 2, 0, 3);                         // -> (n_embd, width, height)
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);             // -> (n_embd, width * height)

    return pos_embd;
}

// build vision transformer (ViT) cgraph
// this function should cover most of the models
// if your model has specific features, you should probably duplicate this function
ggml_tensor * clip_graph::build_vit(
            ggml_tensor * inp,
            int64_t n_pos,
            norm_type norm_t,
            ffn_op_type ffn_t,
            ggml_tensor * learned_pos_embd,
            std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> add_pos
        ) {
    if (learned_pos_embd) {
        inp = ggml_add(ctx0, inp, learned_pos_embd);
        cb(inp, "pos_embed", -1);
    }

    ggml_tensor * inpL = inp;

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "layer_inp_normed", il);

        // self-attention
        {
            ggml_tensor * Qcur = nullptr;
            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;
            if (layer.qkv_w != nullptr) {
                // fused qkv
                cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
                if (layer.qkv_b != nullptr) {
                    cur = ggml_add(ctx0, cur, layer.qkv_b);
                }

                Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ 0);

                Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, n_embd));

                Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, 2 * n_embd));

                // TODO: q/k norm requires row size == n_embd, while here it's d_head
                // we can add support in the future if needed
                GGML_ASSERT(layer.q_norm == nullptr && layer.k_norm == nullptr);

            } else {
                // separate q, k, v
                Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }

                Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }

                Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                if (layer.q_norm) {
                    Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                    cb(Qcur, "Qcur_norm", il);
                }

                if (layer.k_norm) {
                    Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                    cb(Kcur, "Kcur_norm", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (add_pos) {
                Qcur = add_pos(Qcur, layer);
                Kcur = add_pos(Kcur, layer);
                cb(Qcur, "Qcur_pos", il);
                cb(Kcur, "Kcur_pos", il);
            }

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (layer.ls_1_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_1_w);
            cb(cur, "attn_out_scaled", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            ffn_t, il);

        cb(cur, "ffn_out", il);

        if (layer.ls_2_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_2_w);
            cb(cur, "ffn_out_scaled", il);
        }

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    if (model.audio_has_avgpool()) {
        ggml_tensor * cur = inpL;
        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont(ctx0, cur);
        cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0);
        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont(ctx0, cur);
        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, -1);
    }
    return inpL;
}

// build the input after conv2d (inp_raw --> patches)
// returns tensor with shape [n_embd, n_patches]
ggml_tensor * clip_graph::build_inp() {
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    if (model.patch_bias) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }
    return inp;
}

ggml_tensor * clip_graph::build_inp_raw(int channels) {
    ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, img.nx, img.ny, channels);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    return inp_raw;
}

ggml_tensor * clip_graph::build_norm(
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb,
        norm_type type,
        float norm_eps,
        int il) const {

    cur = type == NORM_TYPE_RMS
        ? ggml_rms_norm(ctx0, cur, norm_eps)
        : ggml_norm(ctx0, cur, norm_eps);

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        cb(cur, "norm_w", il);
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
        cb(cur, "norm_b", il);
    }

    return cur;
}

ggml_tensor * clip_graph::build_ffn(
        ggml_tensor * cur,
        ggml_tensor * up,
        ggml_tensor * up_b,
        ggml_tensor * gate,
        ggml_tensor * gate_b,
        ggml_tensor * down,
        ggml_tensor * down_b,
        ffn_op_type type_op,
        int il) const {

    ggml_tensor * tmp = up ? ggml_mul_mat(ctx0, up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (gate) {
        cur = ggml_mul_mat(ctx0, gate, cur);
        cb(cur, "ffn_gate", il);

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }
    } else {
        cur = tmp;
    }

    // we only support parallel ffn for now
    switch (type_op) {
        case FFN_SILU:
            if (gate) {
                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case FFN_GELU:
            if (gate) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
            } break;
        case FFN_GELU_ERF:
            if (gate) {
                cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_erf", il);
            } else {
                cur = ggml_gelu_erf(ctx0, cur);
                cb(cur, "ffn_gelu_erf", il);
            } break;
        case FFN_GELU_QUICK:
            if (gate) {
                cur = ggml_geglu_quick_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_quick", il);
            } else {
                cur = ggml_gelu_quick(ctx0, cur);
                cb(cur, "ffn_gelu_quick", il);
            } break;
    }

    if (down) {
        cur = ggml_mul_mat(ctx0, down, cur);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    return cur;
}

ggml_tensor * clip_graph::build_attn(
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float kq_scale,
        int il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    //cb(k, "k", il);

    ggml_tensor * cur;

    if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);

        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

    } else {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v = ggml_cont(ctx0, v);

        const auto n_tokens = q->ne[1];
        const auto n_head   = q->ne[2];

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        // F32 may not needed for vision encoders?
        // ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*n_head, n_tokens);
    }

    cb(cur, "kqv_out", il);

    if (wo) {
        cur = ggml_mul_mat(ctx0, wo, cur);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// implementation of the 2D RoPE without adding a new op in ggml
// this is not efficient (use double the memory), but works on all backends
// TODO: there was a more efficient which relies on ggml_view and ggml_rope_ext_inplace, but the rope inplace does not work well with non-contiguous tensors ; we should fix that and revert back to the original implementation in https://github.com/ggml-org/llama.cpp/pull/13065
ggml_tensor * clip_graph::build_rope_2d(
    ggml_context * ctx0,
    ggml_tensor * cur,
    ggml_tensor * pos_a, // first half
    ggml_tensor * pos_b, // second half
    const float freq_base,
    const bool interleave_freq
) {
    const int64_t n_dim  = cur->ne[0];
    const int64_t n_head = cur->ne[1];
    const int64_t n_pos  = cur->ne[2];

    // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
    // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
    // first half of cur will use 1e-0, 1e-2 (even)
    // second half of cur will use 1e-1, 1e-3 (odd)
    // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
    //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
    // then for the second half, we use freq_scale to shift the inv_freq
    //  ^ why? replace (2i) with (2i+1) in the above equation
    const float freq_scale_odd = interleave_freq
                                ? std::pow(freq_base, (float)-2/n_dim)
                                : 1.0;

    // first half
    ggml_tensor * first;
    {
        first = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            ggml_row_size(cur->type, n_dim),
            ggml_row_size(cur->type, n_dim*n_head),
            0);
        first = ggml_rope_ext(
            ctx0,
            first,
            pos_a,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    // second half
    ggml_tensor * second;
    {
        second = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            ggml_row_size(cur->type, n_dim),
            ggml_row_size(cur->type, n_dim*n_head),
            n_dim/2 * ggml_element_size(cur));
        second = ggml_rope_ext(
            ctx0,
            second,
            pos_b,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            freq_scale_odd,
            0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    cur = ggml_concat(ctx0, first, second, 0);
    return cur;
}

// Generic function to stack frames for audio processing
// Abstracts out the StackAudioFrames logic used by ultravox
ggml_tensor * clip_graph::build_stack(ggml_tensor * cur, int32_t stack_factor, int32_t n_embed) {
    if (stack_factor <= 1) {
        return cur;
    }

    int64_t total_elements = ggml_nelements(cur);
    int64_t stride = n_embed * stack_factor;

    // Calculate padded length
    int64_t padded_len = GGML_PAD(total_elements, stride);
    int64_t pad = padded_len - total_elements;

    if (pad > 0) {
        // Pad the tensor to make it divisible by stride
        cur = ggml_view_1d(ctx0, cur, total_elements, 0);
        cur = ggml_pad(ctx0, cur, pad, 0, 0, 0);
    }

    // Reshape to [stride, padded_len / stride]
    cur = ggml_view_2d(ctx0, cur, stride, padded_len / stride,
                        ggml_row_size(cur->type, stride), 0);
    return cur;
}

// aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
// support dynamic resolution
ggml_tensor * clip_graph::build_patch_merge_permute(ggml_tensor * cur, int scale_factor) {
    GGML_ASSERT(scale_factor > 1);

    const int n_embd = cur->ne[0];
    int width  = img.nx / patch_size;
    int height = img.ny / patch_size;

    // pad width and height to factor
    const int64_t pad_width  = CLIP_ALIGN(width,  scale_factor) - width;
    const int64_t pad_height = CLIP_ALIGN(height, scale_factor) - height;
    cur = ggml_reshape_3d(ctx0, cur, n_embd, width, height);
    if (pad_width || pad_height) {
        cur     = ggml_pad(ctx0, cur, 0, pad_width, pad_height, 0);
        width  += pad_width;
        height += pad_height;
    }

    // unshuffle h
    cur = ggml_reshape_3d(ctx0, cur, n_embd * scale_factor, width / scale_factor, height);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    // unshuffle w
    cur = ggml_cont_3d(ctx0, cur, n_embd * scale_factor * scale_factor, height / scale_factor, width / scale_factor);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    cur = ggml_cont_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    cb(cur, "pixel_shuffle", -1);

    return cur;
}

static ggml_cgraph * clip_image_build_graph(clip_ctx * ctx, const clip_image_f32_batch & imgs) {
    GGML_ASSERT(imgs.entries.size() == 1 && "n_batch > 1 is not supported");

    const clip_image_f32 & img = *imgs.entries[0];
    std::unique_ptr<clip_graph> builder;

    switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_JANUS_PRO:
            {
                builder = std::make_unique<clip_graph_siglip>(ctx, img);
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                builder = std::make_unique<clip_graph_pixtral>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            {
                builder = std::make_unique<clip_graph_qwen2vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN3VL:
            {
                builder = std::make_unique<clip_graph_qwen3vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                builder = std::make_unique<clip_graph_minicpmv>(ctx, img);
            } break;
        case PROJECTOR_TYPE_INTERNVL:
            {
                builder = std::make_unique<clip_graph_internvl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                builder = std::make_unique<clip_graph_llama4>(ctx, img);
            } break;
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_GLMA:
            {
                builder = std::make_unique<clip_graph_whisper_enc>(ctx, img);
            } break;
        case PROJECTOR_TYPE_KIMIVL:
            {
                builder = std::make_unique<clip_graph_kimivl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_COGVLM:
            {
                builder = std::make_unique<clip_graph_cogvlm>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                builder = std::make_unique<clip_graph_llava>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GLM4V:
            {
                builder = std::make_unique<clip_graph_glm4v>(ctx, img);
            } break;
        default:
            GGML_ABORT("missing cgraph builder");
    }

    return builder->build();
}

//
// clip_model_loader
//

struct clip_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    std::string fname;

    size_t model_size = 0; // in bytes

    bool has_vision = false;
    bool has_audio  = false;

    // TODO @ngxson : we should not pass clip_ctx here, it should be clip_model
    clip_model_loader(const char * fname) : fname(fname) {
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

        // modalities
        {
            get_bool(KEY_HAS_VISION_ENC, has_vision, false);
            get_bool(KEY_HAS_AUDIO_ENC,  has_audio,  false);

            if (has_vision) {
                LOG_INF("%s: has vision encoder\n", __func__);
            }
            if (has_audio) {
                LOG_INF("%s: has audio encoder\n", __func__);
            }
        }

        // tensors
        {
            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t offset = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf.get(), i);
                ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                    __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams(clip_model & model, clip_modality modality) {
        auto & hparams = model.hparams;
        std::string log_ffn_op; // for logging

        // sanity check
        if (modality == CLIP_MODALITY_VISION) {
            GGML_ASSERT(has_vision);
        } else if (modality == CLIP_MODALITY_AUDIO) {
            GGML_ASSERT(has_audio);
        }
        model.modality = modality;


        // projector type
        std::string proj_type;
        {
            // default key
            get_string(KEY_PROJ_TYPE, proj_type, false);

            // for models with mixed modalities
            if (proj_type.empty()) {
                if (modality == CLIP_MODALITY_VISION) {
                    get_string(KEY_VISION_PROJ_TYPE, proj_type, false);
                    if (proj_type.empty()) {
                        // Assume MLP if no projector type listed
                        proj_type = "mlp";
                    }
                } else if (modality == CLIP_MODALITY_AUDIO) {
                    get_string(KEY_AUDIO_PROJ_TYPE, proj_type, false);
                } else {
                    GGML_ABORT("unknown modality");
                }
            }

            model.proj_type = clip_projector_type_from_string(proj_type);

            if (model.proj_type == PROJECTOR_TYPE_UNKNOWN) {
                throw std::runtime_error(string_format("%s: unknown projector type: %s\n", __func__, proj_type.c_str()));
            }

            // correct arch for multimodal models (legacy method)
            if (model.proj_type == PROJECTOR_TYPE_QWEN25O) {
                model.proj_type = modality == CLIP_MODALITY_VISION
                                    ? PROJECTOR_TYPE_QWEN25VL
                                    : PROJECTOR_TYPE_QWEN2A;
            }
        }

        const bool is_vision = model.modality == CLIP_MODALITY_VISION;
        const bool is_audio  = model.modality == CLIP_MODALITY_AUDIO;

        // other hparams
        {
            const char * prefix = is_vision ? "vision" : "audio";
            get_u32(string_format(KEY_N_EMBD,         prefix), hparams.n_embd);
            get_u32(string_format(KEY_N_HEAD,         prefix), hparams.n_head);
            get_u32(string_format(KEY_N_FF,           prefix), hparams.n_ff);
            get_u32(string_format(KEY_N_BLOCK,        prefix), hparams.n_layer);
            get_u32(string_format(KEY_PROJ_DIM,       prefix), hparams.projection_dim);
            get_f32(string_format(KEY_LAYER_NORM_EPS, prefix), hparams.eps);

            if (is_vision) {
                get_u32(KEY_IMAGE_SIZE, hparams.image_size);
                get_u32(KEY_PATCH_SIZE, hparams.patch_size);
                get_u32(KEY_IMAGE_CROP_RESOLUTION, hparams.image_crop_resolution, false);
                get_i32(KEY_MINICPMV_VERSION, hparams.minicpmv_version, false); // legacy
                get_u32(KEY_MINICPMV_QUERY_NUM, hparams.minicpmv_query_num, false);
                if (hparams.minicpmv_query_num == 0) {
                    // Fallback to hardcoded values for legacy models
                    if (hparams.minicpmv_version == 3) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 4) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 5) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 6) {
                        hparams.minicpmv_query_num = 64;
                    } else {
                        hparams.minicpmv_query_num = 96;
                    }
                }
            } else if (is_audio) {
                get_u32(KEY_A_NUM_MEL_BINS, hparams.n_mel_bins);
                // some hparams are unused, but still need to set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;

            } else {
                GGML_ASSERT(false && "unknown modality");
            }

            // for pinpoints, we need to convert it into a list of resolution candidates
            {
                std::vector<int> pinpoints;
                get_arr_int(KEY_IMAGE_GRID_PINPOINTS, pinpoints, false);
                if (!pinpoints.empty()) {
                    for (size_t i = 0; i < pinpoints.size(); i += 2) {
                        hparams.image_res_candidates.push_back({
                            pinpoints[i],
                            pinpoints[i+1],
                        });
                    }
                }
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;

            hparams.has_llava_projector = model.proj_type == PROJECTOR_TYPE_MLP
                                       || model.proj_type == PROJECTOR_TYPE_MLP_NORM
                                       || model.proj_type == PROJECTOR_TYPE_LDP
                                       || model.proj_type == PROJECTOR_TYPE_LDPV2;

            {
                bool use_gelu = false;
                bool use_silu = false;
                get_bool(KEY_USE_GELU, use_gelu, false);
                get_bool(KEY_USE_SILU, use_silu, false);
                if (use_gelu && use_silu) {
                    throw std::runtime_error(string_format("%s: both use_gelu and use_silu are set to true\n", __func__));
                }
                if (use_gelu) {
                    hparams.ffn_op = FFN_GELU;
                    log_ffn_op = "gelu";
                } else if (use_silu) {
                    hparams.ffn_op = FFN_SILU;
                    log_ffn_op = "silu";
                } else {
                    hparams.ffn_op = FFN_GELU_QUICK;
                    log_ffn_op = "gelu_quick";
                }
            }

            {
                std::string mm_patch_merge_type;
                get_string(KEY_MM_PATCH_MERGE_TYPE, mm_patch_merge_type, false);
                if (mm_patch_merge_type == "spatial_unpad") {
                    hparams.mm_patch_merge_type = PATCH_MERGE_SPATIAL_UNPAD;
                }
            }

            if (is_vision) {
                int idx_mean = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_MEAN);
                int idx_std  = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_STD);
                GGML_ASSERT(idx_mean >= 0 && "image_mean not found");
                GGML_ASSERT(idx_std >= 0  && "image_std not found");
                const float * mean_data = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_mean);
                const float * std_data  = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_std);
                for (int i = 0; i < 3; ++i) {
                    hparams.image_mean[i] = mean_data[i];
                    hparams.image_std[i]  = std_data[i];
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

            // model-specific params
            switch (model.proj_type) {
                case PROJECTOR_TYPE_MINICPMV:
                    {
                        if (hparams.minicpmv_version == 0) {
                            hparams.minicpmv_version = 2; // default to 2 if not set
                        }
                    } break;
                case PROJECTOR_TYPE_INTERNVL:
                    {
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;
                case PROJECTOR_TYPE_IDEFICS3:
                    {
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                    } break;
                case PROJECTOR_TYPE_LFM2:
                    {
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // ref: https://huggingface.co/LiquidAI/LFM2-VL-3B/blob/main/preprocessor_config.json
                        // config above specifies number of tokens after downsampling, while here it is before, relax lowerbound to 64
                        hparams.set_limit_image_tokens(64, 1024);
                    } break;
                case PROJECTOR_TYPE_PIXTRAL:
                case PROJECTOR_TYPE_LIGHTONOCR:
                    {
                        // ref: https://huggingface.co/mistral-community/pixtral-12b/blob/main/preprocessor_config.json
                        // TODO: verify the image_min_tokens
                        hparams.n_merge = 1; // the original pixtral does not use patch merging
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.set_limit_image_tokens(8, 1024);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_KIMIVL:
                    {
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // TODO: check kimivl preprocessor for exact values
                        hparams.set_limit_image_tokens(8, 1024);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_GEMMA3:
                    {
                        // default value (used by all model sizes in gemma 3 family)
                        // number of patches for each **side** is reduced by a factor of 4
                        hparams.n_merge = 4;
                        // test model (tinygemma3) has a different value, we optionally read it
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;
                case PROJECTOR_TYPE_QWEN2VL:
                case PROJECTOR_TYPE_QWEN25VL:
                case PROJECTOR_TYPE_QWEN3VL:
                    {
                        hparams.n_merge = 2; // default value for Qwen 2 and 2.5
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_WIN_ATTN_PATTERN, hparams.n_wa_pattern, model.proj_type == PROJECTOR_TYPE_QWEN25VL); // only 2.5 requires it
                        // ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
                        hparams.set_limit_image_tokens(8, 4096);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                        const int warn_min_pixels = 1024 * hparams.n_merge * hparams.n_merge * hparams.patch_size * hparams.patch_size;
                        if (hparams.image_min_pixels < warn_min_pixels) {
                            LOG_WRN("%s: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks\n", __func__);
                            LOG_WRN("%s: if you encounter problems with accuracy, try adding --image-min-tokens 1024\n", __func__);
                            LOG_WRN("%s: more info: https://github.com/ggml-org/llama.cpp/issues/16842\n\n", __func__);
                        }
                    } break;
                case PROJECTOR_TYPE_GLM4V:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.n_merge = 2; // default value for GLM4-V
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.set_limit_image_tokens(8, 4096);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_LLAMA4:
                    {
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        set_llava_uhd_res_candidates(model, 3);
                    } break;
                case PROJECTOR_TYPE_ULTRAVOX:
                case PROJECTOR_TYPE_QWEN2A:
                case PROJECTOR_TYPE_GLMA:
                case PROJECTOR_TYPE_VOXTRAL:
                    {
                        bool require_stack = model.proj_type == PROJECTOR_TYPE_ULTRAVOX ||
                                             model.proj_type == PROJECTOR_TYPE_VOXTRAL ||
                                             model.proj_type == PROJECTOR_TYPE_GLMA;
                        get_u32(KEY_A_PROJ_STACK_FACTOR, hparams.proj_stack_factor, require_stack);
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf"; // temporary solution for logging

                        // audio preprocessing params
                        hparams.audio_chunk_len    = 30; // in seconds
                        hparams.audio_sample_rate  = 16000;
                        hparams.audio_n_fft        = 400;
                        hparams.audio_window_len   = 400;
                        hparams.audio_hop_len      = 160;
                    } break;
                default:
                    break;
            }

            // sanity check
            {
                if (hparams.image_max_pixels < hparams.image_min_pixels) {
                    throw std::runtime_error(string_format("%s: image_max_pixels (%d) is less than image_min_pixels (%d)\n", __func__, hparams.image_max_pixels, hparams.image_min_pixels));
                }
            }

            LOG_INF("%s: projector:          %s\n", __func__, proj_type.c_str());
            LOG_INF("%s: n_embd:             %d\n", __func__, hparams.n_embd);
            LOG_INF("%s: n_head:             %d\n", __func__, hparams.n_head);
            LOG_INF("%s: n_ff:               %d\n", __func__, hparams.n_ff);
            LOG_INF("%s: n_layer:            %d\n", __func__, hparams.n_layer);
            LOG_INF("%s: ffn_op:             %s\n", __func__, log_ffn_op.c_str());
            LOG_INF("%s: projection_dim:     %d\n", __func__, hparams.projection_dim);
            if (is_vision) {
                LOG_INF("\n--- vision hparams ---\n");
                LOG_INF("%s: image_size:         %d\n", __func__, hparams.image_size);
                LOG_INF("%s: patch_size:         %d\n", __func__, hparams.patch_size);
                LOG_INF("%s: has_llava_proj:     %d\n", __func__, hparams.has_llava_projector);
                LOG_INF("%s: minicpmv_version:   %d\n", __func__, hparams.minicpmv_version);
                LOG_INF("%s: n_merge:            %d\n", __func__, hparams.n_merge);
                LOG_INF("%s: n_wa_pattern:       %d\n", __func__, hparams.n_wa_pattern);
                if (hparams.image_min_pixels > 0) {
                    LOG_INF("%s: image_min_pixels:   %d%s\n", __func__, hparams.image_min_pixels, hparams.custom_image_min_tokens > 0 ? " (custom value)" : "");
                }
                if (hparams.image_max_pixels > 0) {
                    LOG_INF("%s: image_max_pixels:   %d%s\n", __func__, hparams.image_max_pixels, hparams.custom_image_max_tokens > 0 ? " (custom value)" : "");
                }
            } else if (is_audio) {
                LOG_INF("\n--- audio hparams ---\n");
                LOG_INF("%s: n_mel_bins:         %d\n", __func__, hparams.n_mel_bins);
                LOG_INF("%s: proj_stack_factor:  %d\n", __func__, hparams.proj_stack_factor);
                LOG_INF("%s: audio_chunk_len:    %d\n", __func__, hparams.audio_chunk_len);
                LOG_INF("%s: audio_sample_rate:  %d\n", __func__, hparams.audio_sample_rate);
                LOG_INF("%s: audio_n_fft:        %d\n", __func__, hparams.audio_n_fft);
                LOG_INF("%s: audio_window_len:   %d\n", __func__, hparams.audio_window_len);
                LOG_INF("%s: audio_hop_len:      %d\n", __func__, hparams.audio_hop_len);
            }
            LOG_INF("\n");
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors(clip_ctx & ctx_clip) {
        auto & model = ctx_clip.model;
        auto & hparams = model.hparams;
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;

        // TODO @ngxson : support both audio and video in the future
        const char * prefix = model.modality == CLIP_MODALITY_AUDIO ? "a" : "v";

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        struct ggml_init_params params = {
            /*.mem_size =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        ctx_clip.ctx_data.reset(ggml_init(params));
        if (!ctx_clip.ctx_data) {
            throw std::runtime_error(string_format("%s: failed to init ggml context\n", __func__));
        }

        // helper function
        auto get_tensor = [&](const std::string & name, bool required = true) {
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                // add tensors to context
                ggml_tensor * data_tensor = ggml_dup_tensor(ctx_clip.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                cur = data_tensor;
            }
            return cur;
        };

        model.class_embedding = get_tensor(TN_CLASS_EMBD, false);

        model.pre_ln_w = get_tensor(string_format(TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = get_tensor(string_format(TN_LN_PRE, prefix, "bias"),   false);

        model.post_ln_w = get_tensor(string_format(TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = get_tensor(string_format(TN_LN_POST, prefix, "bias"),   false);

        model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = get_tensor(TN_PATCH_EMBD,   false);
        model.patch_embeddings_1 = get_tensor(TN_PATCH_EMBD_1, false);

        model.norm_embd_w = get_tensor(string_format(TN_NORM_EMBD, "weight"), false);
        model.norm_embd_b = get_tensor(string_format(TN_NORM_EMBD, "bias"),   false);

        model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, prefix), false);

        // layers
        model.layers.resize(hparams.n_layer);
        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = model.layers[il];
            layer.k_w    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "weight"), false);
            layer.q_w    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "weight"), false);
            layer.v_w    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "weight"), false);
            layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "weight"));
            layer.qkv_w  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "weight"), false);
            layer.k_norm = get_tensor(string_format(TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = get_tensor(string_format(TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = get_tensor(string_format(TN_LN_1,        prefix, il, "weight"), false);
            layer.ln_2_w = get_tensor(string_format(TN_LN_2,        prefix, il, "weight"), false);
            layer.ls_1_w = get_tensor(string_format(TN_LS_1,        prefix, il, "weight"), false); // no bias
            layer.ls_2_w = get_tensor(string_format(TN_LS_2,        prefix, il, "weight"), false); // no bias

            layer.k_b    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "bias"), false);
            layer.q_b    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "bias"), false);
            layer.v_b    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "bias"), false);
            layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.qkv_b  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "bias"), false);
            layer.ln_1_b = get_tensor(string_format(TN_LN_1,        prefix, il, "bias"), false);
            layer.ln_2_b = get_tensor(string_format(TN_LN_2,        prefix, il, "bias"), false);

            // ffn
            layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "weight"));
            layer.ff_up_b   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "bias"),   false);
            layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = get_tensor(string_format(TN_FFN_GATE, prefix, il, "bias"),   false);
            layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "weight"));
            layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "bias"),   false);


            // qwen3vl deepstack layer
            layer.deepstack_norm_w = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "weight"), false);
            layer.deepstack_norm_b = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "bias"), false);
            layer.deepstack_fc1_w  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "weight"), false);
            layer.deepstack_fc1_b  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "bias"), false);
            layer.deepstack_fc2_w  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "weight"), false);
            layer.deepstack_fc2_b  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "bias"), false);
            if (layer.has_deepstack()) {
                model.n_deepstack_layers++;
            }

            // some models already exported with legacy (incorrect) naming which is quite messy, let's fix it here
            // note: Qwen model converted from the old surgery script has n_ff = 0, so we cannot use n_ff to check!
            bool is_ffn_swapped = (
                    // only old models need this fix
                    model.proj_type == PROJECTOR_TYPE_MLP
                    || model.proj_type == PROJECTOR_TYPE_MLP_NORM
                    || model.proj_type == PROJECTOR_TYPE_LDP
                    || model.proj_type == PROJECTOR_TYPE_LDPV2
                    || model.proj_type == PROJECTOR_TYPE_QWEN2VL
                    || model.proj_type == PROJECTOR_TYPE_QWEN25VL
                    || model.proj_type == PROJECTOR_TYPE_GLM_EDGE
                    || model.proj_type == PROJECTOR_TYPE_GEMMA3
                    || model.proj_type == PROJECTOR_TYPE_IDEFICS3
                    || model.proj_type == PROJECTOR_TYPE_MINICPMV
                ) && layer.ff_up_w && layer.ff_down_w && layer.ff_down_w->ne[0] == hparams.n_embd;
            if (is_ffn_swapped) {
                // swap up and down weights
                ggml_tensor * tmp = layer.ff_up_w;
                layer.ff_up_w = layer.ff_down_w;
                layer.ff_down_w = tmp;
                // swap up and down biases
                tmp = layer.ff_up_b;
                layer.ff_up_b = layer.ff_down_b;
                layer.ff_down_b = tmp;
                if (il == 0) {
                    LOG_WRN("%s: ffn up/down are swapped\n", __func__);
                }
            }
        }

        switch (model.proj_type) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
                {
                    // LLaVA projection
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"), false);
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    // Yi-type llava
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"), false);
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    // missing in Yi-type llava
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"), false);
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // Yi-type llava
                    model.mm_3_w = get_tensor(string_format(TN_LLAVA_PROJ, 3, "weight"), false);
                    model.mm_3_b = get_tensor(string_format(TN_LLAVA_PROJ, 3, "bias"), false);
                    model.mm_4_w = get_tensor(string_format(TN_LLAVA_PROJ, 4, "weight"), false);
                    model.mm_4_b = get_tensor(string_format(TN_LLAVA_PROJ, 4, "bias"), false);
                    if (model.mm_3_w) {
                        // TODO: this is a hack to support Yi-type llava
                        model.proj_type = PROJECTOR_TYPE_MLP_NORM;
                    }
                    model.image_newline = get_tensor(TN_IMAGE_NEWLINE, false);
                } break;
            case PROJECTOR_TYPE_LDP:
                {
                    // MobileVLM projection
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_model_mlp_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                    model.mm_model_block_1_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "0.weight"));
                    model.mm_model_block_1_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.weight"));
                    model.mm_model_block_1_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.bias"));
                    model.mm_model_block_1_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.weight"));
                    model.mm_model_block_1_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.bias"));
                    model.mm_model_block_1_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.weight"));
                    model.mm_model_block_1_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.bias"));
                    model.mm_model_block_1_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "0.weight"));
                    model.mm_model_block_1_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.weight"));
                    model.mm_model_block_1_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.bias"));
                    model.mm_model_block_2_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "0.weight"));
                    model.mm_model_block_2_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.weight"));
                    model.mm_model_block_2_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.bias"));
                    model.mm_model_block_2_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.weight"));
                    model.mm_model_block_2_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.bias"));
                    model.mm_model_block_2_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.weight"));
                    model.mm_model_block_2_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.bias"));
                    model.mm_model_block_2_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "0.weight"));
                    model.mm_model_block_2_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.weight"));
                    model.mm_model_block_2_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.bias"));
                } break;
            case PROJECTOR_TYPE_LDPV2:
                {
                    // MobilVLM_V2 projection
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_model_mlp_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                    model.mm_model_mlp_2_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "bias"));
                    model.mm_model_peg_0_w = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "weight"));
                    model.mm_model_peg_0_b = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "bias"));
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    // model.mm_model_pos_embed = get_tensor(new_clip->ctx_data, TN_MINICPMV_POS_EMBD);
                    model.mm_model_pos_embed_k = get_tensor(TN_MINICPMV_POS_EMBD_K);
                    model.mm_model_query = get_tensor(TN_MINICPMV_QUERY);
                    model.mm_model_proj = get_tensor(TN_MINICPMV_PROJ);
                    model.mm_model_kv_proj = get_tensor(TN_MINICPMV_KV_PROJ);
                    model.mm_model_attn_q_w = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "weight"));
                    model.mm_model_attn_k_w = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "weight"));
                    model.mm_model_attn_v_w = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "weight"));
                    model.mm_model_attn_q_b = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "bias"));
                    model.mm_model_attn_k_b = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "bias"));
                    model.mm_model_attn_v_b = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "bias"));
                    model.mm_model_attn_o_w = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "weight"));
                    model.mm_model_attn_o_b = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "bias"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_MINICPMV_LN, "q", "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_MINICPMV_LN, "q", "bias"));
                    model.mm_model_ln_kv_w = get_tensor(string_format(TN_MINICPMV_LN, "kv", "weight"));
                    model.mm_model_ln_kv_b = get_tensor(string_format(TN_MINICPMV_LN, "kv", "bias"));
                    model.mm_model_ln_post_w = get_tensor(string_format(TN_MINICPMV_LN, "post", "weight"));
                    model.mm_model_ln_post_b = get_tensor(string_format(TN_MINICPMV_LN, "post", "bias"));
                } break;
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    model.mm_model_adapter_conv_w = get_tensor(string_format(TN_GLM_ADAPER_CONV, "weight"));
                    model.mm_model_adapter_conv_b = get_tensor(string_format(TN_GLM_ADAPER_CONV, "bias"));
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_GLM_ADAPTER_LINEAR, "weight"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "bias"));
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_GLM_ADAPTER_D_H_2_4H, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_GLM_ADAPTER_GATE, "weight"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_GLM_ADAPTER_D_4H_2_H, "weight"));
                    model.mm_boi = get_tensor(string_format(TN_TOK_GLM_BOI, "weight"));
                    model.mm_eoi = get_tensor(string_format(TN_TOK_GLM_EOI, "weight"));
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN3VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_GLM4V:
                {
                    model.projection     = get_tensor(TN_MM_PROJECTOR);
                    model.mm_ffn_up_w    = get_tensor(string_format(TN_MM_UP,        "weight"));
                    model.mm_ffn_up_b    = get_tensor(string_format(TN_MM_UP,        "bias"), false);
                    model.mm_ffn_gate_w  = get_tensor(string_format(TN_MM_GATE,      "weight"));
                    model.mm_ffn_gate_b  = get_tensor(string_format(TN_MM_GATE,      "bias"), false);
                    model.mm_ffn_down_w  = get_tensor(string_format(TN_MM_DOWN,      "weight"));
                    model.mm_ffn_down_b  = get_tensor(string_format(TN_MM_DOWN,      "bias"), false);
                    model.mm_post_norm_w = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                    model.mm_post_norm_b = get_tensor(string_format(TN_MM_POST_NORM, "bias"), false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"));
                    model.mm_patch_merger_b = get_tensor(string_format(TN_MM_PATCH_MERGER, "bias"));
                } break;
            case PROJECTOR_TYPE_GEMMA3:
                {
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.mm_soft_emb_norm_w = get_tensor(TN_MM_SOFT_EMB_N);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    model.projection = get_tensor(TN_MM_PROJECTOR);
                } break;
            case PROJECTOR_TYPE_LFM2:
            case PROJECTOR_TYPE_KIMIVL:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // [IMG_BREAK] token embedding
                    model.token_embd_img_break = get_tensor(TN_TOK_IMG_BREAK);
                    // for mistral small 3.1
                    model.mm_input_norm_w   = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"), false);
                } break;
            case PROJECTOR_TYPE_LIGHTONOCR:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    model.mm_input_norm_w   = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"), false);
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_mid_w = get_tensor(string_format(TN_MM_NORM_MID, "weight"));
                } break;
            case PROJECTOR_TYPE_QWEN2A:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_fc_w = get_tensor(string_format(TN_MM_AUDIO_FC, "weight"));
                    model.mm_fc_b = get_tensor(string_format(TN_MM_AUDIO_FC, "bias"));
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                } break;
            case PROJECTOR_TYPE_GLMA:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_pre_b = get_tensor(string_format(TN_MM_NORM_PRE, "bias"));
                    model.mm_boi = get_tensor(string_format(TN_TOK_BOI, "weight"));
                    model.mm_eoi = get_tensor(string_format(TN_TOK_EOI, "weight"));
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    model.mm_model_proj    = get_tensor(TN_MM_PROJECTOR);
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_COGVLM:
                {
                    model.mm_model_proj     = get_tensor(TN_MM_PROJECTOR);
                    model.mm_post_fc_norm_w = get_tensor(string_format(TN_MM_POST_FC_NORM, "weight"));
                    model.mm_post_fc_norm_b = get_tensor(string_format(TN_MM_POST_FC_NORM, "bias"));
                    model.mm_h_to_4h_w      = get_tensor(string_format(TN_MM_H_TO_4H,      "weight"));
                    model.mm_gate_w         = get_tensor(string_format(TN_MM_GATE,         "weight"));
                    model.mm_4h_to_h_w      = get_tensor(string_format(TN_MM_4H_TO_H,      "weight"));
                    model.mm_boi            = get_tensor(TN_TOK_BOI);
                    model.mm_eoi            = get_tensor(TN_TOK_EOI);
                } break;
            case PROJECTOR_TYPE_JANUS_PRO:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
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
                ggml_tensor * cur = ggml_get_tensor(ctx_clip.ctx_data.get(), t->name);
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

    struct support_info_op {
        ggml_tensor * op;

        // true if the op runs on the accelerated ctx_clip.backend
        bool is_accel = true;
    };

    struct support_info_graph {
        // whether the clip_ctx.backend supports flash attention
        bool fattn = true;
        ggml_tensor * fattn_op = nullptr; // for debugging

        std::vector<support_info_op> ops;
    };

    static void warmup(clip_ctx & ctx_clip) {
        // create a fake batch
        const auto & hparams = ctx_clip.model.hparams;
        clip_image_f32_batch batch;
        clip_image_f32_ptr img(clip_image_f32_init());
        if (ctx_clip.model.modality == CLIP_MODALITY_VISION) {
            img->nx = hparams.warmup_image_size;
            img->ny = hparams.warmup_image_size;
            LOG_INF("%s: warmup with image size = %d x %d\n", __func__, img->nx, img->ny);
        } else {
            img->nx = hparams.warmup_audio_size;
            img->ny = hparams.n_mel_bins;
            LOG_INF("%s: warmup with audio size = %d\n", __func__, img->nx);
        }
        batch.entries.push_back(std::move(img));
        warmup(ctx_clip, batch);
    }

    static void warmup(clip_ctx & ctx_clip, const clip_image_f32_batch & batch) {
        support_info_graph info;

        if (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_AUTO) {
            // try to enable flash attention to see if it's supported
            ctx_clip.flash_attn_type = CLIP_FLASH_ATTN_TYPE_ENABLED;
            info = alloc_compute_meta(ctx_clip, batch);
            if (!info.fattn && info.fattn_op) {
                auto op = info.fattn_op;
                LOG_WRN("%s: *****************************************************************\n", __func__);
                LOG_WRN("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", __func__, ggml_backend_name(ctx_clip.backend));
                LOG_WRN("%s: op params: \n", __func__);
                static auto print_shape = [](const char * fn, const char * name, ggml_tensor * t) {
                    LOG_WRN("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", fn,
                            name, ggml_type_name(t->type),
                            t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                            t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
                };
                print_shape(__func__, " dst", op);
                print_shape(__func__, "src0", op->src[0]);
                print_shape(__func__, "src1", op->src[1]);
                print_shape(__func__, "src2", op->src[2]);
                LOG_WRN("%s: please report this on github as an issue\n", __func__);
                LOG_WRN("%s: *****************************************************************\n", __func__);
                ctx_clip.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
                alloc_compute_meta(ctx_clip, batch);
            }
        } else {
            info = alloc_compute_meta(ctx_clip, batch);
            if (!info.fattn && ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
                LOG_WRN("%s: flash attention is not supported by the current backend; falling back to CPU (performance will be degraded)\n", __func__);
            }
        }

        ctx_clip.is_allocated = true; // mark buffers as allocated

        LOG_INF("%s: flash attention is %s\n", __func__,
            (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");

        // print ops that are not supported by the GPU backend (if there is one)
        if (ctx_clip.backend && ctx_clip.backend != ctx_clip.backend_cpu) {
            std::vector<support_info_op> unsupported_ops;
            for (const auto & op : info.ops) {
                if (!op.is_accel) {
                    unsupported_ops.push_back(op);
                }
            }
            if (!unsupported_ops.empty()) {
                LOG_WRN("%s: *****************************************************************\n", __func__);
                LOG_WRN("%s: WARNING: the CLIP graph uses unsupported operators by the backend\n", __func__);
                LOG_WRN("%s:          the performance will be suboptimal                      \n", __func__);
                LOG_WRN("%s:          list of unsupported ops (backend=%s):\n", __func__, ggml_backend_name(ctx_clip.backend));
                for (const auto & op : unsupported_ops) {
                    LOG_WRN("%s: %16s: type = %s, ne = [%d %d %d %d]\n", __func__,
                            ggml_op_name(op.op->op),
                            ggml_type_name(op.op->type),
                            op.op->ne[0], op.op->ne[1], op.op->ne[2], op.op->ne[3]);
                }
                LOG_WRN("%s: flash attention is %s\n", __func__,
                    (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");
                LOG_WRN("%s: please report this on github as an issue\n", __func__);
                LOG_WRN("%s: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118\n", __func__);
                LOG_WRN("%s: *****************************************************************\n", __func__);
            }
        }
    }

    static support_info_graph alloc_compute_meta(clip_ctx & ctx_clip, const clip_image_f32_batch & batch) {
        ctx_clip.buf_compute_meta.resize(ctx_clip.max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());

        ggml_cgraph * gf = clip_image_build_graph(&ctx_clip, batch);
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

        const int n_splits = ggml_backend_sched_get_n_splits(ctx_clip.sched.get());
        const int n_nodes  = ggml_graph_n_nodes(gf);

        LOG_INF("%s: graph splits = %d, nodes = %d\n", __func__,  n_splits, n_nodes);

        support_info_graph res {
            /*.fattn    = */ true,
            /*.fattn_op = */ nullptr,
            /*.ops      = */ {},
        };

        // check op support
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            res.ops.push_back({node, true});
            if (!ggml_backend_supports_op(ctx_clip.backend, node)) {
                res.ops.back().is_accel = false;
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    res.fattn    = false;
                    res.fattn_op = node;
                }
            }
        }

        return res;
    }

    void get_bool(const std::string & key, bool & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_i32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_i32(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_u32(ctx_gguf.get(), i);
    }

    void get_f32(const std::string & key, float & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_string(const std::string & key, std::string & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }

    void get_arr_int(const std::string & key, std::vector<int> & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        int n = gguf_get_arr_n(ctx_gguf.get(), i);
        output.resize(n);
        const int32_t * values = (const int32_t *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (int i = 0; i < n; ++i) {
            output[i] = values[i];
        }
    }

    static void set_llava_uhd_res_candidates(clip_model & model, const int max_patches_per_side) {
        auto & hparams = model.hparams;
        for (int x = 1; x <= max_patches_per_side; x++) {
            for (int y = 1; y <= max_patches_per_side; y++) {
                if (x == 1 && y == 1) {
                    continue; // skip the first point
                }
                hparams.image_res_candidates.push_back(clip_image_size{
                    x*hparams.image_size,
                    y*hparams.image_size,
                });
            }
        }
    }
};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params) {
    clip_ctx * ctx_vision = nullptr;
    clip_ctx * ctx_audio = nullptr;

    try {
        clip_model_loader loader(fname);

        if (loader.has_vision) {
            ctx_vision = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_vision->model, CLIP_MODALITY_VISION);
            loader.load_tensors(*ctx_vision);
            if (ctx_params.warmup) {
                loader.warmup(*ctx_vision);
            }

            // clip_debug_encode(ctx_vision, 24*14, 24*14, 0.5f);
        }

        if (loader.has_audio) {
            ctx_audio = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_audio->model, CLIP_MODALITY_AUDIO);
            loader.load_tensors(*ctx_audio);
            if (ctx_params.warmup) {
                loader.warmup(*ctx_audio);
            }
        }

    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load model '%s': %s\n", __func__, fname, e.what());

        delete ctx_vision;
        delete ctx_audio;

        return {nullptr, nullptr};
    }

    return {ctx_vision, ctx_audio};
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
void clip_image_u8_free(struct clip_image_u8  * img) { delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { delete img; }
void clip_image_u8_batch_free(struct clip_image_u8_batch * batch) { delete batch; }
void clip_image_f32_batch_free(struct clip_image_f32_batch * batch) { delete batch; }

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

// set of tools to manupulate images
// in the future, we can have HW acceleration by allowing this struct to access 3rd party lib like imagick or opencv
struct img_tool {
    enum resize_algo {
        RESIZE_ALGO_BILINEAR,
        RESIZE_ALGO_BICUBIC,
        // RESIZE_ALGO_LANCZOS, // TODO
    };

    static void resize(
            const clip_image_u8 & src,
            clip_image_u8 & dst,
            const clip_image_size & target_resolution,
            resize_algo algo,
            bool add_padding = true, // TODO: define the behavior for add_padding = false
            std::array<uint8_t, 3> pad_color = {0, 0, 0}) {
        dst.nx = target_resolution.width;
        dst.ny = target_resolution.height;
        dst.buf.resize(3 * dst.nx * dst.ny);

        if (dst.nx == src.nx && dst.ny == src.ny) {
            // no resize needed, simple copy
            dst.buf = src.buf;
            return;
        }

        if (!add_padding) {
            // direct resize
            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, dst, target_resolution.width, target_resolution.height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }
        } else {
            // resize with padding
            clip_image_u8 resized_image;
            float scale_w = static_cast<float>(target_resolution.width) / src.nx;
            float scale_h = static_cast<float>(target_resolution.height) / src.ny;
            float scale = std::min(scale_w, scale_h);
            int new_width  = std::min(static_cast<int>(std::ceil(src.nx * scale)), target_resolution.width);
            int new_height = std::min(static_cast<int>(std::ceil(src.ny * scale)), target_resolution.height);

            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, resized_image, new_width, new_height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }

            // fill dst with pad_color
            fill(dst, pad_color);

            int offset_x = (target_resolution.width  - new_width)  / 2;
            int offset_y = (target_resolution.height - new_height) / 2;

            composite(dst, resized_image, offset_x, offset_y);
        }
    }

    static void crop(const clip_image_u8 & image, clip_image_u8 & dst, int x, int y, int w, int h) {
        dst.nx = w;
        dst.ny = h;
        dst.buf.resize(3 * w * h);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int src_idx = 3 * ((y + i)*image.nx + (x + j));
                int dst_idx = 3 * (i*w + j);
                dst.buf[dst_idx]     = image.buf[src_idx];
                dst.buf[dst_idx + 1] = image.buf[src_idx + 1];
                dst.buf[dst_idx + 2] = image.buf[src_idx + 2];
            }
        }
    }

    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will be aligned to the nearest multiple of align_size
    // if H or W size is larger than longest_edge, it will be resized to longest_edge
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const int align_size, const int longest_edge) {
        GGML_ASSERT(align_size > 0);
        if (inp_size.width <= 0 || inp_size.height <= 0 || longest_edge <= 0) {
            return {0, 0};
        }

        float scale = std::min(static_cast<float>(longest_edge) / inp_size.width,
                               static_cast<float>(longest_edge) / inp_size.height);

        float target_width_f  = static_cast<float>(inp_size.width)  * scale;
        float target_height_f = static_cast<float>(inp_size.height) * scale;

        auto ceil_by_factor = [f = align_size](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
        int aligned_width  = ceil_by_factor(target_width_f);
        int aligned_height = ceil_by_factor(target_height_f);

        return {aligned_width, aligned_height};
    }

    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will have min_pixels <= W*H <= max_pixels
    // this is referred as "smart_resize" in transformers code
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const int align_size, const int min_pixels, const int max_pixels) {
        GGML_ASSERT(align_size > 0);
        const int width  = inp_size.width;
        const int height = inp_size.height;

        auto round_by_factor = [f = align_size](float x) { return static_cast<int>(std::round(x / static_cast<float>(f))) * f; };
        auto ceil_by_factor  = [f = align_size](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
        auto floor_by_factor = [f = align_size](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };

        // always align up first
        int h_bar = std::max(align_size, round_by_factor(height));
        int w_bar = std::max(align_size, round_by_factor(width));

        if (h_bar * w_bar > max_pixels) {
            const auto beta = std::sqrt(static_cast<float>(height * width) / max_pixels);
            h_bar = std::max(align_size, floor_by_factor(height / beta));
            w_bar = std::max(align_size, floor_by_factor(width  / beta));
        } else if (h_bar * w_bar < min_pixels) {
            const auto beta = std::sqrt(static_cast<float>(min_pixels) / (height * width));
            h_bar = ceil_by_factor(height * beta);
            w_bar = ceil_by_factor(width * beta);
        }

        return {w_bar, h_bar};
    }

    // draw src image into dst image at offset (offset_x, offset_y)
    static void composite(clip_image_u8 & dst, const clip_image_u8 & src, int offset_x, int offset_y) {
        for (int y = 0; y < src.ny; ++y) {
            for (int x = 0; x < src.nx; ++x) {
                int dx = x + offset_x;
                int dy = y + offset_y;
                // skip pixels that would be out of bounds in the destination
                if (dx < 0 || dy < 0 || dx >= dst.nx || dy >= dst.ny) {
                    continue;
                }
                size_t dst_idx = 3 * (static_cast<size_t>(dy) * dst.nx + static_cast<size_t>(dx));
                size_t src_idx = 3 * (static_cast<size_t>(y) * src.nx + static_cast<size_t>(x));
                dst.buf[dst_idx + 0] = src.buf[src_idx + 0];
                dst.buf[dst_idx + 1] = src.buf[src_idx + 1];
                dst.buf[dst_idx + 2] = src.buf[src_idx + 2];
            }
        }
    }

    // fill the image with a solid color
    static void fill(clip_image_u8 & img, const std::array<uint8_t, 3> & color) {
        for (size_t i = 0; i < img.buf.size(); i += 3) {
            img.buf[i]     = color[0];
            img.buf[i + 1] = color[1];
            img.buf[i + 2] = color[2];
        }
    }

private:
    // Bilinear resize function
    static void resize_bilinear(const clip_image_u8 & src, clip_image_u8 & dst, int target_width, int target_height) {
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
                    float top = lerp(
                        static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                        static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]),
                        x_lerp
                    );
                    float bottom = lerp(
                        static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                        static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]),
                        x_lerp
                    );
                    dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(lerp(top, bottom, y_lerp));
                }
            }
        }
    }

    // Bicubic resize function
    // part of image will be cropped if the aspect ratio is different
    static bool resize_bicubic(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        const int nx = img.nx;
        const int ny = img.ny;

        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float Cc;
        float C[5] = {};
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

    static inline int clip(int x, int lower, int upper) {
        return std::max(lower, std::min(x, upper));
    }

    // Linear interpolation between two points
    static inline float lerp(float s, float e, float t) {
        return s + (e - s) * t;
    }
};

/**
 * implementation of LLaVA-UHD:
 *  - https://arxiv.org/pdf/2403.11703
 *  - https://github.com/thunlp/LLaVA-UHD
 *  - https://github.com/thunlp/LLaVA-UHD/blob/302301bc2175f7e717fb8548516188e89f649753/llava_uhd/train/llava-uhd/slice_logic.py#L118
 *
 * overview:
 *   - an image always have a single overview (downscaled image)
 *   - an image can have 0 or multiple slices, depending on the image size
 *   - each slice can then be considered as a separate image
 *
 * for example:
 *
 * [overview] --> [slice 1] --> [slice 2]
 *           |                |
 *           +--> [slice 3] --> [slice 4]
 */
struct llava_uhd {
    struct slice_coordinates {
        int x;
        int y;
        clip_image_size size;
    };

    struct slice_instructions {
        clip_image_size overview_size; // size of downscaled image
        clip_image_size refined_size;  // size of image right before slicing (must be multiple of slice size)
        clip_image_size grid_size;     // grid_size.width * grid_size.height = number of slices
        std::vector<slice_coordinates> slices;

        img_tool::resize_algo interpolation_overview = img_tool::RESIZE_ALGO_BILINEAR;
        bool padding_overview = false;  // if true, refine image will be padded to the grid size (e.g. llava-1.6)
        std::array<uint8_t, 3> pad_color_overview = {0, 0, 0};

        img_tool::resize_algo interpolation_refined = img_tool::RESIZE_ALGO_BICUBIC;
        bool padding_refined = false;  // if true, refine image will be padded to the grid size (e.g. llava-1.6)
        std::array<uint8_t, 3> pad_color_refined = {0, 0, 0};
    };

    static slice_instructions get_slice_instructions(struct clip_ctx * ctx, const clip_image_size & original_size) {
        slice_instructions res;
        const int patch_size      = clip_get_patch_size(ctx);
        const int slice_size      = clip_get_image_size(ctx);
        const int original_width  = original_size.width;
        const int original_height = original_size.height;

        const bool has_slices    = original_size.width > slice_size || original_size.height > slice_size;
        const bool has_pinpoints = !ctx->model.hparams.image_res_candidates.empty();

        if (!has_slices) {
            // skip slicing logic
            res.overview_size = clip_image_size{slice_size, slice_size};
            res.refined_size  = clip_image_size{0, 0};
            res.grid_size     = clip_image_size{0, 0};

            return res;
        }

        if (has_pinpoints) {
            // has pinpoints, use them to calculate the grid size (e.g. llava-1.6)
            auto refine_size = llava_uhd::select_best_resolution(
                original_size,
                ctx->model.hparams.image_res_candidates);
            res.overview_size         = clip_image_size{slice_size, slice_size};
            res.refined_size          = refine_size;
            res.grid_size             = clip_image_size{0, 0};
            res.padding_refined       = true;
            res.interpolation_refined = img_tool::RESIZE_ALGO_BILINEAR;  // preserve old behavior when padding

            LOG_DBG("%s: using pinpoints for slicing\n", __func__);
            LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d\n",
                    __func__, original_width, original_height,
                    res.overview_size.width, res.overview_size.height,
                    res.refined_size.width,  res.refined_size.height);

            for (int y = 0; y < refine_size.height; y += slice_size) {
                for (int x = 0; x < refine_size.width; x += slice_size) {
                    slice_coordinates slice;
                    slice.x = x;
                    slice.y = y;
                    slice.size.width  = std::min(slice_size, refine_size.width  - x);
                    slice.size.height = std::min(slice_size, refine_size.height - y);
                    res.slices.push_back(slice);
                    LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                            __func__, (int)res.slices.size() - 1,
                            slice.x, slice.y, slice.size.width, slice.size.height);
                }
            }

            res.grid_size.height = refine_size.height / slice_size;
            res.grid_size.width  = refine_size.width  / slice_size;
            LOG_DBG("%s: grid size: %d x %d\n", __func__, res.grid_size.width, res.grid_size.height);

            return res;
        }

        // no pinpoints, dynamically calculate the grid size (e.g. minicpmv)

        auto best_size    = get_best_resize(original_size, slice_size, patch_size, !has_slices);
        res.overview_size = best_size;

        {
            const int max_slice_nums = 9; // TODO: this is only used by minicpmv, maybe remove it
            const float log_ratio = log((float)original_width / original_height);
            const float ratio = (float)original_width * original_height / (slice_size * slice_size);
            const int multiple = fmin(ceil(ratio), max_slice_nums);

            auto best_grid   = get_best_grid(max_slice_nums, multiple, log_ratio);
            auto refine_size = get_refine_size(original_size, best_grid, slice_size, patch_size, true);
            res.grid_size    = best_grid;
            res.refined_size = refine_size;

            LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n",
                    __func__, original_width, original_height,
                    res.overview_size.width, res.overview_size.height,
                    res.refined_size.width, res.refined_size.height,
                    res.grid_size.width, res.grid_size.height);

            int width  = refine_size.width;
            int height = refine_size.height;
            int grid_x = int(width  / best_grid.width);
            int grid_y = int(height / best_grid.height);
            for (int patches_y = 0,                    ic = 0;
                    patches_y < refine_size.height && ic < best_grid.height;
                    patches_y += grid_y,              ic += 1) {
                for (int patches_x = 0,                   jc = 0;
                        patches_x < refine_size.width && jc < best_grid.width;
                        patches_x += grid_x,             jc += 1) {
                    slice_coordinates slice;
                    slice.x = patches_x;
                    slice.y = patches_y;
                    slice.size.width  = grid_x;
                    slice.size.height = grid_y;
                    res.slices.push_back(slice);
                    LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                            __func__, (int)res.slices.size() - 1,
                            slice.x, slice.y, slice.size.width, slice.size.height);
                }
            }
        }

        return res;
    }

    static std::vector<clip_image_u8_ptr> slice_image(const clip_image_u8 * img, const slice_instructions & inst) {
        std::vector<clip_image_u8_ptr> output;

        // resize to overview size
        clip_image_u8_ptr resized_img(clip_image_u8_init());
        img_tool::resize(*img, *resized_img, inst.overview_size, inst.interpolation_overview,
                         inst.padding_overview, inst.pad_color_overview);
        output.push_back(std::move(resized_img));

        if (inst.slices.empty()) {
            // no slices, just return the resized image
            return output;
        }

        // resize to refined size
        clip_image_u8_ptr refined_img(clip_image_u8_init());
        img_tool::resize(*img, *refined_img, inst.refined_size, inst.interpolation_refined,
                         inst.padding_refined, inst.pad_color_refined);

        // create slices
        for (const auto & slice : inst.slices) {
            int x = slice.x;
            int y = slice.y;
            int w = slice.size.width;
            int h = slice.size.height;

            clip_image_u8_ptr img_slice(clip_image_u8_init());
            img_tool::crop(*refined_img, *img_slice, x, y, w, h);
            output.push_back(std::move(img_slice));
        }

        return output;
    }

private:
    static clip_image_size get_best_resize(const clip_image_size & original_size, int scale_resolution, int patch_size, bool allow_upscale = false) {
        int width  = original_size.width;
        int height = original_size.height;
        if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
            float r = static_cast<float>(width) / height;
            height  = static_cast<int>(scale_resolution / std::sqrt(r));
            width   = static_cast<int>(height * r);
        }
        clip_image_size res;
        res.width  = ensure_divide(width,  patch_size);
        res.height = ensure_divide(height, patch_size);
        return res;
    }

    static clip_image_size resize_maintain_aspect_ratio(const clip_image_size & orig, const clip_image_size & target_max) {
        float scale_width  = static_cast<float>(target_max.width)  / orig.width;
        float scale_height = static_cast<float>(target_max.height) / orig.height;
        float scale = std::min(scale_width, scale_height);
        return clip_image_size{
            static_cast<int>(orig.width  * scale),
            static_cast<int>(orig.height * scale),
        };
    }

    /**
     * Selects the best resolution from a list of possible resolutions based on the original size.
     *
     * For example, when given a list of resolutions:
     *  - 100x100
     *  - 200x100
     *  - 100x200
     *  - 200x200
     *
     * And an input image of size 111x200, then 100x200 is the best fit (least wasted resolution).
     *
     * @param original_size The original size of the image
     * @param possible_resolutions A list of possible resolutions
     * @return The best fit resolution
     */
    static clip_image_size select_best_resolution(const clip_image_size & original_size, const std::vector<clip_image_size> & possible_resolutions) {
        clip_image_size best_fit;
        int min_wasted_area = std::numeric_limits<int>::max();
        int max_effective_resolution = 0;

        for (const clip_image_size & candidate : possible_resolutions) {
            auto target_size = resize_maintain_aspect_ratio(original_size, candidate);
            int effective_resolution = std::min(
                target_size.width * target_size.height,
                original_size.width * original_size.height);
            int wasted_area = (candidate.width * candidate.height) - effective_resolution;

            if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_area < min_wasted_area)) {
                max_effective_resolution = effective_resolution;
                min_wasted_area = wasted_area;
                best_fit = candidate;
            }

            LOG_DBG("%s: candidate: %d x %d, target: %d x %d, wasted: %d, effective: %d\n", __func__, candidate.width, candidate.height, target_size.width, target_size.height, wasted_area, effective_resolution);
        }

        return best_fit;
    }

    static int ensure_divide(int length, int patch_size) {
        return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
    }

    static clip_image_size get_refine_size(const clip_image_size & original_size, const clip_image_size & grid, int scale_resolution, int patch_size, bool allow_upscale = false) {
        int width  = original_size.width;
        int height = original_size.height;
        int grid_x = grid.width;
        int grid_y = grid.height;

        int refine_width  = ensure_divide(width, grid_x);
        int refine_height = ensure_divide(height, grid_y);

        clip_image_size grid_size;
        grid_size.width  = refine_width  / grid_x;
        grid_size.height = refine_height / grid_y;

        auto best_grid_size  = get_best_resize(grid_size, scale_resolution, patch_size, allow_upscale);
        int best_grid_width  = best_grid_size.width;
        int best_grid_height = best_grid_size.height;

        clip_image_size refine_size;
        refine_size.width  = best_grid_width  * grid_x;
        refine_size.height = best_grid_height * grid_y;
        return refine_size;
    }

    static clip_image_size get_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
        std::vector<int> candidate_split_grids_nums;
        for (int i : {multiple - 1, multiple, multiple + 1}) {
            if (i == 1 || i > max_slice_nums) {
                continue;
            }
            candidate_split_grids_nums.push_back(i);
        }

        std::vector<clip_image_size> candidate_grids;
        for (int split_grids_nums : candidate_split_grids_nums) {
            int m = 1;
            while (m <= split_grids_nums) {
                if (split_grids_nums % m == 0) {
                    candidate_grids.push_back(clip_image_size{m, split_grids_nums / m});
                }
                ++m;
            }
        }

        clip_image_size best_grid{1, 1};
        float min_error = std::numeric_limits<float>::infinity();
        for (const auto& grid : candidate_grids) {
            float error = std::abs(log_ratio - std::log(1.0 * grid.width / grid.height));
            if (error < min_error) {
                best_grid = grid;
                min_error = error;
            }
        }
        return best_grid;
    }
};

// returns the normalized float tensor for llava-1.5, for spatial_unpad with anyres processing for llava-1.6 it returns the normalized image patch tensors as a vector
// res_imgs memory is being allocated here, previous allocations will be freed if found
bool clip_image_preprocess(struct clip_ctx * ctx, const clip_image_u8 * img, struct clip_image_f32_batch * res_imgs) {
    clip_image_size original_size{img->nx, img->ny};
    auto & params = ctx->model.hparams;

    switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_MINICPMV:
            {
                auto const inst = llava_uhd::get_slice_instructions(ctx, original_size);
                std::vector<clip_image_u8_ptr> imgs = llava_uhd::slice_image(img, inst);

                for (size_t i = 0; i < imgs.size(); ++i) {
                    // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
                    clip_image_f32_ptr res(clip_image_f32_init());
                    normalize_image_u8_to_f32(*imgs[i], *res, params.image_mean, params.image_std);
                    res_imgs->entries.push_back(std::move(res));
                }

                res_imgs->grid_x = inst.grid_size.width;
                res_imgs->grid_y = inst.grid_size.height;
            } break;

        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
            {
                GGML_ASSERT(params.image_min_pixels > 0 && params.image_max_pixels > 0);
                clip_image_u8 resized;
                const clip_image_size new_size = img_tool::calc_size_preserved_ratio(
                    original_size,
                    params.patch_size * 2,
                    params.image_min_pixels,
                    params.image_max_pixels);
                img_tool::resize(*img, resized, new_size, img_tool::RESIZE_ALGO_BILINEAR, false);
                // clip_image_save_to_bmp(resized, "preproc.bmp");
                clip_image_f32_ptr img_f32(clip_image_f32_init());
                // clip_image_f32_ptr res(clip_image_f32_init());
                normalize_image_u8_to_f32(resized, *img_f32, params.image_mean, params.image_std);
                // res_imgs->data[0] = *res;
                res_imgs->entries.push_back(std::move(img_f32));
            } break;

        case PROJECTOR_TYPE_IDEFICS3:
            {
                // The refined size has two steps:
                // 1. Resize w/ aspect-ratio preserving such that the longer side is
                //      the preprocessor longest size
                // 2. Resize w/out preserving aspect ratio such that both sides are
                //      multiples of image_size (always rounding up)
                //
                // CITE: https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py#L737
                const clip_image_size refined_size = img_tool::calc_size_preserved_ratio(
                    original_size, params.image_size, params.image_longest_edge);
                // LOG_INF("%s: original size: %d x %d, refined size: %d x %d\n",
                //         __func__, original_size.width, original_size.height,
                //         refined_size.width, refined_size.height);

                llava_uhd::slice_instructions instructions;
                instructions.overview_size = clip_image_size{params.image_size, params.image_size};
                instructions.refined_size = refined_size;
                instructions.grid_size = clip_image_size{
                    static_cast<int>(std::ceil(static_cast<float>(refined_size.width) / params.image_size)),
                    static_cast<int>(std::ceil(static_cast<float>(refined_size.height) / params.image_size)),
                };
                for (int y = 0; y < refined_size.height; y += params.image_size) {
                    for (int x = 0; x < refined_size.width; x += params.image_size) {
                        // LOG_INF("%s: adding slice at x=%d, y=%d\n", __func__, x, y);
                        instructions.slices.push_back(llava_uhd::slice_coordinates{
                            /* x    */x,
                            /* y    */y,
                            /* size */clip_image_size{
                                std::min(params.image_size, refined_size.width - x),
                                std::min(params.image_size, refined_size.height - y)
                            }
                        });
                    }
                }
                auto imgs = llava_uhd::slice_image(img, instructions);

                // cast and normalize to f32
                for (size_t i = 0; i < imgs.size(); ++i) {
                    // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
                    clip_image_f32_ptr res(clip_image_f32_init());
                    normalize_image_u8_to_f32(*imgs[i], *res, params.image_mean, params.image_std);
                    res_imgs->entries.push_back(std::move(res));
                }

                res_imgs->grid_x = instructions.grid_size.width;
                res_imgs->grid_y = instructions.grid_size.height;
            } break;

        case PROJECTOR_TYPE_GLM_EDGE:
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_INTERNVL: // TODO @ngxson : support dynamic resolution
            {
                clip_image_u8 resized_image;
                int sz = params.image_size;
                img_tool::resize(*img, resized_image, {sz, sz}, img_tool::RESIZE_ALGO_BILINEAR);
                clip_image_f32_ptr img_f32(clip_image_f32_init());
                //clip_image_save_to_bmp(resized_image, "resized.bmp");
                normalize_image_u8_to_f32(resized_image, *img_f32, params.image_mean, params.image_std);
                res_imgs->entries.push_back(std::move(img_f32));
            } break;

        case PROJECTOR_TYPE_JANUS_PRO:
            {
                // Janus Pro preprocessing: pad to square with gray(127), resize to 384x384
                const std::array<uint8_t, 3> pad_color = {127, 127, 127};
                clip_image_u8 resized_image;
                int sz = params.image_size;
                img_tool::resize(*img, resized_image, {sz, sz}, img_tool::RESIZE_ALGO_BILINEAR, true, pad_color);
                clip_image_f32_ptr img_f32(clip_image_f32_init());
                normalize_image_u8_to_f32(resized_image, *img_f32, params.image_mean, params.image_std);
                res_imgs->entries.push_back(std::move(img_f32));
            } break;

        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                GGML_ASSERT(params.image_min_pixels > 0 && params.image_max_pixels > 0);
                clip_image_u8 resized_image;
                // the original pixtral model doesn't have n_merge
                const int cur_merge = params.n_merge == 0 ? 1 : params.n_merge;
                const clip_image_size target_size = img_tool::calc_size_preserved_ratio(
                    original_size,
                    params.patch_size * cur_merge,
                    params.image_min_pixels,
                    params.image_max_pixels);
                img_tool::resize(*img, resized_image, target_size, img_tool::RESIZE_ALGO_BILINEAR);
                clip_image_f32_ptr img_f32(clip_image_f32_init());
                normalize_image_u8_to_f32(resized_image, *img_f32, params.image_mean, params.image_std);
                res_imgs->entries.push_back(std::move(img_f32));
            } break;

        case PROJECTOR_TYPE_LLAMA4:
            {
                GGML_ASSERT(!params.image_res_candidates.empty());
                auto const inst = llava_uhd::get_slice_instructions(ctx, original_size);
                std::vector<clip_image_u8_ptr> imgs = llava_uhd::slice_image(img, inst);

                for (size_t i = 0; i < imgs.size(); ++i) {
                    clip_image_f32_ptr res(clip_image_f32_init());
                    normalize_image_u8_to_f32(*imgs[i], *res, params.image_mean, params.image_std);
                    res_imgs->entries.push_back(std::move(res));
                }

                res_imgs->grid_x = inst.grid_size.width;
                res_imgs->grid_y = inst.grid_size.height;
            } break;

        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_KIMIVL:
            {
                GGML_ASSERT(params.image_min_pixels > 0 && params.image_max_pixels > 0);
                const clip_image_size target_size = img_tool::calc_size_preserved_ratio(
                    original_size,
                    params.patch_size * params.n_merge,
                    params.image_min_pixels,
                    params.image_max_pixels);
                const std::array<uint8_t, 3> pad_color = {122, 116, 104};

                clip_image_u8 resized_img;
                const bool pad = (ctx->proj_type() != PROJECTOR_TYPE_LFM2);
                img_tool::resize(*img, resized_img, target_size, img_tool::RESIZE_ALGO_BILINEAR, pad, pad_color);
                clip_image_f32_ptr res(clip_image_f32_init());
                normalize_image_u8_to_f32(resized_img, *res, params.image_mean, params.image_std);
                res_imgs->entries.push_back(std::move(res));
            } break;

        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_COGVLM: // TODO @ngxson : is this correct for cogvlm?
            {
                // TODO @ngxson : refactor the code below to avoid duplicated logic

                // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
                // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

                clip_image_u8_ptr temp(clip_image_u8_init()); // we will keep the input image data here temporarily

                // The model config actually contains all we need to decide on how to preprocess, here we automatically switch to the new llava-1.6 preprocessing
                if (params.image_res_candidates.empty()) { // pad_to_square
                    // for llava-1.5, we resize image to a square, and pad the shorter side with a background color
                    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156
                    const int longer_side = std::max(img->nx, img->ny);
                    temp->nx = longer_side;
                    temp->ny = longer_side;
                    temp->buf.resize(3 * longer_side * longer_side);

                    // background color in RGB from LLaVA (this is the mean rgb color * 255)
                    const std::array<uint8_t, 3> pad_color = {122, 116, 104};

                    // resize the image to the target_size
                    img_tool::resize(*img, *temp, clip_image_size{params.image_size, params.image_size}, img_tool::RESIZE_ALGO_BILINEAR, true, pad_color);

                    clip_image_f32_ptr res(clip_image_f32_init());
                    normalize_image_u8_to_f32(*temp, *res, params.image_mean, params.image_std);
                    res_imgs->entries.push_back(std::move(res));

                } else {
                    // "spatial_unpad" with "anyres" processing for llava-1.6
                    auto const inst = llava_uhd::get_slice_instructions(ctx, original_size);
                    std::vector<clip_image_u8_ptr> imgs = llava_uhd::slice_image(img, inst);

                    for (size_t i = 0; i < imgs.size(); ++i) {
                        // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
                        clip_image_f32_ptr res(clip_image_f32_init());
                        normalize_image_u8_to_f32(*imgs[i], *res, params.image_mean, params.image_std);
                        res_imgs->entries.push_back(std::move(res));
                    }
                }
            } break;

        default:
            LOG_ERR("%s: unsupported projector type %d\n", __func__, ctx->proj_type());
            return false;
    }

    return true;
}

ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx) {
    return ctx->model.image_newline;
}

void clip_free(clip_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

// deprecated
size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    const int32_t nx = ctx->model.hparams.image_size;
    const int32_t ny = ctx->model.hparams.image_size;
    return clip_embd_nbytes_by_img(ctx, nx, ny);
}

size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_w, int img_h) {
    clip_image_f32 img;
    img.nx = img_w;
    img.ny = img_h;
    return clip_n_output_tokens(ctx, &img) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

int32_t clip_get_image_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.image_size;
}

int32_t clip_get_patch_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.patch_size;
}

int32_t clip_get_hidden_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.n_embd;
}

const char * clip_patch_merge_type(const struct clip_ctx * ctx) {
    return ctx->model.hparams.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD ? "spatial_unpad" : "flat";
}

int clip_n_output_tokens_x(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    const int n_total = clip_n_output_tokens(ctx, img);
    const auto & proj = ctx->proj_type();
    switch (proj) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
            return (img->nx / params.patch_size) / 2;
        default:
            break;
    }
    return n_total;
}

int clip_n_output_tokens_y(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    const auto & proj = ctx->proj_type();
    switch (proj) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
            return (img->ny / params.patch_size) / 2;
        default:
            break;
    }
    return 1;
}

int clip_n_output_tokens(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;

    // for models with fixed size image, the input image is already pre-processed and resized to square
    int patch_size = params.patch_size;
    int n_patches = (img->nx / patch_size) * (img->ny / patch_size);

    projector_type proj = ctx->proj_type();

    switch (proj) {
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_JANUS_PRO:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                n_patches /= 4;
                if (ctx->model.mm_boi) {
                    n_patches += 2; // for BOI and EOI token embeddings
                }
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                // Use actual config value if available, otherwise fall back to hardcoded values
                if (params.minicpmv_query_num > 0) {
                    n_patches = params.minicpmv_query_num;
                } else {
                    // Fallback to hardcoded values for legacy models
                    if (params.minicpmv_version == 2) {
                        n_patches = 96;
                    } else if (params.minicpmv_version == 3) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 4) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 5) {
                        // MiniCPM-V 4.0
                        n_patches = 64;
                    } else if (params.minicpmv_version == 6) {
                        // MiniCPM-V 4.5
                        n_patches = 64;
                    } else {
                        GGML_ABORT("Unknown minicpmv version");
                    }
                }
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
            {
                // dynamic size (2 conv, so double patch size)
                int x_patch = img->nx / (params.patch_size * 2);
                int y_patch = img->ny / (params.patch_size * 2);
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_LLAMA4:
            {
                // both X and Y are downscaled by the scale factor
                int scale_factor = ctx->model.hparams.n_merge;
                n_patches /= (scale_factor * scale_factor);
            } break;
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_KIMIVL:
            {
                // dynamic size
                int out_patch_size = params.patch_size * ctx->model.hparams.n_merge;
                int x_patch = CLIP_ALIGN(img->nx, out_patch_size) / out_patch_size;
                int y_patch = CLIP_ALIGN(img->ny, out_patch_size) / out_patch_size;
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                // dynamic size
                int n_merge = ctx->model.hparams.n_merge;
                int n_patches_x = img->nx / patch_size / (n_merge > 0 ? n_merge : 1);
                int n_patches_y = img->ny / patch_size / (n_merge > 0 ? n_merge : 1);
                if (ctx->model.token_embd_img_break) {
                    n_patches = n_patches_y * n_patches_x + n_patches_y - 1; // + one [IMG_BREAK] per row, except the last row
                } else {
                    n_patches = n_patches_y * n_patches_x;
                }
            } break;
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_QWEN2A:
            {
                n_patches = img->nx;

                const int proj_stack_factor = ctx->model.hparams.proj_stack_factor;
                if (ctx->model.audio_has_stack_frames()) {
                    GGML_ASSERT(proj_stack_factor > 0);
                    const int n_len = CLIP_ALIGN(n_patches, proj_stack_factor);
                    n_patches = n_len / proj_stack_factor;
                }

                // whisper downscales input token by half after conv1d
                n_patches /= 2;

                if (ctx->model.audio_has_avgpool()) {
                    // divide by 2 because of nn.AvgPool1d(2, stride=2)
                    n_patches /= 2;
                }
            } break;
        case PROJECTOR_TYPE_GLMA:
            {
                n_patches = img->nx;
                // whisper downscales input token by half after conv1d
                n_patches /= 2;
                // reshape by merge_factor
                n_patches /= ctx->model.hparams.proj_stack_factor;
                // for BOI and EOI token embeddings
                n_patches += 2;
            } break;
        case PROJECTOR_TYPE_COGVLM:
            {
                n_patches += 2; // for BOI and EOI token embeddings
            } break;
        default:
            GGML_ABORT("unsupported projector type");
    }

    return n_patches;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    clip_image_f32_batch imgs;
    clip_image_f32_ptr img_copy(clip_image_f32_init());
    *img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs_c_ptr, float * vec) {
    const clip_image_f32_batch & imgs = *imgs_c_ptr;
    int batch_size = imgs.entries.size();

    // TODO @ngxson : implement batch size > 1 as a loop
    //                we don't need true batching support because the cgraph will gonna be big anyway
    if (batch_size != 1) {
        return false; // only support batch size of 1
    }

    // if buffers are not allocated, we need to do a warmup run to allocate them
    if (!ctx->is_allocated) {
        clip_model_loader::warmup(*ctx, *imgs_c_ptr);
    }

    // build the inference graph
    ctx->debug_print_tensors.clear();
    ggml_backend_sched_reset(ctx->sched.get());
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs);
    ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);

    // set inputs
    const auto & model   = ctx->model;
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0]->nx;
    const int image_size_height = imgs.entries[0]->ny;

    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int n_pos = num_patches + (model.class_embedding ? 1 : 0);
    const int pos_w = image_size_width  / patch_size;
    const int pos_h = image_size_height / patch_size;

    const bool use_window_attn = hparams.n_wa_pattern > 0; // for qwen2.5vl

    auto get_inp_tensor = [&gf](const char * name) {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (inp == nullptr) {
            GGML_ABORT("Failed to get tensor %s", name);
        }
        if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            GGML_ABORT("Tensor %s is not an input tensor", name);
        }
        return inp;
    };

    auto set_input_f32 = [&get_inp_tensor](const char * name, std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    auto set_input_i32 = [&get_inp_tensor](const char * name, std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // set input pixel values
    if (!imgs.is_audio) {
        size_t nelem = 0;
        for (const auto & img : imgs.entries) {
            nelem += img->nx * img->ny * 3;
        }
        std::vector<float> inp_raw(nelem);

        // layout of data (note: the channel dim is unrolled to better visualize the layout):
        //
        // ┌──W──┐
        // │     H │  channel = R
        // ├─────┤ │
        // │     H │  channel = G
        // ├─────┤ │
        // │     H │  channel = B
        // └─────┘ │
        //   ──────┘ x B

        for (size_t i = 0; i < imgs.entries.size(); i++) {
            const int nx = imgs.entries[i]->nx;
            const int ny = imgs.entries[i]->ny;
            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                float * batch_entry = inp_raw.data() + b * (3*n);
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        size_t base_src = 3*(y * nx + x); // idx of the first channel
                        size_t base_dst =    y * nx + x;  // idx of the first channel
                        batch_entry[      base_dst] = imgs.entries[b]->buf[base_src    ];
                        batch_entry[1*n + base_dst] = imgs.entries[b]->buf[base_src + 1];
                        batch_entry[2*n + base_dst] = imgs.entries[b]->buf[base_src + 2];
                    }
                }
            }
        }
        set_input_f32("inp_raw", inp_raw);

    } else {
        // audio input
        GGML_ASSERT(imgs.entries.size() == 1);
        const auto & mel_inp = imgs.entries[0];
        const int n_step = mel_inp->nx;
        const int n_mel  = mel_inp->ny;
        std::vector<float> inp_raw(n_step * n_mel);
        std::memcpy(inp_raw.data(), mel_inp->buf.data(), n_step * n_mel * sizeof(float));
        set_input_f32("inp_raw", inp_raw);
    }

    // set input per projector
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_MINICPMV:
            {
                // inspired from siglip:
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
                std::vector<int32_t> positions(pos_h * pos_w);
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
                        positions[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
                    }
                }
                set_input_i32("positions", positions);

                // inputs for resampler projector
                // set the 2D positions (using float for sinusoidal embedding)
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<float> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = static_cast<float>(i / n_patches_per_col);
                }
                set_input_f32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = static_cast<float>(i % n_patches_per_col);
                }
                set_input_f32("pos_w", pos_data);
                // base frequency omega
                const float base_freq   = 10000.0f;
                const int   n_embd_proj = clip_n_mmproj_embd(ctx);
                std::vector<float> omega(n_embd_proj / 4);
                for (int i = 0; i < n_embd_proj / 4; ++i) {
                    omega[i] = 1.0f / std::pow(base_freq, static_cast<float>(i) / (n_embd_proj / 4));
                }
                set_input_f32("omega", omega);
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
            {
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size;
                const int ph = image_size_height / patch_size;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                for (int y = 0; y < ph; y += merge_ratio) {
                    for (int x = 0; x < pw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                positions[                  ptr] = y + dy;
                                positions[    num_patches + ptr] = x + dx;
                                positions[2 * num_patches + ptr] = y + dy;
                                positions[3 * num_patches + ptr] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_QWEN25VL:
            {
                // pw * ph = number of tokens output by ViT after apply patch merger
                // ipw * ipw = number of vision token been processed inside ViT
                const int merge_ratio = 2;
                const int pw  = image_size_width  / patch_size / merge_ratio;
                const int ph  = image_size_height / patch_size / merge_ratio;
                const int ipw = image_size_width  / patch_size;
                const int iph = image_size_height / patch_size;

                std::vector<int> idx    (ph * pw);
                std::vector<int> inv_idx(ph * pw);

                if (use_window_attn) {
                    const int attn_window_size = 112;
                    const int grid_window = attn_window_size / patch_size / merge_ratio;
                    int dst = 0;
                    // [num_vision_tokens, num_vision_tokens] attention mask tensor
                    std::vector<float> mask(pow(ipw * iph, 2), std::numeric_limits<float>::lowest());
                    int mask_row = 0;

                    for (int y = 0; y < ph; y += grid_window) {
                        for (int x = 0; x < pw; x += grid_window) {
                            const int win_h = std::min(grid_window, ph - y);
                            const int win_w = std::min(grid_window, pw - x);
                            const int dst_0 = dst;
                            // group all tokens belong to the same window togather (to a continue range)
                            for (int dy = 0; dy < win_h; dy++) {
                                for (int dx = 0; dx < win_w; dx++) {
                                    const int src = (y + dy) * pw + (x + dx);
                                    GGML_ASSERT(src < (int)idx.size());
                                    GGML_ASSERT(dst < (int)inv_idx.size());
                                    idx    [src] = dst;
                                    inv_idx[dst] = src;
                                    dst++;
                                }
                            }

                            for (int r=0; r < win_h * win_w * merge_ratio * merge_ratio; r++) {
                                int row_offset = mask_row * (ipw * iph);
                                std::fill(
                                    mask.begin() + row_offset + (dst_0 * merge_ratio * merge_ratio),
                                    mask.begin() + row_offset + (dst   * merge_ratio * merge_ratio),
                                    0.0);
                                mask_row++;
                            }
                        }
                    }

                    set_input_i32("window_idx",     idx);
                    set_input_i32("inv_window_idx", inv_idx);
                    set_input_f32("window_mask",    mask);
                } else {
                    for (int i = 0; i < ph * pw; i++) {
                        idx[i] = i;
                    }
                }

                const int mpow = merge_ratio * merge_ratio;
                std::vector<int> positions(n_pos * 4);

                int ptr = 0;
                for (int y = 0; y < iph; y += merge_ratio) {
                    for (int x = 0; x < ipw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                auto remap = idx[ptr / mpow];
                                remap = (remap * mpow) + (ptr % mpow);

                                positions[                  remap] = y + dy;
                                positions[    num_patches + remap] = x + dx;
                                positions[2 * num_patches + remap] = y + dy;
                                positions[3 * num_patches + remap] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / n_patches_per_col;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % n_patches_per_col;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_GLM_EDGE:
        {
            // llava and other models
            std::vector<int32_t> positions(n_pos);
            for (int i = 0; i < n_pos; i++) {
                positions[i] = i;
            }
            set_input_i32("positions", positions);
        } break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
            {
                // llava and other models
                std::vector<int32_t> positions(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    positions[i] = i;
                }
                set_input_i32("positions", positions);

                // The patches vector is used to get rows to index into the embeds with;
                // we should skip dim 0 only if we have CLS to avoid going out of bounds
                // when retrieving the rows.
                int patch_offset = model.class_embedding ? 1 : 0;
                std::vector<int32_t> patches(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    patches[i] = i + patch_offset;
                }
                set_input_i32("patches", patches);
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_COGVLM:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(num_patches + 1, 0); // +1 for the [CLS] token
                // last pos is always kept 0, it's for CLS
                // dimension H
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i / n_patches_per_col) + 1;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i % n_patches_per_col) + 1;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        default:
            GGML_ABORT("Unknown projector type");
    }

    // ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(ctx->backend_cpu, n_threads);
        }
    }

    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    // print debug nodes
    if (ctx->debug_graph) {
        LOG_INF("\n\n---\n\n");
        LOG_INF("\n\nDebug graph:\n\n");
        for (ggml_tensor * t : ctx->debug_print_tensors) {
            std::vector<uint8_t> data(ggml_nbytes(t));
            ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
            print_tensor_shape(t);
            print_tensor_data(t, data.data(), 3);
        }
    }

    // the last node is the embedding tensor
    ggml_tensor * embeddings = ggml_graph_node(gf, -1);

    // sanity check (only support batch size of 1 for now)
    const int n_tokens_out = embeddings->ne[1];
    const int expected_n_tokens_out = clip_n_output_tokens(ctx, imgs.entries[0].get());
    if (n_tokens_out != expected_n_tokens_out) {
        LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
        GGML_ABORT("Invalid number of output tokens");
    }

    // copy the embeddings to the location passed by the user
    if (vec != nullptr) {
        ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));
    }

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_LDP:
            return ctx->model.mm_model_block_1_block_2_1_b->ne[0];
        case PROJECTOR_TYPE_LDPV2:
            return ctx->model.mm_model_peg_0_b->ne[0];
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_MLP_NORM:
            return ctx->model.mm_3_b->ne[0];
        case PROJECTOR_TYPE_MINICPMV:
            return ctx->model.mm_model_proj->ne[0];
        case PROJECTOR_TYPE_GLM_EDGE:
            return ctx->model.mm_model_mlp_3_w->ne[1];
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_JANUS_PRO:
            return ctx->model.mm_1_b->ne[0];
        case PROJECTOR_TYPE_QWEN3VL:
            // main path + deepstack paths
            return ctx->model.mm_1_b->ne[0] * (1 + ctx->model.n_deepstack_layers);
        case PROJECTOR_TYPE_GEMMA3:
            return ctx->model.mm_input_proj_w->ne[0];
        case PROJECTOR_TYPE_IDEFICS3:
            return ctx->model.projection->ne[1];
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_INTERNVL:
            return ctx->model.mm_3_w->ne[1];
        case PROJECTOR_TYPE_LLAMA4:
            return ctx->model.mm_model_proj->ne[1];
        case PROJECTOR_TYPE_QWEN2A:
            return ctx->model.mm_fc_w->ne[1];
        case PROJECTOR_TYPE_GLMA:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_KIMIVL:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_COGVLM:
            return ctx->model.mm_4h_to_h_w->ne[1];
        case PROJECTOR_TYPE_GLM4V:
            return ctx->model.mm_ffn_down_w->ne[1];
        default:
            GGML_ABORT("Unknown projector type");
    }
}

int clip_is_minicpmv(const struct clip_ctx * ctx) {
    if (ctx->proj_type() == PROJECTOR_TYPE_MINICPMV) {
        return ctx->model.hparams.minicpmv_version;
    }
    return 0;
}

bool clip_is_glm(const struct clip_ctx * ctx) {
    return ctx->proj_type() == PROJECTOR_TYPE_GLM_EDGE;
}

bool clip_is_mrope(const struct clip_ctx * ctx) {
    return ctx->proj_type() == PROJECTOR_TYPE_QWEN2VL
        || ctx->proj_type() == PROJECTOR_TYPE_QWEN25VL
        || ctx->proj_type() == PROJECTOR_TYPE_QWEN3VL
        || ctx->proj_type() == PROJECTOR_TYPE_GLM4V;
}

bool clip_is_llava(const struct clip_ctx * ctx) {
    return ctx->model.hparams.has_llava_projector;
}

bool clip_is_gemma3(const struct clip_ctx * ctx) {
    return ctx->proj_type() == PROJECTOR_TYPE_GEMMA3;
}

bool clip_has_vision_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_VISION;
}

bool clip_has_audio_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_AUDIO;
}

bool clip_has_whisper_encoder(const struct clip_ctx * ctx) {
    return ctx->proj_type() == PROJECTOR_TYPE_ULTRAVOX
        || ctx->proj_type() == PROJECTOR_TYPE_QWEN2A
        || ctx->proj_type() == PROJECTOR_TYPE_GLMA
        || ctx->proj_type() == PROJECTOR_TYPE_VOXTRAL;
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
    return ctx->proj_type();
}

void clip_image_f32_batch_add_mel(struct clip_image_f32_batch * batch, int n_mel, int n_frames, float * mel) {
    clip_image_f32 * audio = new clip_image_f32;
    audio->nx = n_frames;
    audio->ny = n_mel;
    audio->buf.resize(n_frames * n_mel);
    std::memcpy(audio->buf.data(), mel, n_frames * n_mel * sizeof(float));

    batch->entries.push_back(clip_image_f32_ptr(audio));
    batch->is_audio = true;
}

const clip_hparams * clip_get_hparams(const struct clip_ctx * ctx) {
    return &ctx->model.hparams;
}

//
// API for debugging
//

void clip_debug_encode(clip_ctx * ctx, int h, int w, float fill_value) {
    clip_image_f32 img;
    img.nx = w;
    img.ny = h;
    img.buf.resize(h * w * 3);
    for (int i = 0; i < h * w * 3; i++) {
        img.buf[i] = static_cast<float>(fill_value);
    }
    bool cur_debug_graph = ctx->debug_graph;
    ctx->debug_graph = true;
    clip_image_encode(ctx, 1, &img, nullptr);
    ctx->debug_graph = cur_debug_graph;
    GGML_ASSERT(img.buf.empty() && "expected, always stop here");
}
