#pragma once

#include "ggml.h"
#include "ggml-cpp.h"
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"

#include <vector>
#include <functional>

struct clip_graph {
    const clip_model & model;
    const clip_hparams & hparams;
    projector_type proj_type;

    // we only support single image per batch
    const clip_image_f32 & img;

    const int patch_size;
    const int n_patches_x;
    const int n_patches_y;
    const int n_patches;
    const int n_embd;
    const int n_head;
    const int d_head;
    const int n_layer;
    const int n_mmproj_embd;
    const float eps;
    const float kq_scale;
    const clip_flash_attn_type flash_attn_type;

    // for debugging
    const bool debug_graph;
    std::vector<ggml_tensor *> & debug_print_tensors;

    ggml_context_ptr ctx0_ptr;
    ggml_context * ctx0;
    ggml_cgraph * gf;

    clip_graph(clip_ctx * ctx, const clip_image_f32 & img);

    virtual ~clip_graph() = default;
    virtual ggml_cgraph * build() = 0;

    //
    // utility functions
    //
    void cb(ggml_tensor * cur0, const char * name, int il) const;

    // siglip2 naflex
    ggml_tensor * resize_position_embeddings();

    // build vision transformer (ViT) cgraph
    // this function should cover most of the models
    // if your model has specific features, you should probably duplicate this function
    ggml_tensor * build_vit(
                ggml_tensor * inp,
                int64_t n_pos,
                norm_type norm_t,
                ffn_op_type ffn_t,
                ggml_tensor * learned_pos_embd,
                std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> add_pos);

    // build the input after conv2d (inp_raw --> patches)
    // returns tensor with shape [n_embd, n_patches]
    ggml_tensor * build_inp();

    ggml_tensor * build_inp_raw(int channels = 3);

    ggml_tensor * build_norm(
            ggml_tensor * cur,
            ggml_tensor * mw,
            ggml_tensor * mb,
            norm_type type,
            float norm_eps,
            int il) const;

    ggml_tensor * build_ffn(
            ggml_tensor * cur,
            ggml_tensor * up,
            ggml_tensor * up_b,
            ggml_tensor * gate,
            ggml_tensor * gate_b,
            ggml_tensor * down,
            ggml_tensor * down_b,
            ffn_op_type type_op,
            int il) const;

    ggml_tensor * build_attn(
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * kq_mask,
            float kq_scale,
            int il) const;

    // implementation of the 2D RoPE without adding a new op in ggml
    // this is not efficient (use double the memory), but works on all backends
    // TODO: there was a more efficient which relies on ggml_view and ggml_rope_ext_inplace, but the rope inplace does not work well with non-contiguous tensors ; we should fix that and revert back to the original implementation in https://github.com/ggml-org/llama.cpp/pull/13065
    ggml_tensor * build_rope_2d(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * pos_a, // first half
        ggml_tensor * pos_b, // second half
        const float freq_base,
        const bool interleave_freq
    );

    // aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
    // support dynamic resolution
    ggml_tensor * build_patch_merge_permute(ggml_tensor * cur, int scale_factor);
};
