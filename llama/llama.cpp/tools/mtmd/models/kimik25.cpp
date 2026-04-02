#include "models.h"
#include <cstring>
#include <cmath>

// note: this is similar to clip_graph::resize_position_embeddings, major difference is having
// the w/h in ne[1] and ne[2] instead of assuming with sqrt. Could try storing the tensor in 2D instead
// with a w*h? Also the permute is a bit different at (2, 1, 0, 3) instead of (2, 0, 1, 3).
ggml_tensor * clip_graph_kimik25::resize_position_embeddings_3d(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny / patch_size;
    const int width        = img.nx / patch_size;
    const uint32_t mode    = interpolation_mode;

    GGML_ASSERT(pos_embd);

    const int64_t stored_c = pos_embd->ne[0];  // C = 1152
    const int64_t orig_w = pos_embd->ne[1];    // W = 64
    const int64_t orig_h = pos_embd->ne[2];    // H = 64

    GGML_ASSERT(stored_c == n_embd);

    if (height == (int)orig_h && width == (int)orig_w) {
        // No interpolation needed, just flatten to [C, H*W]
        return ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    }

    pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = ggml_interpolate(ctx0, pos_embd, height, width, n_embd, 1, mode);
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    return pos_embd;
}

ggml_cgraph * clip_graph_kimik25::build() {
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * learned_pos_embd = resize_position_embeddings_3d(GGML_SCALE_MODE_BICUBIC);

    // Kimi-K2.5 uses interleaved 2D RoPE pattern natively, but
    // Q / K are permuted during conversion to use split format.
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        cur = build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
        return cur;
    };

    ggml_tensor * inp = build_inp();

    // I don't know why, but doing this in the build_vit lead to the ggml_add not occurring?
    // Doing it manually here does work.
    inp = ggml_add(ctx0, inp, learned_pos_embd);

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            nullptr,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        // patch_merger
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        // projection norm
        int proj_inp_dim = cur->ne[0];
        int n_merged_patches = cur->ne[1];
        cur = ggml_view_2d(ctx0, cur,
            n_embd, n_merged_patches * scale_factor * scale_factor,
            ggml_row_size(cur->type, n_embd), 0);
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        cur = ggml_view_2d(ctx0, cur,
            proj_inp_dim, n_merged_patches,
            ggml_row_size(cur->type, proj_inp_dim), 0);
        cb(cur, "proj_inp_normed", -1);

        // projection mlp
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

        cb(cur, "proj_out", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
