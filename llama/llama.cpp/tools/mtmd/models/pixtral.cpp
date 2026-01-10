#include "models.h"

ggml_cgraph * clip_graph_pixtral::build() {
    const int n_merge = hparams.n_merge;

    // 2D input positions
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_h, pos_w, hparams.rope_theta, true);
    };

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_RMS,
                            hparams.ffn_op,
                            nullptr, // no learned pos embd
                            add_pos);

    // mistral small 3.1 patch merger
    // ref: https://github.com/huggingface/transformers/blob/7a3e208892c06a5e278144eaf38c8599a42f53e7/src/transformers/models/mistral3/modeling_mistral3.py#L67
    if (model.mm_patch_merger_w) {
        GGML_ASSERT(hparams.n_merge > 0);

        cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), model.mm_input_norm_w);

        // reshape image tokens to 2D grid
        cur = ggml_reshape_3d(ctx0, cur, n_embd, n_patches_x, n_patches_y);
        cur = ggml_permute(ctx0, cur, 2, 0, 1, 3); // [x, y, n_embd]
        cur = ggml_cont(ctx0, cur);

        // torch.nn.functional.unfold is just an im2col under the hood
        // we just need a dummy kernel to make it work
        ggml_tensor * kernel = ggml_view_3d(ctx0, cur, n_merge, n_merge, cur->ne[2], 0, 0, 0);
        cur = ggml_im2col(ctx0, kernel, cur, n_merge, n_merge, 0, 0, 1, 1, true, inp->type);

        // project to n_embd
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
        cur = ggml_mul_mat(ctx0, model.mm_patch_merger_w, cur);
    }

    // LlavaMultiModalProjector (always using GELU activation)
    {
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);
    }

    // arrangement of the [IMG_BREAK] token
    if (model.token_embd_img_break) {
        // not efficient, but works
        // the trick is to view the embeddings as a 3D tensor with shape [n_embd, n_patches_per_row, n_rows]
        // and then concatenate the [IMG_BREAK] token to the end of each row, aka n_patches_per_row dimension
        // after the concatenation, we have a tensor with shape [n_embd, n_patches_per_row + 1, n_rows]

        const int p_y             = n_merge > 0 ? n_patches_y / n_merge : n_patches_y;
        const int p_x             = n_merge > 0 ? n_patches_x / n_merge : n_patches_x;
        const int p_total         = p_x * p_y;
        const int n_embd_text     = cur->ne[0];
        const int n_tokens_output = p_total + p_y - 1; // one [IMG_BREAK] per row, except the last row

        ggml_tensor * tmp = ggml_reshape_3d(ctx0, cur, n_embd_text, p_x, p_y);
        ggml_tensor * tok = ggml_new_tensor_3d(ctx0, tmp->type, n_embd_text, 1, p_y);
        tok = ggml_scale(ctx0, tok, 0.0); // clear the tensor
        tok = ggml_add(ctx0, tok, model.token_embd_img_break);
        tmp = ggml_concat(ctx0, tmp, tok, 1);
        cur = ggml_view_2d(ctx0, tmp,
            n_embd_text, n_tokens_output,
            ggml_row_size(tmp->type, n_embd_text), 0);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
