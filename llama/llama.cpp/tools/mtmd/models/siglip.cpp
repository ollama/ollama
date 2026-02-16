#include "models.h"

ggml_cgraph * clip_graph_siglip::build() {
    ggml_tensor * inp = build_inp();

    ggml_tensor * learned_pos_embd = model.position_embeddings;
    if (proj_type == PROJECTOR_TYPE_LFM2) {
        learned_pos_embd = resize_position_embeddings();
    }

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            nullptr);

    if (proj_type == PROJECTOR_TYPE_GEMMA3) {
        const int batch_size = 1;
        GGML_ASSERT(n_patches_x == n_patches_y);
        const int patches_per_image = n_patches_x;
        const int kernel_size = hparams.n_merge;

        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont_4d(ctx0, cur, patches_per_image, patches_per_image, n_embd, batch_size);

        // doing a pool2d to reduce the number of output tokens
        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[0], n_embd, batch_size);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        // apply norm before projection
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);

        // apply projection
        cur = ggml_mul_mat(ctx0,
            ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_input_proj_w)),
            cur);

    } else if (proj_type == PROJECTOR_TYPE_IDEFICS3) {
        // pixel_shuffle
        // https://github.com/huggingface/transformers/blob/0a950e0bbe1ed58d5401a6b547af19f15f0c195e/src/transformers/models/idefics3/modeling_idefics3.py#L578
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);
        cur = ggml_mul_mat(ctx0, model.projection, cur);

    } else if (proj_type == PROJECTOR_TYPE_LFM2) {
        // pixel unshuffle block
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        // projection
        cur = ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
        cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = ggml_add(ctx0, cur, model.mm_input_norm_b);

        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_JANUS_PRO) {
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_1_w, model.mm_1_b,
            hparams.ffn_op,
            -1);

    } else {
        GGML_ABORT("SigLIP: Unsupported projector type");
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
