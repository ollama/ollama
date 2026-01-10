#include "models.h"

ggml_cgraph * clip_graph_llama4::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_pos = n_patches + 1; // +1 for [CLS]

    // 2D input positions
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * inp = build_inp_raw();

    // Llama4UnfoldConvolution
    {
        ggml_tensor * kernel = ggml_reshape_4d(ctx0, model.patch_embeddings_0,
                                                patch_size, patch_size, 3, n_embd);
        inp = ggml_im2col(ctx0, kernel, inp, patch_size, patch_size, 0, 0, 1, 1, true, inp->type);
        inp = ggml_mul_mat(ctx0, model.patch_embeddings_0, inp);
        inp = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);
        cb(inp, "patch_conv", -1);
    }

    // add CLS token
    inp = ggml_concat(ctx0, inp, model.class_embedding, 1);

    // build ViT with 2D position embeddings
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        // first half is X axis and second half is Y axis
        // ref: https://github.com/huggingface/transformers/blob/40a493c7ed4f19f08eadb0639cf26d49bfa5e180/src/transformers/models/llama4/modeling_llama4.py#L1312
        // ref: https://github.com/Blaizzy/mlx-vlm/blob/a57156aa87b33cca6e5ee6cfc14dd4ef8f611be6/mlx_vlm/models/llama4/vision.py#L441
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };
    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            model.position_embeddings,
                            add_pos);

    // remove CLS token
    cur = ggml_view_2d(ctx0, cur,
        n_embd, n_patches,
        ggml_row_size(cur->type, n_embd), 0);

    // pixel shuffle
    // based on Llama4VisionPixelShuffleMLP
    // https://github.com/huggingface/transformers/blob/2932f318a20d9e54cc7aea052e040164d85de7d6/src/transformers/models/llama4/modeling_llama4.py#L1151
    {
        const int scale_factor = model.hparams.n_merge;
        const int bsz = 1; // batch size, always 1 for now since we don't support batching
        GGML_ASSERT(scale_factor > 0);
        GGML_ASSERT(n_patches_x == n_patches_y); // llama4 only supports square images
        cur = ggml_reshape_4d(ctx0, cur,
            n_embd * scale_factor,
            n_patches_x / scale_factor,
            n_patches_y,
            bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_4d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            n_patches_x / scale_factor,
            n_patches_y / scale_factor,
            bsz);
        //cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        // flatten to 2D
        cur = ggml_cont_2d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            n_patches / scale_factor / scale_factor);
        cb(cur, "pixel_shuffle", -1);
    }

    // based on Llama4VisionMLP2 (always uses GELU activation, no bias)
    {
        cur = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w, cur);
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, cur);
        cur = ggml_gelu(ctx0, cur);
        cb(cur, "adapter_mlp", -1);
    }

    // Llama4MultiModalProjector
    cur = ggml_mul_mat(ctx0, model.mm_model_proj, cur);
    cb(cur, "projected", -1);

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
