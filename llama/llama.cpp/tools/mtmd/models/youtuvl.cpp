#include "models.h"

ggml_cgraph * clip_graph_youtuvl::build() {
    GGML_ASSERT(model.class_embedding == nullptr);
    const int batch_size       = 1;
    const bool use_window_attn = !hparams.wa_layer_indexes.empty();
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4;
    const int m = 2;
    const int Wp = n_patches_x;
    const int Hp = n_patches_y;
    const int Hm = Hp / m;
    const int Wm = Wp / m;
    norm_type norm_t = NORM_TYPE_NORMAL;

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp = build_inp_raw();

    // change conv3d to linear
    // reshape and permute to get patches, permute from (patch_size, m, Wm, patch_size, m, Hm, C) to (C, patch_size, patch_size, m, m, Wm, Hm)
    {
        inp = ggml_reshape_4d(
            ctx0, inp,
            Wm * m * patch_size, m * patch_size, Hm, 3);
        inp = ggml_permute(ctx0, inp, 1, 2, 3, 0);
        inp = ggml_cont_4d(
            ctx0, inp,
            m * patch_size * 3, Wm, m * patch_size, Hm);

        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_4d(
            ctx0, inp,
            m * patch_size * 3, patch_size, m, Hm * Wm);

        inp = ggml_permute(ctx0, inp, 1, 0, 2, 3);
        inp = ggml_cont_4d(
            ctx0, inp,
            patch_size, 3, patch_size, Hm * Wm * m * m);

        inp = ggml_permute(ctx0, inp, 2, 0, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            3*patch_size* patch_size,  Hm * Wm * m * m, 1);
    }
    inp = ggml_mul_mat(ctx0, model.patch_embeddings_0, inp);

    if (model.patch_bias) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
    }

    inp = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);

    ggml_tensor * inpL           = inp;
    ggml_tensor * window_mask    = nullptr;
    ggml_tensor * window_idx     = nullptr;
    ggml_tensor * inv_window_idx = nullptr;

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }
    if (use_window_attn) {
        inv_window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(inv_window_idx, "inv_window_idx");
        ggml_set_input(inv_window_idx);
        // mask for window attention
        window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
        ggml_set_name(window_mask, "window_mask");
        ggml_set_input(window_mask);

        // if flash attn is used, we need to pad the mask and cast to f16
        if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
            window_mask = ggml_cast(ctx0, window_mask, GGML_TYPE_F16);
        }

        // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        GGML_ASSERT(batch_size == 1);
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_patches_x * n_patches_y * batch_size / 4);
        inpL = ggml_get_rows(ctx0, inpL, inv_window_idx);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const bool full_attn = use_window_attn ? hparams.wa_layer_indexes.count(il) > 0 : true;

        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        // self-attention
        {
            ggml_tensor * Qcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, layer.q_w, cur), layer.q_b);
            ggml_tensor * Kcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, layer.k_w, cur), layer.k_b);
            ggml_tensor * Vcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, layer.v_w, cur), layer.v_b);

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

            Qcur = ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
        }
        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);

        inpL = cur;
    }

    ggml_tensor * embeddings = inpL;
    if (use_window_attn) {
        const int spatial_merge_unit = 4;
        window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / spatial_merge_unit);
        ggml_set_name(window_idx, "window_idx");
        ggml_set_input(window_idx);
        GGML_ASSERT(batch_size == 1);
        embeddings = ggml_reshape_2d(ctx0, embeddings, n_embd * spatial_merge_unit, n_patches / spatial_merge_unit);
        embeddings = ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd, n_patches, batch_size);
        cb(embeddings, "window_order_restored", -1);
    }

    // post-layernorm (part of Siglip2VisionTransformer, applied after encoder)
    if (model.post_ln_w) {
        embeddings = build_norm(embeddings, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    }

    // Now apply merger (VLPatchMerger):
    // 1. Apply RMS norm (ln_q in VLPatchMerger)
    embeddings = build_norm(embeddings, model.mm_input_norm_w, nullptr, NORM_TYPE_RMS, 1e-6, -1);
    cb(embeddings, "merger_normed", -1);

    // 2. First reshape for spatial merge (merge 2x2 patches)
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);
    cb(embeddings, "merger_reshaped", -1);

    embeddings = build_ffn(embeddings,
                    model.mm_0_w, model.mm_0_b,
                    nullptr, nullptr,
                    model.mm_1_w, model.mm_1_b,
                    FFN_GELU,
                    -1);
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
