#include "models.h"

ggml_cgraph * clip_graph_cogvlm::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_pos = n_patches + 1; // +1 for [CLS]

    // build input and concatenate class embedding
    ggml_tensor * inp = build_inp();
    inp = ggml_concat(ctx0, inp, model.class_embedding, 1);

    inp = ggml_add(ctx0, inp, model.position_embeddings);
    cb(inp, "inp_pos", -1);

    ggml_tensor * inpL = inp;

    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);

        cur = ggml_add(ctx0, cur, layer.qkv_b);

        ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, d_head*sizeof(float),
            cur->nb[1], 0);
        ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, d_head*sizeof(float),
            cur->nb[1], n_embd * sizeof(float));
        ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, d_head*sizeof(float),
            cur->nb[1], 2 * n_embd * sizeof(float));

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(layer.o_w, layer.o_b,
            Qcur, Kcur, Vcur, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "attn_post_norm", il);

        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "layer_out", il);
        inpL = cur;

    }

    // remove CLS token (like build_llama4 does)
    ggml_tensor * cur = ggml_view_2d(ctx0, inpL,
        n_embd, n_patches,
        ggml_row_size(inpL->type, n_embd), 0);

    // Multiply with mm_model_proj
    cur = ggml_mul_mat(ctx0, model.mm_model_proj, cur);

    // Apply layernorm, weight, bias
    cur = build_norm(cur, model.mm_post_fc_norm_w, model.mm_post_fc_norm_b, NORM_TYPE_NORMAL, 1e-5, -1);

    // Apply GELU
    cur = ggml_gelu_inplace(ctx0, cur);

    // Branch 1: multiply with mm_h_to_4h_w
    ggml_tensor * h_to_4h = ggml_mul_mat(ctx0, model.mm_h_to_4h_w, cur);

    // Branch 2: multiply with mm_gate_w
    ggml_tensor * gate = ggml_mul_mat(ctx0, model.mm_gate_w, cur);

    // Apply silu
    gate = ggml_swiglu_split(ctx0, gate, h_to_4h);

    // Apply mm_4h_to_h_w
    cur = ggml_mul_mat(ctx0, model.mm_4h_to_h_w, gate);

    // Concatenate with boi and eoi
    cur = ggml_concat(ctx0, model.mm_boi, cur, 1);
    cur = ggml_concat(ctx0, cur, model.mm_eoi, 1);

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
