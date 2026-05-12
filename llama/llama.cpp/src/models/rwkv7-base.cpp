#include "models.h"

llm_build_rwkv7_base::llm_build_rwkv7_base(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params),
    model(model) {}

ggml_tensor * llm_build_rwkv7_base::build_rwkv7_channel_mix(const llama_layer * layer,
                                                            ggml_tensor *       cur,
                                                            ggml_tensor *       x_prev,
                                                            llm_arch            arch) const {
    ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
    switch (arch) {
        case LLM_ARCH_RWKV7:
            {
                ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);

                ggml_tensor * k = ggml_sqr(ctx0, ggml_relu(ctx0, build_lora_mm(layer->channel_mix_key, xk)));

                cur = build_lora_mm(layer->channel_mix_value, k);
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }
    return cur;
}

ggml_tensor * llm_build_rwkv7_base::build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                                         ggml_tensor *        cur,
                                                         ggml_tensor *        x_prev,
                                                         ggml_tensor *&       first_layer_value,
                                                         const llama_ubatch & ubatch,
                                                         int                  il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto n_tokens     = ubatch.n_tokens;
    const auto n_seqs       = ubatch.n_seqs;
    const auto n_embd       = hparams.n_embd;
    const auto head_size    = hparams.wkv_head_size;
    const auto head_count   = n_embd / head_size;
    const auto n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    const auto & layer = model.layers[il];

    bool has_gating = layer.time_mix_g1 && layer.time_mix_g2;

    ggml_tensor * sx    = ggml_sub(ctx0, x_prev, cur);
    ggml_tensor * dummy = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_embd, n_seq_tokens, n_seqs, has_gating ? 6 : 5);
    sx                  = ggml_repeat(ctx0, sx, dummy);

    ggml_tensor * xxx = ggml_add(ctx0, ggml_mul(ctx0, sx, layer.time_mix_lerp_fused), cur);

    ggml_tensor * xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
    ggml_tensor * xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
    ggml_tensor * xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
    ggml_tensor * xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
    ggml_tensor * xa = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));
    ggml_tensor * xg =
        has_gating ? ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 5 * sizeof(float)) :
                     nullptr;

    ggml_tensor * r = build_lora_mm(layer.time_mix_receptance, xr);
    ggml_tensor * w = ggml_add(
        ctx0, ggml_mul_mat(ctx0, layer.time_mix_w2, ggml_tanh(ctx0, ggml_mul_mat(ctx0, layer.time_mix_w1, xw))),
        layer.time_mix_w0);
    w = ggml_exp(ctx0, ggml_scale(ctx0, ggml_sigmoid(ctx0, w), -0.606531));

    ggml_tensor * k = build_lora_mm(layer.time_mix_key, xk);
    ggml_tensor * v = build_lora_mm(layer.time_mix_value, xv);
    if (first_layer_value == nullptr) {
        first_layer_value = v;
    } else {
        // Add the first layer value as a residual connection.
        v = ggml_add(ctx0, v,
                     ggml_mul(ctx0, ggml_sub(ctx0, first_layer_value, v),
                              ggml_sigmoid(ctx0, ggml_add(ctx0,
                                                          ggml_mul_mat(ctx0, layer.time_mix_v2,
                                                                       ggml_mul_mat(ctx0, layer.time_mix_v1, xv)),
                                                          layer.time_mix_v0))));
    }
    ggml_tensor * g = nullptr;
    if (layer.time_mix_g1 && layer.time_mix_g2) {
        g = ggml_mul_mat(ctx0, layer.time_mix_g2, ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, layer.time_mix_g1, xg)));
    }
    ggml_tensor * a = ggml_sigmoid(
        ctx0, ggml_add(ctx0, ggml_mul_mat(ctx0, layer.time_mix_a2, ggml_mul_mat(ctx0, layer.time_mix_a1, xa)),
                       layer.time_mix_a0));

    ggml_tensor * kk = ggml_reshape_3d(ctx0, ggml_mul(ctx0, k, layer.time_mix_k_k), head_size, head_count, n_tokens);
    kk               = ggml_l2_norm(ctx0, kk, 1e-12);

    ggml_tensor * ka = ggml_mul(ctx0, k, layer.time_mix_k_a);
    k                = ggml_add(ctx0, k, ggml_sub(ctx0, ggml_mul(ctx0, a, ka), ka));

    r = ggml_reshape_3d(ctx0, r, head_size, head_count, n_tokens);
    w = ggml_reshape_3d(ctx0, w, head_size, head_count, n_tokens);
    k = ggml_reshape_3d(ctx0, k, head_size, head_count, n_tokens);
    v = ggml_reshape_3d(ctx0, v, head_size, head_count, n_tokens);
    a = ggml_reshape_3d(ctx0, a, head_size, head_count, n_tokens);

    ggml_tensor * wkv_state = build_rs(inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);

    ggml_tensor * wkv_output = ggml_rwkv_wkv7(ctx0, r, w, k, v, ggml_neg(ctx0, kk), ggml_mul(ctx0, kk, a), wkv_state);
    cur                      = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
    wkv_state = ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

    ggml_build_forward_expand(
        gf, ggml_cpy(ctx0, wkv_state,
                     ggml_view_1d(ctx0, mctx_cur->get_s_l(il), hparams.n_embd_s() * n_seqs,
                                  hparams.n_embd_s() * kv_head * ggml_element_size(mctx_cur->get_s_l(il)))));

    if (layer.time_mix_ln && layer.time_mix_ln_b) {
        // group norm with head_count groups
        cur = ggml_reshape_3d(ctx0, cur, n_embd / head_count, head_count, n_tokens);
        cur = ggml_norm(ctx0, cur, 64e-5f);

        // Convert back to regular vectors.
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.time_mix_ln), layer.time_mix_ln_b);
    } else {
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    }
    ggml_tensor * rk = ggml_sum_rows(
        ctx0, ggml_mul(ctx0, ggml_mul(ctx0, k, r), ggml_reshape_2d(ctx0, layer.time_mix_r_k, head_size, head_count)));
    cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, ggml_mul(ctx0, v, rk), n_embd, n_tokens));

    if (has_gating) {
        cur = ggml_mul(ctx0, cur, g);
    }
    cur = build_lora_mm(layer.time_mix_output, cur);

    return ggml_reshape_3d(ctx0, cur, n_embd, n_seq_tokens, n_seqs);
}
