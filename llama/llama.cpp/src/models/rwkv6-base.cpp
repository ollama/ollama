#include "models.h"

#include "llama-memory-recurrent.h"

llm_build_rwkv6_base::llm_build_rwkv6_base(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params),
    model(model) {}

ggml_tensor * llm_build_rwkv6_base::build_rwkv6_channel_mix(const llama_layer * layer,
                                                            ggml_tensor *       cur,
                                                            ggml_tensor *       x_prev,
                                                            llm_arch            arch) const {
    ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
    switch (arch) {
        case LLM_ARCH_RWKV6:
            {
                ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);
                ggml_tensor * xr = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_r), cur);

                ggml_tensor * r = ggml_sigmoid(ctx0, build_lora_mm(layer->channel_mix_receptance, xr));
                ggml_tensor * k = ggml_sqr(ctx0, ggml_relu(ctx0, build_lora_mm(layer->channel_mix_key, xk)));
                cur             = ggml_mul(ctx0, r, build_lora_mm(layer->channel_mix_value, k));
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }
    return cur;
}

ggml_tensor * llm_build_rwkv6_base::build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                                         ggml_tensor *        cur,
                                                         ggml_tensor *        x_prev,
                                                         const llama_ubatch & ubatch,
                                                         int                  il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto n_tokens     = ubatch.n_tokens;
    const auto n_seqs       = ubatch.n_seqs;
    const auto n_seq_tokens = ubatch.n_seq_tokens;
    const auto n_embd       = hparams.n_embd;
    const auto head_size    = hparams.wkv_head_size;
    const auto n_head       = n_embd / head_size;
    const auto n_head_kv    = hparams.n_head_kv(il);

    const auto kv_head = mctx_cur->get_head();

    const auto & layer = model.layers[il];

    bool is_qrwkv = layer.time_mix_first == nullptr;

    ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);

    sx  = ggml_reshape_2d(ctx0, sx, n_embd, n_tokens);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

    ggml_tensor * xxx = ggml_add(ctx0, ggml_mul(ctx0, sx, layer.time_mix_lerp_x), cur);

    xxx = ggml_reshape_4d(ctx0, ggml_tanh(ctx0, ggml_mul_mat(ctx0, layer.time_mix_w1, xxx)),
                          layer.time_mix_w1->ne[1] / 5, 1, 5, n_tokens);

    xxx = ggml_cont(ctx0, ggml_permute(ctx0, xxx, 0, 1, 3, 2));

    xxx = ggml_mul_mat(
        ctx0, ggml_reshape_4d(ctx0, layer.time_mix_w2, layer.time_mix_w2->ne[0], layer.time_mix_w2->ne[1], 1, 5), xxx);

    ggml_tensor *xw, *xk, *xv, *xr, *xg;
    if (layer.time_mix_lerp_fused) {
        // fusing these weights makes some performance improvement
        sx  = ggml_reshape_3d(ctx0, sx, n_embd, 1, n_tokens);
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        xxx = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xxx, layer.time_mix_lerp_fused), sx), cur);
        xw  = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
        xk  = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
        xv  = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
        xr  = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
        xg  = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));
    } else {
        // for backward compatibility
        xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
        xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
        xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
        xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
        xg = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));

        xw = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xw, layer.time_mix_lerp_w), sx), cur);
        xk = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xk, layer.time_mix_lerp_k), sx), cur);
        xv = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xv, layer.time_mix_lerp_v), sx), cur);
        xr = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xr, layer.time_mix_lerp_r), sx), cur);
        xg = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xg, layer.time_mix_lerp_g), sx), cur);
    }
    ggml_tensor * r = build_lora_mm(layer.time_mix_receptance, xr);
    ggml_tensor * k = build_lora_mm(layer.time_mix_key, xk);
    ggml_tensor * v = build_lora_mm(layer.time_mix_value, xv);
    if (layer.time_mix_receptance_b) {
        r = ggml_add(ctx0, r, layer.time_mix_receptance_b);
    }
    if (layer.time_mix_key_b) {
        k = ggml_add(ctx0, k, layer.time_mix_key_b);
    }
    if (layer.time_mix_value_b) {
        v = ggml_add(ctx0, v, layer.time_mix_value_b);
    }
    ggml_tensor * g = build_lora_mm(layer.time_mix_gate, xg);
    if (is_qrwkv) {
        g = ggml_sigmoid(ctx0, g);
    } else {
        g = ggml_silu(ctx0, g);
    }
    if (n_head_kv != 0 && n_head_kv != n_head) {
        GGML_ASSERT(n_head % n_head_kv == 0);
        k                 = ggml_reshape_4d(ctx0, k, head_size, 1, n_head_kv, n_tokens);
        v                 = ggml_reshape_4d(ctx0, v, head_size, 1, n_head_kv, n_tokens);
        ggml_tensor * tmp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_size, n_head / n_head_kv, n_head_kv, n_tokens);
        k                 = ggml_repeat(ctx0, k, tmp);
        v                 = ggml_repeat(ctx0, v, tmp);
    }
    k = ggml_reshape_3d(ctx0, k, head_size, n_head, n_tokens);
    v = ggml_reshape_3d(ctx0, v, head_size, n_head, n_tokens);
    r = ggml_reshape_3d(ctx0, r, head_size, n_head, n_tokens);

    ggml_tensor * w =
        ggml_mul_mat(ctx0, layer.time_mix_decay_w2, ggml_tanh(ctx0, ggml_mul_mat(ctx0, layer.time_mix_decay_w1, xw)));

    w = ggml_add(ctx0, w, layer.time_mix_decay);
    w = ggml_exp(ctx0, ggml_neg(ctx0, ggml_exp(ctx0, w)));
    w = ggml_reshape_3d(ctx0, w, head_size, n_head, n_tokens);

    if (is_qrwkv) {
        // k = k * (1 - w)
        k = ggml_sub(ctx0, k, ggml_mul(ctx0, k, w));
    }
    ggml_tensor * wkv_state = build_rs(inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);

    ggml_tensor * wkv_output;
    if (is_qrwkv) {
        wkv_output = ggml_gated_linear_attn(ctx0, k, v, r, w, wkv_state, pow(head_size, -0.5f));
    } else {
        wkv_output = ggml_rwkv_wkv6(ctx0, k, v, r, layer.time_mix_first, w, wkv_state);
    }
    cur       = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
    wkv_state = ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

    ggml_build_forward_expand(
        gf, ggml_cpy(ctx0, wkv_state,
                     ggml_view_1d(ctx0, mctx_cur->get_s_l(il), hparams.n_embd_s() * n_seqs,
                                  hparams.n_embd_s() * kv_head * ggml_element_size(mctx_cur->get_s_l(il)))));

    if (!is_qrwkv) {
        // group norm with head_count groups
        cur = ggml_reshape_3d(ctx0, cur, n_embd / n_head, n_head, n_tokens);
        cur = ggml_norm(ctx0, cur, 64e-5f);

        // Convert back to regular vectors.
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.time_mix_ln), layer.time_mix_ln_b);
    } else {
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    }
    cur = ggml_mul(ctx0, cur, g);
    cur = build_lora_mm(layer.time_mix_output, cur);

    return ggml_reshape_3d(ctx0, cur, n_embd, n_seq_tokens, n_seqs);
}
