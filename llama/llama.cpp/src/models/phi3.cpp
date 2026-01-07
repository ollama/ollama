#include "models.h"

template<bool iswa>
llm_build_phi3<iswa>::llm_build_phi3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>;
    inp_attn_type * inp_attn = nullptr;

    if constexpr (iswa) {
        inp_attn = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        auto * residual = inpL;

        // self-attention
        {
            // rope freq factors for 128k context
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            ggml_tensor* attn_norm_output = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM_RMS, il);
            cb(attn_norm_output, "attn_norm", il);

            ggml_tensor * Qcur = nullptr;
            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;

            if (model.layers[il].wqkv) {
                cur = build_lora_mm(model.layers[il].wqkv, attn_norm_output);
                cb(cur, "wqkv", il);

                Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head * sizeof(float), cur->nb[1], 0 * sizeof(float) * (n_embd));
                Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float), cur->nb[1], 1 * sizeof(float) * (n_embd));
                Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float), cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa));
                }
                else {
                Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, attn_norm_output), model.layers[il].bv);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            }
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
            cb(Qcur, "Qcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }
        cur = ggml_add(ctx0, cur, residual);
        residual = cur;

        cur = build_norm(cur,
                model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, residual, cur);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = build_norm(inpL,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    if (model.output_b != nullptr) {
        cb(cur, "result_output_no_bias", -1);
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Explicit template instantiations
template struct llm_build_phi3<false>;
template struct llm_build_phi3<true>;
