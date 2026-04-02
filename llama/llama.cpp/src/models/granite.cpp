#include "models.h"


llm_build_granite::llm_build_granite(
    const llama_model & model,
    const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - built only if rope enabled
    ggml_tensor * inp_pos = nullptr;
    if (hparams.rope_finetuned) {
        inp_pos = build_inp_pos();
    }
    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        cur = build_attention_layer(
            cur, inp_pos, inp_attn,
            model, n_embd_head, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        // ffn
        cur = build_layer_ffn(cur, inpSA, model, il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    // For Granite architectures - scale logits
    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_granite::build_attention_layer(
          ggml_tensor             * cur,
          ggml_tensor             * inp_pos,
          llm_graph_input_attn_kv * inp_attn,
    const llama_model             & model,
    const int64_t                 n_embd_head,
    const int                     il) {

    // compute Q and K and (optionally) RoPE them
    ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
    cb(Qcur, "Qcur", il);
    if (model.layers[il].bq) {
        Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
        cb(Qcur, "Qcur", il);
    }

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);
    if (model.layers[il].bk) {
        Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
        cb(Kcur, "Kcur", il);
    }

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);
    if (model.layers[il].bv) {
        Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
        cb(Vcur, "Vcur", il);
    }

    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il),    n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(il), n_tokens);

    const bool use_rope = hparams.rope_finetuned;
    if (use_rope) {
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
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
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    cur = build_attn(inp_attn,
            model.layers[il].wo, model.layers[il].bo,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
    return cur;
}

ggml_tensor * llm_build_granite::build_layer_ffn(
          ggml_tensor       * cur,
          ggml_tensor       * inpSA,
    const llama_model       & model,
    const int                 il) {

    // For Granite architectures - scale residual
    if (hparams.f_residual_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
    }
    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "ffn_inp", il);

    // feed-forward network (non-MoE)
    if (model.layers[il].ffn_gate_inp == nullptr) {

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);

    } else {
        // MoE branch
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
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
        cb(moe_out, "ffn_moe_out", il);

        // For Granite MoE Shared
        if (hparams.n_ff_shexp > 0) {
            ggml_tensor * ffn_shexp = build_ffn(cur,
                model.layers[il].ffn_up_shexp,   NULL, NULL,
                model.layers[il].ffn_gate_shexp, NULL, NULL,
                model.layers[il].ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
    }

    // For Granite architectures - scale residual
    if (hparams.f_residual_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
    }
    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "ffn_out", il);

    cur = build_cvec(cur, il);
    cb(cur, "l_out", il);

    return cur;
}
