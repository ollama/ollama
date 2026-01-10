#include "models.h"



llm_build_nemotron_h::llm_build_nemotron_h(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context_mamba(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            // ssm layer //
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        } else if (hparams.n_ff(il) == 0) {
            // attention layer //
            cur = build_attention_layer(cur, inp->get_attn(), model, n_embd_head, il);
        } else {
            cur = build_ffn_layer(cur, model, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // add residual
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "nemotron_h_block_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_nemotron_h::build_attention_layer(ggml_tensor *             cur,
                                                          llm_graph_input_attn_kv * inp_attn,
                                                          const llama_model &       model,
                                                          const int64_t             n_embd_head,
                                                          const int                 il) {
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

    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il), n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(il), n_tokens);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    cur = build_attn(inp_attn,
            model.layers[il].wo, model.layers[il].bo,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_out", il);
    return cur;
}

ggml_tensor * llm_build_nemotron_h::build_ffn_layer(ggml_tensor * cur, const llama_model & model, const int il) {
    if (model.layers[il].ffn_gate_inp == nullptr) {
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                NULL,                      NULL,                        NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
    } else {
        ggml_tensor * ffn_inp = cur;
        ggml_tensor * moe_out =
            build_moe_ffn(ffn_inp,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    nullptr, // no gate
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_RELU_SQR, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                    il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(ffn_inp,
                    model.layers[il].ffn_up_shexp,  NULL, NULL,
                    NULL /* no gate */           ,  NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    }

    cur = build_cvec(cur, il);
    cb(cur, "l_out", il);

    return cur;
}
