#include "models.h"

llm_build_jamba::llm_build_jamba(const llama_model & model, const llm_graph_params & params) : llm_graph_context_mamba(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    auto * inp_hybrid = build_inp_mem_hybrid();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_head_kv = hparams.n_head_kv(il);

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (n_head_kv == 0) {
            cur = build_mamba_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);
        } else {
            // Attention

            struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // No RoPE :)
            cur = build_attn(inp_hybrid->get_attn(),
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, NULL, NULL, NULL, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        // residual
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, inpL, cur);
        cb(cur, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // FFN
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
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
                    LLM_FFN_SILU, false,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
        }
        // residual
        cur = ggml_add(ctx0, ffn_inp, cur);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    // final rmsnorm
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
