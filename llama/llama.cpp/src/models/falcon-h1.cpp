#include "models.h"

llm_build_falcon_h1::llm_build_falcon_h1(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // Build the inputs in the recurrent & kv cache
    auto * inp = build_inp_mem_hybrid();

    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, hparams.rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);

        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, hparams.rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);

        cb(Qcur, "Qcur-post-rope", il);
        cb(Kcur, "Kcur-post-rope", il);
        cb(Vcur, "Vcur-post-rope", il);

        ggml_tensor * attn_out = build_attn(inp->get_attn(),
                                    model.layers[il].wo, NULL,
                                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(attn_out, "attn_out", il);

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        // Mamba2 layer
        cb(cur, "ssm_in", il);

        ggml_tensor * ssm_out = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        cb(ssm_out, "ssm_out", il);

        // // Aggregation
        cur   = ggml_add(ctx0, attn_out, ssm_out);
        inpSA = ggml_add(ctx0, cur, inpSA);
        cb(cur, "layer_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = inpSA;
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, inpSA);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

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
