#include "models.h"

llm_build_afmoe::llm_build_afmoe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // MuP scaling: embeddings * sqrt(hidden_size)
    // mup_enabled = true, hidden_size = 1024, scale = 32.0
    inpL = ggml_scale(ctx0, inpL, sqrtf(float(n_embd)));
    cb(inpL, "inp_embd_scaled", -1);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // dual attention normalization (pre)
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * attn_inp = cur;  // save input for gate computation

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            // compute gate from input
            ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, attn_inp);
            cb(gate, "attn_gate_proj", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            // Q/K normalization
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            // RoPE only for sliding_attention layers
            const bool use_rope = hparams.n_no_rope_layer_step > 0 &&
                                ((il + 1) % hparams.n_no_rope_layer_step) != 0;
            if (use_rope) {
                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur_rope", il);
            }

            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            cur = build_attn(inp_attn,
                    NULL, NULL,  // wo will be applied after gating
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // attention gating: attn_out * sigmoid(gate) BEFORE o_proj
            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate_sig", il);
            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", il);

            // now apply output projection
            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_o_proj", il);
        }

        // dual attention normalization (post)
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // dual ffn normalization (pre)
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // MoE or dense FFN
        if ((uint32_t)il >= hparams.n_layer_dense_lead) {
            // MoE layer with sigmoid routing, normalization, and scaling
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,           // norm_w (route_norm=True)
                    hparams.expert_weights_scale,          // scale_w
                    hparams.expert_weights_scale,          // w_scale (route_scale=2.826)
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert
            if (hparams.n_expert_shared > 0) {
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
        } else {
            // dense layer
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        // dual ffn normalization (post)
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

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
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
