#include "models.h"



llm_build_bert::llm_build_bert(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;
    ggml_tensor * inp_pos = nullptr;

    if (model.arch != LLM_ARCH_JINA_BERT_V2) {
        inp_pos = build_inp_pos();
    }

    // construct input embeddings (token, type, position)
    inpL = build_inp_embd(model.tok_embd);

    // token types are hardcoded to zero ("Sentence A")
    if (model.type_embd) {
        ggml_tensor * type_row0 = ggml_view_1d(ctx0, model.type_embd, n_embd, 0);
        inpL                    = ggml_add(ctx0, inpL, type_row0);
    }
    if (model.arch == LLM_ARCH_BERT) {
        inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
    }
    cb(inpL, "inp_embd", -1);

    // embed layer norm
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);
    cb(inpL, "inp_norm", -1);

    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * cur = inpL;

        {
            ggml_tensor * Qcur;
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;

            // self-attention
            if (model.layers[il].wqkv) {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                if (model.layers[il].bqkv) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
                }

                Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, n_embd_head * sizeof(float), cur->nb[1],
                                    0 * sizeof(float) * (n_embd));
                Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], 1 * sizeof(float) * (n_embd));
                Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa));
            } else {
                Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, cur), model.layers[il].bq);
                Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, cur), model.layers[il].bk);
                Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, cur), model.layers[il].bv);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            }

            if (model.layers[il].attn_q_norm) {
                Qcur = ggml_reshape_2d(ctx0, Qcur, n_embd_head * n_head, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, model.layers[il].attn_q_norm_b, LLM_NORM, il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            }

            if (model.layers[il].attn_k_norm) {
                Kcur = ggml_reshape_2d(ctx0, Kcur, n_embd_head * n_head_kv, n_tokens);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, model.layers[il].attn_k_norm_b, LLM_NORM, il);

                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            }

            // RoPE
            if (model.arch == LLM_ARCH_NOMIC_BERT || model.arch == LLM_ARCH_NOMIC_BERT_MOE ||
                model.arch == LLM_ARCH_JINA_BERT_V3) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // re-add the layer input
        cur = ggml_add(ctx0, cur, inpL);

        // attention layer norm
        cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);

        if (model.layers[il].attn_norm_2 != nullptr) {
            cur = ggml_add(ctx0, cur, inpL);  // re-add the layer input
            cur = build_norm(cur, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, il);
        }

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        if (hparams.moe_every_n_layers > 0 && il % hparams.moe_every_n_layers == 1) {
            // MoE branch
            cur = build_moe_ffn(cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps, nullptr,
                                model.layers[il].ffn_down_exps, nullptr, hparams.n_expert, hparams.n_expert_used,
                                LLM_FFN_GELU, false, false, 0.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
            cb(cur, "ffn_moe_out", il);
        } else if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_NOMIC_BERT_MOE ||
                   model.arch == LLM_ARCH_JINA_BERT_V3) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);
        } else if (model.arch == LLM_ARCH_JINA_BERT_V2) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL,
                    model.layers[il].ffn_gate ? LLM_FFN_GELU : LLM_FFN_GEGLU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        // attentions bypass the intermediate layer
        cur = ggml_add(ctx0, cur, ffn_inp);

        // output layer norm
        cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
