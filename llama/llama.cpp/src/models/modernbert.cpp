#include "models.h"
#include "../llama-impl.h"
#include <stdexcept>

llm_build_modernbert::llm_build_modernbert(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
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
    cb(inpL, "tok_embd_lookup", -1);

    // For ModernBERT, mark embedding result as output to prevent buffer reuse
    if (model.arch == LLM_ARCH_MODERNBERT) {
        ggml_set_output(inpL);
    }

    // token types are hardcoded to zero ("Sentence A")
    if (model.type_embd) {
        ggml_tensor * type_row0 = ggml_view_1d(ctx0, model.type_embd, n_embd, 0);
        if (model.arch == LLM_ARCH_MODERNBERT) {
            // For ModernBERT, force explicit copy of operands to avoid memory aliasing
            ggml_tensor* inpL_copy = ggml_dup_tensor(ctx0, inpL);
            inpL_copy = ggml_cpy(ctx0, inpL, inpL_copy);
            inpL_copy->flags |= GGML_TENSOR_FLAG_OUTPUT;

            ggml_tensor* type_row0_copy = ggml_dup_tensor(ctx0, type_row0);
            type_row0_copy = ggml_cpy(ctx0, type_row0, type_row0_copy);
            type_row0_copy->flags |= GGML_TENSOR_FLAG_OUTPUT;

            inpL = ggml_add(ctx0, inpL_copy, type_row0_copy);
        } else {
            inpL = ggml_add(ctx0, inpL, type_row0);
        }
    }
    if (model.arch == LLM_ARCH_BERT) {
        inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
    }
    cb(inpL, "inp_embd", -1);

    // Protect embeddings before norm for ModernBERT
    if (model.arch == LLM_ARCH_MODERNBERT) {
        ggml_set_output(inpL);
    }

    // embed layer norm
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);
    cb(inpL, "inp_norm", -1);

    // Protect normalized embeddings for ModernBERT
    if (model.arch == LLM_ARCH_MODERNBERT) {
        ggml_set_output(inpL);
    }

    auto * inp_attn = build_attn_inp_no_cache();

    ggml_tensor * inp_out_ids = nullptr;

    // ModernBERT: Check if we need alternating attention pattern
    const bool use_alternating_attn = (model.arch == LLM_ARCH_MODERNBERT &&
                                       hparams.global_attn_every_n_layers > 0);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * cur = inpL;

        // PRE-NORM: Apply attn_norm BEFORE attention computation
        // ModernBERT layer 0 has no attn_norm (acts as identity), layers 1-21 have it
        ggml_tensor * attn_residual_base = cur;  // Save unnormalized input for residual add
        if (model.arch == LLM_ARCH_MODERNBERT) {
            if (il == 0) {
                // Layer 0: No attn_norm tensor (identity/no-op in HuggingFace)
            } else {
                // Layers 1-21: Apply normalization BEFORE attention
                cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);
                cb(cur, "attn_norm", il);
            }
        }

        {
            ggml_tensor * Qcur;
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;

            // self-attention
            if (model.arch == LLM_ARCH_MODERNBERT) {
                if (!model.layers[il].wqkv && (!model.layers[il].wq || !model.layers[il].wk || !model.layers[il].wv)) {
                    throw std::runtime_error("ModernBERT layer " + std::to_string(il) + " missing attention weight tensors");
                }
                if (!model.layers[il].wo) {
                    throw std::runtime_error("ModernBERT layer " + std::to_string(il) + " missing attention output tensor (wo)");
                }
            }

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
                model.arch == LLM_ARCH_JINA_BERT_V3 || model.arch == LLM_ARCH_MODERNBERT) {

                // Get per-layer RoPE frequency for ModernBERT (global vs local)
                const float freq_base_l  = model.arch == LLM_ARCH_MODERNBERT ? model.get_rope_freq_base(cparams, il)  : freq_base;
                const float freq_scale_l = model.arch == LLM_ARCH_MODERNBERT ? model.get_rope_freq_scale(cparams, il) : freq_scale;

                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
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

        // PRE-NORM: Add attention output to UNNORMALIZED input
        if (model.arch == LLM_ARCH_MODERNBERT) {
            ggml_set_output(cur);
            ggml_set_output(attn_residual_base);
            cur = ggml_add(ctx0, cur, attn_residual_base);
            ggml_set_output(cur);
        } else {
            cur = ggml_add(ctx0, cur, inpL);
        }

        // PRE-NORM: Save the value BEFORE mlp_norm for FFN residual add
        ggml_tensor * ffn_residual_base = cur;

        // PRE-NORM: Apply mlp_norm BEFORE FFN computation (for ModernBERT)
        if (model.arch == LLM_ARCH_MODERNBERT) {
            if (model.layers[il].layer_out_norm) {
                cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);
                cb(cur, "mlp_norm", il);
            }
        }

        if (model.layers[il].attn_norm_2 != nullptr) {
            cur = ggml_add(ctx0, cur, inpL);
            cur = build_norm(cur, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, il);
        }

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        if (hparams.moe_every_n_layers > 0 && il % hparams.moe_every_n_layers == 1) {
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
        } else if (model.arch == LLM_ARCH_MODERNBERT) {
            // ModernBERT uses GeGLU (Gated GELU) activation with no bias terms
            if (model.layers[il].ffn_gate == nullptr || model.layers[il].ffn_up == nullptr || model.layers[il].ffn_down == nullptr) {
                throw std::runtime_error("ModernBERT layer " + std::to_string(il) + " missing required FFN tensors");
            }
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL, NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
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

        // Protect FFN residual add operands for ModernBERT
        if (model.arch == LLM_ARCH_MODERNBERT) {
            ggml_set_output(cur);
            ggml_set_output(ffn_residual_base);
        }

        // Add FFN output to residual base
        if (model.arch == LLM_ARCH_MODERNBERT) {
            cur = ggml_add(ctx0, cur, ffn_residual_base);
        } else {
            cur = ggml_add(ctx0, cur, ffn_inp);
        }

        if (model.arch == LLM_ARCH_MODERNBERT) {
            ggml_set_output(cur);
        }

        // output layer norm (not for ModernBERT which uses PRE-NORM)
        if (model.arch != LLM_ARCH_MODERNBERT) {
            cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);
        }

        // input for next layer
        inpL = cur;

        // Protect layer outputs for ModernBERT
        if (model.arch == LLM_ARCH_MODERNBERT) {
            ggml_set_output(inpL);
        }
    }

    cur = inpL;

    // ModernBERT applies final_norm (output_norm) after all encoder layers
    if (model.output_norm) {
        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);
        cb(cur, "result_norm", -1);
    }

    // Apply L2 normalization if requested (for sentence-transformers models)
    if (model.arch == LLM_ARCH_MODERNBERT && hparams.normalize_embeddings) {
        cur = ggml_l2_norm(ctx0, cur, 1e-12f);
        cb(cur, "result_l2_norm", -1);
    }

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
