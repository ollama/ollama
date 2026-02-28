#include "ggml.h"
#include "models.h"

#define CHUNK_SIZE 64

llm_build_qwen35moe::llm_build_qwen35moe(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context_mamba(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * causal_mask =
        ggml_tri(ctx0, ggml_fill_inplace(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, CHUNK_SIZE, CHUNK_SIZE), 1.0f),
                     GGML_TRI_TYPE_LOWER);

    ggml_tensor * identity = ggml_diag(ctx0, ggml_fill_inplace(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, CHUNK_SIZE), 1.0f));
    ggml_tensor * diag_mask = ggml_add(ctx0, causal_mask, identity);

    ggml_build_forward_expand(gf, causal_mask);
    ggml_build_forward_expand(gf, identity);
    ggml_build_forward_expand(gf, diag_mask);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, causal_mask, identity, diag_mask, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // MOE FFN layer
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_qwen35moe::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    // Qwen3.5 MoE uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur);
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE (multi-dim RoPE with sections)
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llm_build_qwen35moe::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        ggml_tensor *        causal_mask,
        ggml_tensor *        identity,
        ggml_tensor *        diag_mask,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, cur);
    cb(z, "z", il);

    ggml_tensor * beta_raw = build_lora_mm(model.layers[il].ssm_beta, cur);
    cb(beta_raw, "beta_raw", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur);
    alpha = ggml_cont_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);
    cb(gate, "gate", il);

    // gate stays 3D: [num_v_heads, n_seq_tokens, n_seqs]

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    // Update convolution state cache
    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
        ggml_view_1d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels * n_seqs,
                     kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim * num_v_heads, 1, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * beta = ggml_cont_4d(ctx0, beta_raw, num_v_heads, 1, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    // Choose between chunking and autoregressive based on n_tokens
    ggml_tensor * attn_out;
    if (n_seq_tokens == 1) {
        attn_out = build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il);
    } else {
        attn_out = build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, causal_mask, identity, diag_mask, il);
    }
    cb(attn_out, "attn_out", il);

    // The tensors were concatenated 1d, so we need to extract them 1d as well
    const int64_t output_flat_size = head_v_dim * num_v_heads * n_seq_tokens * n_seqs;
    ggml_tensor * attn_out_1d      = ggml_view_1d(ctx0, attn_out, output_flat_size, 0);
    cb(attn_out_1d, "attn_out_1d", il);

    ggml_tensor * attn_out_final = ggml_cont_4d(ctx0, attn_out_1d, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    cb(attn_out_final, "attn_out_reshaped", il);

    // Extract the state part (second part of the concatenated tensor)
    const int64_t state_flat_size = head_v_dim * head_v_dim * num_v_heads * n_seqs;

    ggml_tensor * state_1d =
        ggml_view_1d(ctx0, attn_out, state_flat_size, output_flat_size * ggml_element_size(attn_out));
    cb(state_1d, "state_1d", il);

    // Update the recurrent states
    ggml_build_forward_expand(gf,
                              ggml_cpy(ctx0, state_1d,
                                       ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                                                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    GGML_ASSERT(ggml_nelements(attn_out_1d) + ggml_nelements(state_1d) == ggml_nelements(attn_out));

    // z: reshape for gated normalization
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization
    ggml_tensor * attn_out_norm = build_norm_gated(attn_out_final, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_build_qwen35moe::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_qwen35moe::build_layer_ffn(ggml_tensor * cur, const int il) {
    // MoE FFN layer
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used, LLM_FFN_SILU,
            true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
    cb(moe_out, "ffn_moe_out", il);

    // Add shared experts if present
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, NULL,
                model.layers[il].ffn_gate_shexp, NULL, NULL,
                model.layers[il].ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // Apply shared expert gating
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

ggml_tensor * llm_build_qwen35moe::build_delta_net_chunking(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * beta,
        ggml_tensor * state,
        ggml_tensor * causal_mask,
        ggml_tensor * identity,
        ggml_tensor * diag_mask,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v * H_v && state->ne[2] == 1 && state->ne[3] == n_seqs);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);

    GGML_ASSERT(H_k == H_v);

    const float eps_norm = hparams.f_norm_rms_eps;

    q = ggml_l2_norm(ctx0, q, eps_norm);
    k = ggml_l2_norm(ctx0, k, eps_norm);

    const float scale = 1.0f / sqrtf(S_v);

    q = ggml_scale(ctx0, q, scale);

    beta = ggml_sigmoid(ctx0, beta);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(beta, "beta_in", il);
    cb(g, "g_in", il);

    q = ggml_cont_4d(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
    k = ggml_cont_4d(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
    v = ggml_cont_4d(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
    g = ggml_cont_4d(ctx0, ggml_permute(ctx0, g, 2, 0, 3, 1), n_tokens, 1, H_k, n_seqs);

    beta  = ggml_cont(ctx0, ggml_permute(ctx0, beta, 2, 0, 1, 3));
    state = ggml_reshape_4d(ctx0, state, S_v, S_v, H_v, n_seqs);

    cb(q, "q_perm", il);
    cb(k, "k_perm", il);
    cb(v, "v_perm", il);
    cb(beta, "beta_perm", il);
    cb(g, "g_perm", il);
    cb(state, "state_in", il);

    GGML_ASSERT(q->ne[1] == n_tokens && q->ne[0] == S_k && q->ne[2] == H_k && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[1] == n_tokens && k->ne[0] == S_k && k->ne[2] == H_k && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[1] == n_tokens && v->ne[0] == S_v && v->ne[2] == H_k && v->ne[3] == n_seqs);
    GGML_ASSERT(beta->ne[1] == n_tokens && beta->ne[2] == H_k && beta->ne[0] == 1 && beta->ne[3] == n_seqs);

    const int64_t chunk_size = CHUNK_SIZE;

    const int64_t pad = (chunk_size - n_tokens % chunk_size) % chunk_size;
    const int64_t n_chunks = (n_tokens + pad) / chunk_size;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, pad, 0, 0, 0);
    beta = ggml_pad(ctx0, beta, 0, pad, 0, 0);

    cb(q, "q_pad", il);
    cb(k, "k_pad", il);
    cb(v, "v_pad", il);
    cb(beta, "beta_pad", il);
    cb(g, "g_pad", il);

    ggml_tensor * v_beta = ggml_mul(ctx0, v, beta);
    ggml_tensor * k_beta = ggml_mul(ctx0, k, beta);

    cb(v_beta, "v_beta", il);
    cb(k_beta, "k_beta", il);

    q      = ggml_reshape_4d(ctx0, q,      S_k, chunk_size, n_chunks, H_k * n_seqs);
    k      = ggml_reshape_4d(ctx0, k,      S_k, chunk_size, n_chunks, H_k * n_seqs);
    k_beta = ggml_reshape_4d(ctx0, k_beta, S_k, chunk_size, n_chunks, H_k * n_seqs);
    v      = ggml_reshape_4d(ctx0, v,      S_v, chunk_size, n_chunks, H_v * n_seqs);
    v_beta = ggml_reshape_4d(ctx0, v_beta, S_v, chunk_size, n_chunks, H_v * n_seqs);

    g    = ggml_reshape_4d(ctx0, g, chunk_size, 1, n_chunks, H_k * n_seqs);
    beta = ggml_reshape_4d(ctx0, beta, 1, chunk_size, n_chunks, H_k * n_seqs);

    ggml_tensor * g_cumsum = ggml_cumsum(ctx0, g);

    cb(g_cumsum, "g_cumsum", il);

    ggml_tensor * gcs_i = ggml_reshape_4d(ctx0, g_cumsum, chunk_size, 1, n_chunks, H_v * n_seqs);
    ggml_tensor * gcs_j = ggml_reshape_4d(ctx0, g_cumsum, 1, chunk_size, n_chunks, H_v * n_seqs);

    ggml_tensor * gcs_j_broadcast =
        ggml_repeat_4d(ctx0, gcs_j, chunk_size, chunk_size, n_chunks, H_v * n_seqs);

    ggml_tensor * decay_mask = ggml_sub(ctx0, gcs_j_broadcast, gcs_i);

    cb(decay_mask, "decay_mask", il);

    decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);
    decay_mask = ggml_exp(ctx0, decay_mask);
    decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);

    ggml_tensor * kmulkbeta = ggml_mul_mat(ctx0, k, k_beta);

    ggml_tensor * k_decay = ggml_mul(ctx0, kmulkbeta, decay_mask);
    ggml_tensor * attn    = ggml_neg(ctx0, ggml_mul(ctx0, k_decay, causal_mask));

    cb(attn, "attn_pre_solve", il);

    ggml_tensor * attn_lower = ggml_mul(ctx0, attn, causal_mask);
    ggml_tensor * lhs        = ggml_sub(ctx0, ggml_repeat(ctx0, identity, attn_lower), attn_lower);

    ggml_tensor * lin_solve  = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn                     = ggml_mul(ctx0, lin_solve, causal_mask);
    attn                     = ggml_add(ctx0, attn, identity);

    cb(attn, "attn_solved", il);

    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_beta)), attn);

    ggml_tensor * g_cumsum_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_cumsum));
    ggml_tensor * gexp       = ggml_exp(ctx0, g_cumsum_t);

    ggml_tensor * kbeta_gexp = ggml_mul(ctx0, k_beta, gexp);

    cb(kbeta_gexp, "kbeta_gexp", il);

    ggml_tensor * k_cumdecay =
        ggml_cont(ctx0, ggml_transpose(ctx0, ggml_mul_mat(ctx0, attn, ggml_cont(ctx0, ggml_transpose(ctx0, kbeta_gexp)))));

    cb(k_cumdecay, "k_cumdecay", il);

    ggml_tensor * core_attn_out = nullptr;
    ggml_tensor * new_state = ggml_dup(ctx0, state);

    cb(new_state, "new_state", il);

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        auto chunkify = [=](ggml_tensor * t) {
            return ggml_cont(ctx0, ggml_view_4d(ctx0, t, t->ne[0], chunk_size, 1, t->ne[3],
                                     t->nb[1], t->nb[2], t->nb[3], t->nb[2] * chunk));
        };

        auto chunkify_g = [=](ggml_tensor * t) {
            return ggml_cont(ctx0, ggml_view_4d(ctx0, t, chunk_size, t->ne[1], 1, t->ne[3],
                                     t->nb[1], t->nb[2], t->nb[3], t->nb[2] * chunk));
        };

        ggml_tensor * k_chunk = chunkify(k);
        ggml_tensor * q_chunk = chunkify(q);
        ggml_tensor * v_chunk = chunkify(v);

        ggml_tensor * g_cs_chunk = chunkify_g(g_cumsum);
        ggml_tensor * g_cs_chunk_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_cs_chunk));

        ggml_tensor * decay_mask_chunk = chunkify(decay_mask);
        ggml_tensor * k_cumdecay_chunk = chunkify(k_cumdecay);

        ggml_tensor * gexp_chunk = ggml_exp(ctx0, g_cs_chunk_t);

        attn = ggml_mul_mat(ctx0, k_chunk, q_chunk);
        attn = ggml_mul(ctx0, attn, decay_mask_chunk);
        attn = ggml_mul(ctx0, attn, diag_mask);

        ggml_tensor * state_t = ggml_cont_4d(ctx0, ggml_permute(ctx0, new_state, 1, 0, 2, 3), S_v, S_v, 1, H_v * n_seqs);

        ggml_tensor * v_prime = ggml_mul_mat(ctx0, state_t, k_cumdecay_chunk);

        ggml_tensor * v_new = ggml_sub(ctx0, ggml_repeat(ctx0, v_chunk, v_prime), v_prime);
        ggml_tensor * v_new_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_new));

        ggml_tensor * q_g_exp    = ggml_mul(ctx0, q_chunk, gexp_chunk);
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, state_t, q_g_exp);

        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_new_t, attn);

        ggml_tensor * core_attn_out_chunk = ggml_add(ctx0, attn_inter, v_attn);

        core_attn_out = core_attn_out == nullptr ? core_attn_out_chunk : ggml_concat(ctx0, core_attn_out, core_attn_out_chunk, 1);

        ggml_tensor * g_cum_last =
            ggml_cont(ctx0, ggml_view_4d(ctx0, g_cs_chunk_t, g_cs_chunk_t->ne[0], 1, g_cs_chunk_t->ne[2], g_cs_chunk_t->ne[3],
                                         g_cs_chunk_t->nb[1], g_cs_chunk_t->nb[2], g_cs_chunk_t->nb[3],
                                         g_cs_chunk_t->nb[0] * (g_cs_chunk_t->ne[1] - 1)));

        ggml_tensor * gexp_last =
            ggml_reshape_4d(ctx0, ggml_exp(ctx0, g_cum_last), 1, 1, g_cum_last->ne[0] * g_cum_last->ne[2], g_cum_last->ne[3]);

        ggml_tensor * g_cum_last_3d =
            ggml_reshape_3d(ctx0, g_cum_last, g_cum_last->ne[0], g_cum_last->ne[2], g_cum_last->ne[3]);

        ggml_tensor * g_cumsum_3d = ggml_reshape_3d(ctx0, g_cs_chunk, g_cs_chunk->ne[0], g_cs_chunk->ne[2], g_cs_chunk->ne[3]);

        ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cumsum_3d, g_cum_last_3d));

        ggml_tensor * g_diff_exp = ggml_exp(ctx0, g_diff);

        ggml_tensor * key_gdiff = ggml_mul(ctx0, k_chunk,
                                        ggml_reshape_4d(ctx0, g_diff_exp, 1, g_diff_exp->ne[0], g_diff_exp->ne[1],
                                                                         g_diff_exp->ne[2] * g_diff_exp->ne[3]));

        ggml_tensor * kgdmulvnew = ggml_mul_mat(ctx0, v_new_t, ggml_cont(ctx0, ggml_transpose(ctx0, key_gdiff)));

        new_state = ggml_add(ctx0,
            ggml_mul(ctx0, new_state, ggml_reshape_4d(ctx0, gexp_last, gexp_last->ne[0], gexp_last->ne[1], H_v, n_seqs)),
                     ggml_reshape_4d(ctx0, kgdmulvnew, kgdmulvnew->ne[0], kgdmulvnew->ne[1], H_v, n_seqs));
    }

    core_attn_out = ggml_cont_4d(ctx0, core_attn_out, S_v, chunk_size * n_chunks, H_v, n_seqs);

    ggml_tensor * output_tokens = ggml_view_4d(ctx0, core_attn_out, S_v, n_tokens, H_v, n_seqs, core_attn_out->nb[1], core_attn_out->nb[2], core_attn_out->nb[3], 0);
    cb(output_tokens, "output_tokens", il);

    ggml_tensor * flat_output =
        ggml_cont_1d(ctx0, ggml_permute(ctx0, output_tokens, 0, 2, 1, 3), S_v * H_v * n_tokens * n_seqs);

    ggml_tensor * flat_state = ggml_cont_1d(ctx0, new_state, S_v * S_v * H_v * n_seqs);

    return ggml_concat(ctx0, flat_output, flat_state, 0);
}

ggml_tensor * llm_build_qwen35moe::build_delta_net_autoregressive(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * beta,
        ggml_tensor * state,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_tokens == 1);
    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v * H_v && state->ne[2] == 1 && state->ne[3] == n_seqs);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);

    GGML_ASSERT(H_k == H_v);

    const float eps_norm = hparams.f_norm_rms_eps;

    q = ggml_l2_norm(ctx0, q, eps_norm);
    k = ggml_l2_norm(ctx0, k, eps_norm);

    const float scale = 1.0f / sqrtf(S_v);

    q    = ggml_scale(ctx0, q, scale);
    beta = ggml_sigmoid(ctx0, beta);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(beta, "beta_in", il);
    cb(g, "g_in", il);

    state = ggml_reshape_4d(ctx0, state, S_v, S_v, H_v, n_seqs);

    ggml_tensor * g_t    = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, g), 1, 1, H_k, n_seqs);
    ggml_tensor * beta_t = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, beta), 1, 1, H_k, n_seqs);

    g_t = ggml_exp(ctx0, g_t);

    state = ggml_mul(ctx0, state, g_t);

    ggml_tensor * k_t_unsqueezed = ggml_reshape_4d(ctx0, k, 1, S_v, H_v, n_seqs);
    ggml_tensor * kv_mem         = ggml_mul(ctx0, state, k_t_unsqueezed);
    kv_mem = ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, kv_mem))));

    ggml_tensor * v_t    = ggml_reshape_4d(ctx0, v, S_v, 1, H_v, n_seqs);
    ggml_tensor * v_diff = ggml_sub(ctx0, v_t, kv_mem);
    ggml_tensor * delta  = ggml_mul(ctx0, v_diff, beta_t);

    ggml_tensor * k_t_delta = ggml_mul(ctx0, ggml_repeat_4d(ctx0, k_t_unsqueezed, S_v, S_v, H_v, n_seqs), delta);
    state                   = ggml_add(ctx0, state, k_t_delta);

    ggml_tensor * q_t_unsqueezed = ggml_reshape_4d(ctx0, q, 1, S_v, H_v, n_seqs);
    ggml_tensor * state_q        = ggml_mul(ctx0, state, q_t_unsqueezed);
    ggml_tensor * core_attn_out =
        ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, state_q))));

    cb(core_attn_out, "output_tokens", il);
    cb(state, "new_state", il);

    ggml_tensor * flat_output = ggml_reshape_1d(ctx0, core_attn_out, S_v * H_v * n_tokens * n_seqs);
    ggml_tensor * flat_state  = ggml_reshape_1d(ctx0, state, S_v * S_v * H_v * n_seqs);

    return ggml_concat(ctx0, flat_output, flat_state, 0);
}
