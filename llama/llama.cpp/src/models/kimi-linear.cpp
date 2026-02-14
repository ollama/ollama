#include "models.h"
#include "ggml.h"

#define CHUNK_SIZE 64

// Causal Conv1d function for Q,K,V
// When qkv is 0, it is Q, 1 is K, 2 is V
static ggml_tensor * causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w, int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total = 3 * conv_state_size;  // Q + K + V

    // conv_state_all is [n_embd_r_total, n_seqs], split into Q, K, V
    // Each conv state is [(d_conv-1) * d_inner] per sequence, need to reshape to [d_conv-1, d_inner, n_seqs]
    // Memory layout: for each seq, Q state is first conv_state_size elements, then K, then V
    // conv_state_all has stride: nb[0] = element_size, nb[1] = n_embd_r_total * element_size
    // View Q conv state: offset 0, size conv_state_size per seq
    // conv_state_all is [n_embd_r_total, n_seqs] with memory layout:
    //   state[i + seq * n_embd_r_total] where i = conv_step + channel * (d_conv-1) + {0, conv_state_size, 2*conv_state_size} for Q/K/V
    // We want [d_conv-1, d_inner, n_seqs] view:
    //   nb1 = (d_conv-1) * element_size (stride between channels)
    //   nb2 = n_embd_r_total * element_size (stride between seqs)
    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),  // nb1: stride between channels
        n_embd_r_total * ggml_element_size(conv_state_all),  // nb2: stride between seqs
        qkv * conv_state_size * ggml_element_size(conv_state_all));

// Causal Conv1d function for Q,K,V
// When qkv is 0, it is Q, 1 is K, 2 is V
    // Step 1: Q, K, V projections -> [d_inner, n_tokens]
    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);

    // Reshape input: {d_inner, n_tokens} -> {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    // Concat Q conv state and current input: {d_conv-1 + n_seq_tokens, d_inner, n_seqs}
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // Save last (d_conv-1) columns back to Q conv state
    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_1d(ctx0, conv_states_all, conv_state_size * n_seqs,
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));
    // Reshape conv weight: GGUF [d_conv, 1, d_inner, 1] -> ggml_ssm_conv expects [d_conv, d_inner]
    // GGUF stores as [d_conv, 1, d_inner, 1] with memory layout w[conv_step + channel * d_conv]
    // vLLM stores as [d_inner, d_conv] with memory layout w[channel * d_conv + conv_step]
    // ggml_ssm_conv computes: c[conv_step + channel * d_conv]
    // GGUF layout: [d_conv, 1, d_inner] or [d_conv, 1, d_inner, 1] -> reshape to [d_conv, d_inner]
    // Reshape conv weight from [d_conv, 1, d_inner, 1] to [d_conv, d_inner] for ggml_ssm_conv
    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    // Apply conv1d
    // ggml_ssm_conv output: {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    // Reshape to 2D for bias add: {d_inner, n_tokens}
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

llm_build_kimi_linear::llm_build_kimi_linear(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context_mamba(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    // Note: Kimi MLA does NOT use RoPE (rotary_emb=None in vLLM)
    // So we don't need inp_pos

    auto * inp_kv = !hparams.is_mla() ? build_inp_mem_hybrid() : nullptr;
    auto * inp_k = hparams.is_mla() ? build_inp_mem_hybrid_k() : nullptr;
    auto * inp_rs = hparams.is_mla() ? inp_k->get_recr() : inp_kv->get_recr();
    auto * inp_attn_kv = !hparams.is_mla() ? inp_kv->get_attn() : nullptr;
    auto * inp_attn_k = hparams.is_mla() ? inp_k->get_attn() : nullptr;

    // Output ids for selecting which tokens to output
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * chunked_causal_mask =
        ggml_tri(ctx0, ggml_fill_inplace(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, CHUNK_SIZE, CHUNK_SIZE), 1.0f),
                    GGML_TRI_TYPE_LOWER);

    ggml_tensor * chunked_identity = ggml_diag(ctx0, ggml_fill_inplace(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, CHUNK_SIZE), 1.0f));
    ggml_tensor * chunked_diag_mask = ggml_add(ctx0, chunked_causal_mask, chunked_identity);

    ggml_build_forward_expand(gf, chunked_causal_mask);
    ggml_build_forward_expand(gf, chunked_identity);
    ggml_build_forward_expand(gf, chunked_diag_mask);

    // Kimi dimension constants
    const int64_t n_head = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t d_inner = n_head * head_dim;  // 32 * 128 = 4096
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // Verify batch consistency for recurrent layers
    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // MLA params
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    // qk_rope_head_dim = 64 (from Kimi config) which is hparams.n_rot
    // Confirmed from tensor shape: wkv_a_mqa [2304, 576] = [n_embd, kv_lora_rank + qk_rope_head_dim]
    const int64_t n_embd_head_qk_rope = hparams.n_rot;  // config.qk_rope_head_dim
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;  // 192 - 64 = 128
    // Attention scale for MLA
    const float kq_scale_mla = 1.0f / sqrtf((float)n_embd_head_k_mla);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        // Attention Norm
        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Check layer type by checking which tensors exist
        // KDA layers have ssm_a_log tensor, MLA layers have wkv_a_mqa tensor
        bool is_kda = (layer.ssm_a != nullptr);
        bool is_mla = (layer.wkv_a_mqa != nullptr);

        if (is_kda) {
            // === KDA Layer (Kimi Delta Attention) with Recurrent State ===
            // Reference: vLLM kda.py
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            // Get conv states from r_l tensor (Q, K, V each have separate state)
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            cb(conv_states_all, "conv_states_all", il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
            ggml_tensor * Qcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Kcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Vcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // g1 = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
            ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
            cb(g1, "g1 f_b(f_a(cur))", il);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_softplus(ctx0, g1);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

            // A_log shape is [1, n_head] or [1, n_head, 1, 1], need to broadcast to [head_dim, n_head, n_tokens]. No need to -exp(a_log) because it was done in convert_hf_to_gguf.py
            // Reshape to [1, n_head, 1] for broadcasting with g1 [head_dim, n_head, n_tokens]
            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A);
            cb(g1, "kda_g1", il);

            // Compute beta (mixing coefficient)
            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx0, beta, n_head, 1, n_seq_tokens, n_seqs);
            cb(beta, "kda_beta", il);

            // Reshape for KDA recurrence
            // {n_embd, n_tokens} -> {n_embd, n_seq_tokens, n_seqs}
            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            // Get SSM state and compute KDA recurrence using ggml_kda_scan
            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);
            // Choose between build_kda_chunking and build_kda_recurrent based on n_tokens
            std::pair<ggml_tensor *, ggml_tensor *> attn_out = n_seq_tokens == 1 ?
                build_kda_autoregressive(Qcur, Kcur, Vcur, g1, beta, state, il) :
                build_kda_chunking(Qcur, Kcur, Vcur, g1, beta, state, chunked_causal_mask, chunked_identity, chunked_diag_mask, il);

            ggml_tensor * output = attn_out.first;
            ggml_tensor * new_state = attn_out.second;
            cb(output, "attn_output", il);
            cb(new_state, "new_state", il);

            // Update the recurrent states
            ggml_build_forward_expand(gf,
                                     ggml_cpy(ctx0, new_state,
                                              ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                                                           kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            // Output gating g2 = g_b(g_a(x))
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
            ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
            cb(g2, "g2 g_b(g_a(cur_2d))", il);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);

            // Apply o_norm with sigmoid gating
            // Note: Kimi model uses sigmoid gating, not SiLU (despite FusedRMSNormGated default being swish)
            // Formula: output = RMSNorm(x) * sigmoid(g)
            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head,  n_seq_tokens * n_seqs);
            ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            cb(normed, "kda_normed", il);
            ggml_tensor * gate = ggml_sigmoid(ctx0, g2);
            ggml_tensor * gated = ggml_mul(ctx0, normed, gate);

            // Output projection
            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur = ggml_mul_mat(ctx0, layer.wo, gated);
            cb(cur, "kda_out", il);

        } else if (is_mla) {
            // === MLA Layer (Multi-head Latent Attention) without KV Cache ===
            // Reference: vLLM mla.py
            // Step 1: Q projection and reshape
            // vLLM Kimi: q = q_proj(hidden_states), then view as [n_tokens, n_head, qk_head_dim]
            // Note: Kimi MLA does NOT use RoPE (rotary_emb=None in vLLM)
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);

            // Step 2: KV compression
            // kv_cmpr_pe = kv_a_proj_with_mqa(hidden_states) -> [kv_lora_rank + qk_rope_head_dim, n_tokens]
            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

            // Split: kv_cmpr = kv_lora[:kv_lora_rank], k_pe = kv_lora[kv_lora_rank:]
            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            // Note: Kimi MLA does NOT apply RoPE (rotary_emb=None in vLLM)
            // k_pe is used directly without RoPE
            // Normalize kv_c
            kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

            if (layer.wk_b && layer.wv_b) { // MLA KV cache enabled
                // extract q_nope
                ggml_tensor * q_nope =
                    ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(Qcur->type, n_embd_head_k_mla),
                                 ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);
                cb(q_nope, "q_nope", il);

                // and {n_embd_head_qk_rope, n_head, n_tokens}
                ggml_tensor * q_pe = ggml_view_3d(
                    ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(Qcur->type, n_embd_head_k_mla),
                    ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, ggml_row_size(Qcur->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                // note: rope must go first for in-place context shifting in build_rope_shift()
                Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn_k, layer.wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
                cb(cur, "mla_out", il);
            } else { // MLA KV cache disabled. Fall back to MHA KV cache.
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
                cb(Qcur, "mla_Q", il);
                // KV decompression: kv = kv_b_proj(kv_c_normed)
                ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
                const int64_t kv_per_head = n_embd_head_qk_nope + n_embd_head_v_mla;

                // Split kv into k_nope and v
                ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head), 0);
                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head),
                    ggml_row_size(kv->type, n_embd_head_qk_nope));
                Vcur = ggml_cont(ctx0, Vcur);
                cb(Vcur, "mla_V", il);

                // Concatenate k_nope + k_pe (broadcast k_pe to all heads)
                // K = [k_nope, k_pe] where k_nope is [qk_nope_head_dim, n_head, n_tokens]
                // and k_pe is [qk_rope_head_dim, 1, n_tokens] broadcast to all heads
                // Need to broadcast k_pe from [qk_rope, 1, n_tokens] to [qk_rope, n_head, n_tokens]
                ggml_tensor * k_pe_target = ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens);
                ggml_tensor * k_pe_repeated = ggml_repeat(ctx0, k_pe, k_pe_target);
                ggml_tensor * Kcur = ggml_concat(ctx0, k_pe_repeated, k_nope, 0);
                cb(Kcur, "mla_K", il);

                // Direct softmax attention (with MHA KV cache)
                // Use build_attn with inp_attn for proper mask handling
                cur = build_attn(inp_attn_kv, layer.wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale_mla, il);
                cb(cur, "mla_out", il);
            }
        } else {
            // Unknown layer type - this should not happen
            GGML_ABORT("Kimi layer is neither KDA nor MLA - missing required tensors");
        }

        // On last layer, select only the output tokens
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FFN Norm
        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // Dense FFN layer
            cur = build_ffn(cur,
                layer.ffn_up, NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE layer
            // Kimi uses moe_renormalize=True and routed_scaling_factor (stored as expert_weights_scale) = 2.446
            ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used,
                LLM_FFN_SILU, true,
                true, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            // Shared expert
            {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                        layer.ffn_up_shexp, NULL, NULL,
                        layer.ffn_gate_shexp, NULL, NULL,
                        layer.ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        // Residual
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    // Final Norm
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // Output
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

/*
    This is a ggml implementation of the naive_chunk_kda function of
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
*/
std::pair<ggml_tensor *, ggml_tensor *> llm_build_kimi_linear::build_kda_chunking(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * gk,
        ggml_tensor * beta,
        ggml_tensor * state,
        ggml_tensor * causal_mask,
        ggml_tensor * identity,
        ggml_tensor * diag_mask,
        int           il) {
    GGML_ASSERT(ggml_is_contiguous(state));

    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(gk->ne[0] == S_v && gk->ne[1] == H_v && gk->ne[2] == n_tokens && gk->ne[3] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H_v && state->ne[3] == n_seqs);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);

    GGML_ASSERT(H_k == H_v);  // we did a repeat to make sure this is the case

    // TODO: can this ever be false?
    const bool use_qk_l2norm = true;

    if (use_qk_l2norm) {
        const float eps_norm = hparams.f_norm_rms_eps;

        q = ggml_l2_norm(ctx0, q, eps_norm);
        k = ggml_l2_norm(ctx0, k, eps_norm);
    }

    const float scale = 1.0f / sqrtf(S_v);

    beta = ggml_sigmoid(ctx0, beta);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(beta, "beta_in", il);
    cb(gk, "gk_in", il);

    q = ggml_cont_4d(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3), S_k, n_tokens, H_k, n_seqs);
    k = ggml_cont_4d(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3), S_k, n_tokens, H_k, n_seqs);
    v = ggml_cont_4d(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
    gk = ggml_cont_4d(ctx0, ggml_permute(ctx0, gk, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);

    beta  = ggml_cont(ctx0, ggml_permute(ctx0, beta, 2, 0, 1, 3));
    state = ggml_reshape_4d(ctx0, state, S_v, S_v, H_v, n_seqs);

    cb(q, "q_perm", il);
    cb(k, "k_perm", il);
    cb(v, "v_perm", il);
    cb(beta, "beta_perm", il);
    cb(gk, "gk_perm", il);
    cb(state, "state_in", il);

    GGML_ASSERT(q->ne[1] == n_tokens && q->ne[0] == S_k && q->ne[2] == H_k && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[1] == n_tokens && k->ne[0] == S_k && k->ne[2] == H_k && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[1] == n_tokens && v->ne[0] == S_v && v->ne[2] == H_k && v->ne[3] == n_seqs);
    GGML_ASSERT(beta->ne[1] == n_tokens && beta->ne[2] == H_k && beta->ne[0] == 1 && beta->ne[3] == n_seqs);

    // Do padding
    const int64_t chunk_size = CHUNK_SIZE;

    const int64_t pad = (chunk_size - n_tokens % chunk_size) % chunk_size;
    const int64_t n_chunks = (n_tokens + pad) / chunk_size;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    gk = ggml_pad(ctx0, gk, 0, pad, 0, 0);
    beta = ggml_pad(ctx0, beta, 0, pad, 0, 0);

    cb(q, "q_pad", il);
    cb(k, "k_pad", il);
    cb(v, "v_pad", il);
    cb(beta, "beta_pad", il);
    cb(gk, "gk_pad", il);

    ggml_tensor * v_beta = ggml_mul(ctx0, v, beta);
    ggml_tensor * k_beta = ggml_mul(ctx0, k, beta);

    cb(v_beta, "v_beta", il);
    cb(k_beta, "k_beta", il);

    const int64_t HB = H_k * n_seqs;

    q      = ggml_cont_4d(ctx0, q,      S_k, chunk_size, n_chunks, HB);
    k      = ggml_cont_4d(ctx0, k,      S_k, chunk_size, n_chunks, HB);
    k_beta = ggml_cont_4d(ctx0, k_beta, S_k, chunk_size, n_chunks, HB);
    v      = ggml_cont_4d(ctx0, v,      S_v, chunk_size, n_chunks, HB);
    v_beta = ggml_cont_4d(ctx0, v_beta, S_v, chunk_size, n_chunks, HB);

    gk    = ggml_cont_4d(ctx0, gk, S_k, chunk_size, n_chunks, HB);
    beta = ggml_cont_4d(ctx0, beta, 1, chunk_size, n_chunks, HB);

    // switch for cumsum
    gk = ggml_cont_4d(ctx0, ggml_permute(ctx0, gk, 1, 0, 2, 3), chunk_size, S_k, n_chunks, HB);
    cb(gk, "gk", il);
    ggml_tensor * gk_cumsum = ggml_cumsum(ctx0, gk);
    cb(gk_cumsum, "gk_cumsum", il);

/*
    Compute Akk and Aqk loop together
    Akk loop:
    for i in range(BT):
        k_i = k[..., i, :] # k_i [B,H,NT,S]
        g_i = g[..., i:i+1, :] # g_i [B,H,NT,1,S]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    Aqk loop:
    for j in range(BT):
        k_j = k[:, :, i, j]
        g_j = g[:, :, i, j:j+1, :]
        A[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
*/
    const int64_t CHB = n_chunks * H_k * n_seqs;
    ggml_tensor * gkcs_i = ggml_reshape_4d(ctx0, gk_cumsum, chunk_size, 1, S_k, CHB);  // [chunk_size, 1, S_k, CHB]
    ggml_tensor * gkcs_j = ggml_reshape_4d(ctx0, gkcs_i, 1, chunk_size, S_k, CHB);  // [1, chunk_size, S_k, CHB]

    ggml_tensor * gkcs_j_bc = ggml_repeat_4d(ctx0, gkcs_j, chunk_size, chunk_size, S_k, CHB);  // [1, chunk_size, S_k, CHB] -> [chunk_size, chunk_size, S_k, CHB]
    // decay_mask [chunk_size,chunk_size,S_k,CHB]
    ggml_tensor * decay_mask = ggml_sub(ctx0, gkcs_j_bc, gkcs_i);
    cb(decay_mask, "decay_mask", il);

    decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);
    cb(decay_mask, "decay_masked", il);
    decay_mask = ggml_exp(ctx0, decay_mask);
    decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);

    // decay_mask [S_k,BT_j,BT_i,CHB] *Note* second and third chunk_sizes are switched
    decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, chunk_size, chunk_size, CHB);

    ggml_tensor * k_i = ggml_reshape_4d(ctx0, k, S_k, chunk_size, 1, CHB);
    ggml_tensor * k_j = ggml_reshape_4d(ctx0, k, S_k, 1, chunk_size, CHB);
    ggml_tensor * q_i = ggml_reshape_4d(ctx0, q, S_k, chunk_size, 1, CHB);

    ggml_tensor * decay_k_i = ggml_mul(ctx0, decay_mask, k_i);
    ggml_tensor * decay_q_i = ggml_mul(ctx0, decay_mask, q_i);

    // decay_k_i [S.BT,BT,CHB] @ k_j [S,1,BT,CHB] = Akk [BT,1,BT,CHB]
    ggml_tensor * Akk = ggml_mul_mat(ctx0, decay_k_i, k_j);
    ggml_tensor * Aqk = ggml_mul_mat(ctx0, decay_q_i, k_j);
    Akk = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, Akk, chunk_size, chunk_size, n_chunks, HB)));
    Aqk = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, Aqk, chunk_size, chunk_size, n_chunks, HB)));
    cb(Akk, "Akk", il);
    cb(Aqk, "Aqk", il);

    Akk = ggml_mul(ctx0, Akk, beta);
    Akk = ggml_neg(ctx0, ggml_mul(ctx0, Akk, causal_mask));
    cb(Akk, "attn_pre_solve", il);

    Aqk = ggml_mul(ctx0, Aqk, diag_mask);
    Aqk = ggml_scale(ctx0, Aqk, scale); // scale q
    cb(Aqk, "Aqk_masked", il);

    // for i in range(1, chunk_size):
    //          row = attn[..., i, :i].clone()
    //          sub = attn[..., :i, :i].clone()
    //          attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    // attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    //
    // We reduce this to a linear triangular solve: AX = B, where B = attn, A = I - tril(A)
    ggml_tensor * attn_lower = ggml_mul(ctx0, Akk, causal_mask);
    ggml_tensor * lhs        = ggml_sub(ctx0, ggml_repeat(ctx0, identity, attn_lower), attn_lower);

    ggml_tensor * lin_solve  = ggml_solve_tri(ctx0, lhs, Akk, true, true, false);
    Akk                      = ggml_mul(ctx0, lin_solve, causal_mask);
    Akk                      = ggml_add(ctx0, Akk, identity);

    cb(Akk, "attn_solved", il);

    // switch back for downstream
    gk_cumsum = ggml_cont_4d(ctx0, ggml_permute(ctx0, gk_cumsum, 1, 0, 2, 3), S_k, chunk_size, n_chunks, HB);
    ggml_tensor * gkexp      = ggml_exp(ctx0, gk_cumsum);
    cb(gk_cumsum, "gk_cumsum", il);

    // u = (A*beta[..., None, :]) @ v  aka U_[t]
    ggml_tensor * vb = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_beta)), Akk);

    ggml_tensor * kbeta_gkexp = ggml_mul(ctx0, k_beta, gkexp);
    cb(kbeta_gkexp, "kbeta_gkexp", il);

    ggml_tensor * k_cumdecay = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, kbeta_gkexp)), Akk);
    cb(k_cumdecay, "k_cumdecay", il);

    ggml_tensor * core_attn_out = nullptr;
    ggml_tensor * new_state = ggml_dup(ctx0, state);

    cb(new_state, "new_state", il);

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
// extract one chunk worth of data
        auto chunkify = [=](ggml_tensor * t) {
                    return ggml_cont(ctx0, ggml_view_4d(ctx0, t, t->ne[0], chunk_size, 1, t->ne[3],
                t->nb[1], t->nb[2], t->nb[3], t->nb[2] * chunk));
        };
        auto chunkify_A = [=](ggml_tensor * t) {
                    return ggml_cont(ctx0, ggml_view_4d(ctx0, t, chunk_size, chunk_size, 1, t->ne[3],
                t->nb[1], t->nb[2], t->nb[3], t->nb[2] * chunk));
        };


// k [S,BT,NT,H*B] => k_chunk [S,BT,1,H*B]
        ggml_tensor * k_chunk = chunkify(k);
        ggml_tensor * q_chunk = chunkify(q);
        ggml_tensor * vb_chunk = chunkify(vb);

// gk_cumsum [S,BT,NT,H*B] => gk_cs_chunk [S,BT,1,H*B]
        ggml_tensor * gk_cs_chunk = chunkify(gk_cumsum);
        ggml_tensor * k_cumdecay_chunk = chunkify(k_cumdecay);
        ggml_tensor * gkexp_chunk = ggml_exp(ctx0, gk_cs_chunk);
        ggml_tensor * Aqk_chunk = chunkify_A(Aqk);

        ggml_tensor * state_t = ggml_cont_4d(ctx0, ggml_permute(ctx0, new_state, 1, 0, 2, 3), S_v, S_v, 1, H_v * n_seqs);

        // new_state [S,S,1,H*B] k_cumdecay_chunk [S,BT,1,H*B]
        // v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state or W_[t] @ S_[t]
        ggml_tensor * v_prime = ggml_mul_mat(ctx0, state_t, k_cumdecay_chunk);

        // v_new = v_i - v_prime or U_[t] - W_[t]*S_[t]
        ggml_tensor * v_new = ggml_sub(ctx0, ggml_repeat(ctx0, vb_chunk, v_prime), v_prime);
        ggml_tensor * v_new_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_new));

        // q_chunk [S,BT,1,H*B] gkexp_chunk [S,BT,1,H*B]
        // attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        // or Gamma_[t]*Q_]t] @ S
        ggml_tensor * q_gk_exp   = ggml_mul(ctx0, q_chunk, gkexp_chunk);
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, state_t, q_gk_exp);
        attn_inter = ggml_scale(ctx0, attn_inter, scale); // scale q

        // v_new_t [S,BT,1,H*B] Aqk [BT,BT,1,H*B]
        // core_attn_out[:, :, i] = attn_inter + attn @ v_new or A' @ (U_[t] - W_[t]*S_[t])
        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_new_t, Aqk_chunk);

        // o[:, :, i] = (q_i * g_i.exp()) @ S + A @ v_i
        ggml_tensor * core_attn_out_chunk = ggml_add(ctx0, attn_inter, v_attn);

        core_attn_out = core_attn_out == nullptr ? core_attn_out_chunk : ggml_concat(ctx0, core_attn_out, core_attn_out_chunk, 1);

        ggml_tensor * gk_cum_last =
            ggml_cont(ctx0, ggml_view_4d(ctx0, gk_cs_chunk, gk_cs_chunk->ne[0], 1, gk_cs_chunk->ne[2], gk_cs_chunk->ne[3],
                                        gk_cs_chunk->nb[1], gk_cs_chunk->nb[2], gk_cs_chunk->nb[3],
                                        gk_cs_chunk->nb[1] * (gk_cs_chunk->ne[1] - 1)));

        ggml_tensor * gkexp_last = ggml_exp(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, gk_cum_last)));

        ggml_tensor * gk_diff = ggml_neg(ctx0, ggml_sub(ctx0, gk_cs_chunk, gk_cum_last));

        ggml_tensor * gk_diff_exp = ggml_exp(ctx0, gk_diff);

        ggml_tensor * key_gkdiff = ggml_mul(ctx0, k_chunk, gk_diff_exp);

        // rearrange((g_i[:,:,-1:] - g_i).exp()*k_i, 'b h c k -> b h k c') @ (U_[t] - W_[t] @ S)
        ggml_tensor * kgdmulvnew = ggml_mul_mat(ctx0, v_new_t, ggml_cont(ctx0, ggml_transpose(ctx0, key_gkdiff)));

        new_state = ggml_add(ctx0,
            ggml_mul(ctx0, new_state, ggml_reshape_4d(ctx0, gkexp_last, gkexp_last->ne[0], gkexp_last->ne[1], H_v, n_seqs)),
            ggml_reshape_4d(ctx0, kgdmulvnew, kgdmulvnew->ne[0], kgdmulvnew->ne[1], H_v, n_seqs));
    }

    core_attn_out = ggml_cont_4d(ctx0, core_attn_out, S_v, chunk_size * n_chunks, H_v, n_seqs);

    // truncate padded tokens
    ggml_tensor * output_tokens = ggml_view_4d(ctx0, core_attn_out,
            S_v, n_tokens, H_v, n_seqs,
            ggml_row_size(core_attn_out->type, S_v),
            ggml_row_size(core_attn_out->type, S_v * chunk_size * n_chunks),
            ggml_row_size(core_attn_out->type, S_v * chunk_size * n_chunks * H_v), 0);
    output_tokens = ggml_cont(ctx0, output_tokens);
    // permute back to (S_v, H_v, n_tokens, n_seqs)
    output_tokens = ggml_permute(ctx0, output_tokens, 0, 2, 1, 3);
    output_tokens = ggml_cont(ctx0, output_tokens);

    cb(new_state, "output_state", il);

    return {output_tokens, new_state};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_kimi_linear::build_kda_autoregressive(
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * v,
    ggml_tensor * gk,
    ggml_tensor * beta,
    ggml_tensor * state,
    int il) {
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(gk));

    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_tokens == 1);
    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(gk->ne[0] == S_k && gk->ne[1] == H_k && gk->ne[2] == n_tokens && gk->ne[3] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_k && state->ne[2] == H_v && state->ne[3] == n_seqs);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);

    GGML_ASSERT(H_k == H_v);  // we did a repeat to make sure this is the case

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
    cb(gk, "gk_in", il);

// g [H,1,B,1] g_t [1,H,B,1] => [1,1,H,B]
// gk [S,H,1,B] => [S,1,H,B] gk_t [1,S,H,B]
// beta [H,1,1,B] beta_t [1,H,1,B] => [1,1,H,B]
    gk = ggml_reshape_4d(ctx0, gk, S_k, 1, H_k, n_seqs);
    ggml_tensor * gk_t = ggml_cont(ctx0, ggml_transpose(ctx0, gk));
    ggml_tensor * beta_t = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, beta), 1, 1, H_k, n_seqs);

    // Apply exponential to gk_t
    gk_t = ggml_exp(ctx0, gk_t);
    // Apply the gated delta rule for the single timestep
    // last_recurrent_state = last_recurrent_state * gk_t
    // S = S * g_i[..., None].exp()
    state = ggml_mul(ctx0, state, gk_t);

    ggml_tensor * state_t = ggml_cont(ctx0, ggml_transpose(ctx0, state));

// state [S,S,H,B] k [S,1,H,B] k_state [S_v,1,H,B]
    k = ggml_reshape_4d(ctx0, k, S_k, 1, H_k, n_seqs);
    ggml_tensor * k_state = ggml_mul_mat(ctx0, state_t, k);

    // v_i - (k_i[..., None] * S).sum(-2)
    v = ggml_reshape_4d(ctx0, v, S_v, 1, H_v, n_seqs);
    ggml_tensor * v_diff = ggml_sub(ctx0, v, k_state);

    // b_i[..., None] * k_i
    ggml_tensor * k_beta = ggml_mul(ctx0, k, beta_t);

    // S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
    // v_diff_t [1,S_v,H,B] k_beta_t [1,S_k,H,B] state [S_v,S_k,H,B]
    state = ggml_add(ctx0, state, ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_diff)), ggml_cont(ctx0, ggml_transpose(ctx0, k_beta))));

    q = ggml_reshape_4d(ctx0, q, S_k, 1, H_k, n_seqs);
    state_t = ggml_cont(ctx0, ggml_transpose(ctx0, state));
    ggml_tensor * core_attn_out = ggml_mul_mat(ctx0, state_t, q);
    // core_attn_out should be [S_v, 1, H_v, n_seqs] after this
    cb(core_attn_out, "output_tokens", il);
    cb(state, "new_state", il);

    return {core_attn_out, state};
}

