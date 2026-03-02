#include "ggml.h"
#include "models.h"

#define CHUNK_SIZE 64

llm_build_qwen3next::llm_build_qwen3next(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context_mamba(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

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

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, causal_mask, identity, diag_mask, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
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

        // FFN layer (MoE or dense) - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
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

ggml_tensor * llm_build_qwen3next::build_delta_net_chunking(
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

    GGML_ASSERT(H_k == H_v);  // we did a repeat to make sure this is the case

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

    // Do padding
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

        // attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        attn = ggml_mul_mat(ctx0, k_chunk, q_chunk);
        attn = ggml_mul(ctx0, attn, decay_mask_chunk);
        attn = ggml_mul(ctx0, attn, diag_mask);

        ggml_tensor * state_t = ggml_cont_4d(ctx0, ggml_permute(ctx0, new_state, 1, 0, 2, 3), S_v, S_v, 1, H_v * n_seqs);

        // v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        ggml_tensor * v_prime = ggml_mul_mat(ctx0, state_t, k_cumdecay_chunk);

        // v_new = v_i - v_prime
        ggml_tensor * v_new = ggml_sub(ctx0, ggml_repeat(ctx0, v_chunk, v_prime), v_prime);
        ggml_tensor * v_new_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_new));

        // attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        ggml_tensor * q_g_exp    = ggml_mul(ctx0, q_chunk, gexp_chunk);
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, state_t, q_g_exp);

        // core_attn_out[:, :, i] = attn_inter + attn @ v_new
        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_new_t, attn);

        ggml_tensor * core_attn_out_chunk = ggml_add(ctx0, attn_inter, v_attn);

        core_attn_out = core_attn_out == nullptr ? core_attn_out_chunk : ggml_concat(ctx0, core_attn_out, core_attn_out_chunk, 1);

        // g_last = torch.clamp(g_cum[:, :, -1], max=50.0).exp().unsqueeze(-1).unsqueeze(-1)
        // g_diff = torch.clamp(g_cum[:, :, -1:] - g_cum, max=50.0).exp()
        // key_gdiff = key * g_diff.unsqueeze(-1)
        // kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
        // last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew

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

    // flatten output
    ggml_tensor * flat_output =
        ggml_cont_1d(ctx0, ggml_permute(ctx0, output_tokens, 0, 2, 1, 3), S_v * H_v * n_tokens * n_seqs);

    ggml_tensor * flat_state = ggml_cont_1d(ctx0, new_state, S_v * S_v * H_v * n_seqs);

    return ggml_concat(ctx0, flat_output, flat_state, 0);
}

ggml_tensor * llm_build_qwen3next::build_delta_net_autoregressive(
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

    GGML_ASSERT(n_tokens == 1);  // This function is optimized for single token processing
    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v * H_v && state->ne[2] == 1 && state->ne[3] == n_seqs);

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
    cb(g, "g_in", il);

    state = ggml_reshape_4d(ctx0, state, S_v, S_v, H_v, n_seqs);

    ggml_tensor * g_t    = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, g), 1, 1, H_k, n_seqs);
    ggml_tensor * beta_t = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, beta), 1, 1, H_k, n_seqs);

    // Apply exponential to g_t
    g_t = ggml_exp(ctx0, g_t);

    // Apply the gated delta rule for the single timestep
    // last_recurrent_state = last_recurrent_state * g_t
    state = ggml_mul(ctx0, state, g_t);

    // kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    ggml_tensor * k_t_unsqueezed = ggml_reshape_4d(ctx0, k, 1, S_v, H_v, n_seqs);
    ggml_tensor * kv_mem         = ggml_mul(ctx0, state, k_t_unsqueezed);
    // we need to sum over dim=-2, so we transpose, sum, then transpose again
    kv_mem = ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, kv_mem))));

    // v_t = v.unsqueeze(2) (we insert the singleton dimension after n_seqs and H_v)
    ggml_tensor * v_t    = ggml_reshape_4d(ctx0, v, S_v, 1, H_v, n_seqs);
    // delta = (v_t - kv_mem) * beta_t
    ggml_tensor * v_diff = ggml_sub(ctx0, v_t, kv_mem);  // both should be [S_v, 1, H_v, n_seqs]
    ggml_tensor * delta  = ggml_mul(ctx0, v_diff, beta_t);

    // last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta
    ggml_tensor * k_t_delta = ggml_mul(ctx0, ggml_repeat_4d(ctx0, k_t_unsqueezed, S_v, S_v, H_v, n_seqs), delta);
    state                   = ggml_add(ctx0, state, k_t_delta);

    // Compute the attention output
    // core_attn_out = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    ggml_tensor * q_t_unsqueezed = ggml_reshape_4d(ctx0, q, 1, S_v, H_v, n_seqs);  // unsqueeze q_t
    ggml_tensor * state_q        = ggml_mul(ctx0, state, q_t_unsqueezed);
    // again, since it's over dim = -2, transpose, sum, transpose back
    ggml_tensor * core_attn_out =
        ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, state_q))));

    // core_attn_out should be [S_v, 1, H_v, n_seqs] after this
    cb(core_attn_out, "output_tokens", il);
    cb(state, "new_state", il);

    // flatten output, no need to permute since n_tokens is 1 so [S_v, 1, H_v, n_seqs] and [S_v, H_v, 1, n_seqs] are equivalent memory-layout wise
    ggml_tensor * flat_output = ggml_reshape_1d(ctx0, core_attn_out, S_v * H_v * n_tokens * n_seqs);
    ggml_tensor * flat_state  = ggml_reshape_1d(ctx0, state, S_v * S_v * H_v * n_seqs);

    return ggml_concat(ctx0, flat_output, flat_state, 0);
}

ggml_tensor * llm_build_qwen3next::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_qwen3next::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur);
    cb(Qcur_full, "Qcur_full", il);

    Qcur_full = ggml_reshape_4d(ctx0, Qcur_full, n_embd_head * 2, n_head, n_tokens, 1);

    // Split Q projection into query and gate
    // The split should be along dimension 0 (the feature dimension)
    ggml_tensor * Qcur = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
                                             Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 0);
    ggml_tensor * gate =
        ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
                     Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], n_embd_head * ggml_element_size(Qcur_full));
    cb(Qcur, "Qcur", il);
    cb(gate, "gate", il);

    // Now reshape Qcur to [n_embd_head, n_head, n_tokens] for multi-head attention
    Qcur = ggml_cont_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
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

    // Reshape gate to [n_embd, n_tokens] for the sigmoid gating (flatten the heads)
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply RoPE
    Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_ext(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base,
            freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

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

ggml_tensor * llm_build_qwen3next::build_layer_attn_linear(
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

    // Input projections
    ggml_tensor * mixed_qkvz = build_lora_mm(model.layers[il].ssm_in, cur);
    cb(mixed_qkvz, "linear_attn_mixed_qkvz", il);

    ggml_tensor * mixed_ba = build_lora_mm(model.layers[il].ssm_beta_alpha, cur);
    cb(mixed_ba, "linear_attn_mixed_ba", il);

    int64_t       qkvz_new_dim        = 2 * head_k_dim + 2 * head_v_dim * (num_v_heads / num_k_heads);
    ggml_tensor * mixed_qkvz_reshaped = ggml_reshape_4d(ctx0, mixed_qkvz, qkvz_new_dim, num_k_heads, n_seq_tokens, n_seqs);

    // Reshape mixed_ba: [batch, seq_len, hidden_size] -> [batch, seq_len, num_k_heads, 2*num_v_heads/num_k_heads]
    int64_t       ba_new_dim        = 2 * num_v_heads / num_k_heads;
    ggml_tensor * mixed_ba_reshaped = ggml_reshape_4d(ctx0, mixed_ba, ba_new_dim, num_k_heads, n_seq_tokens, n_seqs);

    // Split mixed_ba into b and a (beta and alpha parameters)
    int64_t split_sizes_ba[2] = {
        num_v_heads / num_k_heads,  // beta size
        num_v_heads / num_k_heads   // alpha size
    };

    ggml_tensor * b = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[0], num_k_heads, n_seq_tokens, n_seqs,
                                   mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3], 0);
    cb(b, "b", il);

    ggml_tensor * a = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[1], num_k_heads, n_seq_tokens, n_seqs,
                                   mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3],
                                   split_sizes_ba[0] * ggml_element_size(mixed_ba_reshaped));
    cb(a, "a", il);

    // Reshape b and a to merge head dimensions: [batch, seq_len, num_k_heads, num_v_heads/num_k_heads] -> [batch, seq_len, num_v_heads]
    ggml_tensor * beta  = ggml_cont_3d(ctx0, b, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * alpha = ggml_cont_3d(ctx0, a, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);
    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    // Split mixed_qkvz into query, key, value, z
    int64_t split_sizes_qkvz[4] = {
        head_k_dim,                              // query size
        head_k_dim,                              // key size
        head_v_dim * num_v_heads / num_k_heads,  // value size
        head_v_dim * num_v_heads / num_k_heads   // z size
    };

    ggml_tensor * query =
        ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[0], num_k_heads, n_seq_tokens, n_seqs,
                     mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3], 0);
    cb(query, "q", il);

    ggml_tensor * key = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[1], num_k_heads, n_seq_tokens, n_seqs,
                                     mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                                     split_sizes_qkvz[0] * sizeof(float));
    cb(key, "k", il);

    ggml_tensor * value =
        ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[2], num_k_heads, n_seq_tokens, n_seqs,
                     mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                     (split_sizes_qkvz[0] + split_sizes_qkvz[1]) * sizeof(float));
    cb(value, "v", il);

    ggml_tensor * z = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[3], num_k_heads, n_seq_tokens, n_seqs,
                                   mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                                   (split_sizes_qkvz[0] + split_sizes_qkvz[1] + split_sizes_qkvz[2]) * sizeof(float));
    cb(z, "z", il);

    // After creating query, key, and value_reshaped, reshape each to flatten the head dimensions
    // query: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
    ggml_tensor * query_flat = ggml_cont_3d(ctx0, query, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
    cb(query_flat, "query_flat", il);

    // key: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
    ggml_tensor * key_flat = ggml_cont_3d(ctx0, key, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
    cb(key_flat, "key_flat", il);

    // value_reshaped: [head_v_dim, num_v_heads, n_tokens, n_seqs] -> [head_v_dim * num_v_heads, n_tokens, n_seqs]
    ggml_tensor * value_flat = ggml_cont_3d(ctx0, value, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(value_flat, "value_flat", il);

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // bool use_precomputed_states = n_seq_tokens == 1 && mctx_cur->has_previous_state();

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    // Now concatenate along the feature dimension (dim 0) to get [conv_dim, n_tokens, n_seqs]
    ggml_tensor * qkv_mixed = ggml_concat(ctx0, query_flat, key_flat, 0);
    qkv_mixed               = ggml_concat(ctx0, qkv_mixed, value_flat, 0);
    cb(qkv_mixed, "qkv_mixed", il);

    qkv_mixed = ggml_permute(ctx0, qkv_mixed, 1, 0, 2, 3);
    cb(qkv_mixed, "qkv_mixed_permuted", il);

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;
    conv_states                    = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    // Update convolution state cache
    // Extract the last (conv_kernel_size - 1) states from conv_input
    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
        ggml_view_1d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels * n_seqs,
                     kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));
    cb(conv_states_all, "conv_states_updated", il);

    // Apply SSM convolution
    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    conv_output_proper = ggml_cont(ctx0, ggml_transpose(ctx0, conv_output_proper));
    cb(conv_output_proper, "conv_output_pre_silu", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix =
        ggml_cont_2d(ctx0, ggml_transpose(ctx0, conv_output_silu), qkv_dim, n_seq_tokens * n_seqs);
    cb(conv_qkv_mix, "conv_qkv_mix", il);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv =
        ggml_view_2d(ctx0, conv_qkv_mix, head_k_dim * num_k_heads, n_seq_tokens * n_seqs, conv_qkv_mix->nb[1], 0);
    cb(q_conv, "q_conv", il);
    ggml_tensor * k_conv =
        ggml_view_2d(ctx0, conv_qkv_mix, head_k_dim * num_k_heads, n_seq_tokens * n_seqs, conv_qkv_mix->nb[1],
                     head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));
    cb(k_conv, "k_conv", il);
    ggml_tensor * v_conv =
        ggml_view_2d(ctx0, conv_qkv_mix, head_v_dim * num_v_heads, n_seq_tokens * n_seqs, conv_qkv_mix->nb[1],
                     2 * head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));
    cb(v_conv, "v_conv", il);

    // Unsqueeze them
    q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    beta = ggml_cont_4d(ctx0, b, num_v_heads, 1, n_seq_tokens, n_seqs);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state               = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim * num_v_heads, 1, n_seqs);
    cb(state, "state_predelta", il);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        int64_t repeat_factor = num_v_heads / num_k_heads;

        // repeat interleave: reshape to (repeat part, 1, remaining part), do repeat, then reshape back
        ggml_tensor * q_reshaped = ggml_reshape_3d(ctx0, q_conv, head_k_dim, 1, num_k_heads * n_seq_tokens * n_seqs);
        ggml_tensor * k_reshaped = ggml_reshape_3d(ctx0, k_conv, head_k_dim, 1, num_k_heads * n_seq_tokens * n_seqs);

        // Repeat along the third dimension (the new dimension with size 1)
        ggml_tensor * q_repeated =
            ggml_repeat_4d(ctx0, q_reshaped, head_k_dim, repeat_factor, num_k_heads * n_seq_tokens * n_seqs, 1);
        ggml_tensor * k_repeated =
            ggml_repeat_4d(ctx0, k_reshaped, head_k_dim, repeat_factor, num_k_heads * n_seq_tokens * n_seqs, 1);

        // Reshape back to merge the head and repeat dimensions
        // From [head_dim, num_k_heads, repeat_factor, n_seq_tokens * n_seqs]
        // Back to [head_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs]
        q_conv = ggml_reshape_4d(ctx0, q_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
        k_conv = ggml_reshape_4d(ctx0, k_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // Choose between build_delta_net_chunking, build_delta_net_recurrent, and build_delta_net_autoregressive based on n_tokens
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
    // State starts after n_tokens elements along dimension 1
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

    // Reshape both attn_out_final and z to 2D tensors for normalization
    // attn_out_final: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * attn_out_2d_final =
        ggml_cont_2d(ctx0, attn_out_final, head_v_dim, num_v_heads * n_seq_tokens * n_seqs);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_cont_2d(ctx0, z, head_v_dim, num_v_heads * n_seq_tokens * n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(attn_out_2d_final, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_cont_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
    return cur;
}

ggml_tensor * llm_build_qwen3next::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    if (model.layers[il].ffn_gate_inp != nullptr) {
        // MoE branch
        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used, LLM_FFN_SILU,
                true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
        cb(moe_out, "ffn_moe_out", il);

        // Add shared experts if present - following Qwen3Next reference implementation
        if (model.layers[il].ffn_up_shexp != nullptr) {
            ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            // Apply shared expert gating as in the reference implementation
            // The shared expert has its own gate that is sigmoided
            // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
            ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
            cb(shared_gate, "shared_expert_gate", il);

            // Apply sigmoid to the gate
            shared_gate = ggml_sigmoid(ctx0, shared_gate);
            cb(shared_gate, "shared_expert_gate_sigmoid", il);

            // The gate needs to be broadcast to match the dimensions of ffn_shexp
            // ffn_shexp is [n_embd, n_tokens, 1, 1] and shared_gate is [1, n_tokens, 1, 1]
            // We need to repeat the gate along the feature dimension
            shared_gate = ggml_repeat(ctx0, shared_gate, ffn_shexp);
            cb(shared_gate, "shared_expert_gate_broadcast", il);

            // Apply the gate to the shared expert output
            ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
            cb(ffn_shexp, "ffn_shexp_gated", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
    } else {
        // Dense FFN branch (not currently used I believe)
        cur = build_ffn(cur,
            model.layers[il].ffn_up, NULL, NULL,
            model.layers[il].ffn_gate, NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
    }
    return cur;
}
