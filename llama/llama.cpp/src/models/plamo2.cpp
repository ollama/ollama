#include "models.h"

llm_build_plamo2::llm_build_plamo2(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context_mamba(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "embedding_output", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_hybrid = build_inp_mem_hybrid();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;

        // ggml_graph_add_node(gf, model.layers[il].attn_norm);
        // cb(model.layers[il].attn_norm, "attn_norm", il);

        // pre_mixer_norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

        // check if this layer is Mamba or Attention
        bool is_mamba_layer = hparams.is_recurrent(il);

        if (is_mamba_layer) {
            // PLaMo-2 Mamba layer
            cur = build_plamo2_mamba_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);
        } else {
            // PLaMo-2 Attention layer
            cur = build_plamo2_attn_layer(inp_hybrid->get_attn(), inp_pos, cur, model, il);
        }

        // post_mixer_norm
        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // residual connection
        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "attn_residual", il);
        residual = cur;

        // pre-ffn norm
        cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_pre_norm", il);

        // feed-forward network
        cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        // post ffn norm
        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        // residual connection
        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "ffn_residual", il);

        inpL = cur;
    }

    cur = inpL;

    // final norm
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);

    // Explicitly mark as output tensor to ensure proper backend assignment
    ggml_set_output(cur);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_plamo2::build_plamo2_attn_layer(llm_graph_input_attn_kv * inp,
                                                        ggml_tensor *             inp_pos,
                                                        ggml_tensor *             cur,
                                                        const llama_model &       model,
                                                        int                       il) {
    // self-attention
    {
        // PLaMo-2 uses combined QKV tensor
        ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, cur);
        cb(qkv, "wqkv", il);

        // split QKV tensor into Q, K, V
        const int64_t n_embd_head_q = hparams.n_embd_head_k;
        const int64_t n_embd_head_k = hparams.n_embd_head_k;
        const int64_t n_embd_head_v = hparams.n_embd_head_v;
        int32_t       n_head        = hparams.n_head(il);
        int32_t       n_head_kv     = hparams.n_head_kv(il);

        const int64_t q_offset = 0;
        const int64_t k_offset = n_embd_head_q * n_head;
        const int64_t v_offset = k_offset + n_embd_head_k * n_head_kv;

        ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, n_head, n_tokens, n_embd_head_q * sizeof(float),
                                          qkv->nb[1], q_offset * ggml_element_size(qkv));
        ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_kv, n_tokens, n_embd_head_k * sizeof(float),
                                          qkv->nb[1], k_offset * ggml_element_size(qkv));
        ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_kv, n_tokens, n_embd_head_v * sizeof(float),
                                          qkv->nb[1], v_offset * ggml_element_size(qkv));

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);

        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);

        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);

        cur = build_attn(inp,
            model.layers[il].wo, NULL,
            Qcur, Kcur, Vcur, NULL, NULL, NULL, 1.0f / sqrtf(float(n_embd_head_v)), il);
    }

    cb(cur, "attn_out", il);

    return cur;
}

ggml_tensor * llm_build_plamo2::build_plamo2_mamba_layer(llm_graph_input_rs * inp,
                                                         ggml_tensor *        cur,
                                                         const llama_model &  model,
                                                         const llama_ubatch & ubatch,
                                                         int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t d_inner  = hparams.ssm_d_inner;
    const int64_t d_state  = hparams.ssm_d_state;
    const int64_t n_heads  = hparams.ssm_dt_rank;
    const int64_t head_dim = d_inner / n_heads;
    const int64_t n_group  = hparams.ssm_n_group;
    const int64_t n_seqs   = ubatch.n_seqs;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    conv               = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    // in_proj: {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * zx = build_lora_mm(model.layers[il].ssm_in, cur);
    cb(zx, "mamba_in_proj", il);
    // {8192, 5, 1, 1} -> {8192, 1, 5, 1}
    zx = ggml_permute(ctx0, zx, 0, 2, 1, 3);
    zx = ggml_cont_4d(ctx0, zx, head_dim * 2, n_heads, n_seq_tokens, n_seqs);
    cb(zx, "mamba_in_proj_out", il);

    // split into z and x
    // => {head_dim * n_heads, n_seq_tokens, n_seqs}
    ggml_tensor * x = ggml_view_4d(ctx0, zx, head_dim, n_heads, n_seq_tokens, n_seqs, zx->nb[1], zx->nb[2], zx->nb[3],
                                   head_dim * ggml_element_size(zx));
    x               = ggml_cont_3d(ctx0, x, head_dim * n_heads, n_seq_tokens, n_seqs);
    // x = ggml_permute(ctx0, x, 0, 2, 1, 3);
    cb(x, "mamba_x_split", il);

    ggml_tensor * z =
        ggml_view_4d(ctx0, zx, head_dim, n_heads, n_seq_tokens, n_seqs, zx->nb[1], zx->nb[2], zx->nb[3], 0);
    cb(z, "mamba_z_split", il);

    // conv1d
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);
        cb(conv_x, "mamba_conv1d_input", il);

        // copy last (d_conv - 1) columns back into the state cache
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2],
                                               n_seq_tokens * (conv_x->nb[0]));

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv,
                                               ggml_view_1d(ctx0, conv_states_all,
                                                            (d_conv - 1) * (d_inner + 2 * n_group * d_state) * (n_seqs),
                                                            kv_head * (d_conv - 1) * (d_inner + 2 * n_group * d_state) *
                                                                ggml_element_size(conv_states_all))));
        cb(conv_states_all, "mamba_conv1d_state", il);

        // 1D convolution
        x = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);
        cb(x, "mamba_conv1d", il);

        x = ggml_silu(ctx0, x);
        cb(x, "mamba_conv1d_silu", il);
    }

    // SSM
    {
        // bcdt_proj: {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        ggml_tensor * x_bcdt = build_lora_mm(model.layers[il].ssm_x, x);
        cb(x_bcdt, "mamba_bcdt_proj", il);

        // split into dt, B, C
        const int64_t dt_dim = std::max(64, int(hparams.n_embd / 16));
        ggml_tensor * B  = ggml_view_3d(ctx0, x_bcdt, d_state, n_seq_tokens, n_seqs, x_bcdt->nb[1], x_bcdt->nb[2], 0);
        ggml_tensor * C  = ggml_view_3d(ctx0, x_bcdt, d_state, n_seq_tokens, n_seqs, x_bcdt->nb[1], x_bcdt->nb[2],
                                        ggml_element_size(x_bcdt) * d_state);
        ggml_tensor * dt = ggml_view_3d(ctx0, x_bcdt, dt_dim, n_seq_tokens, n_seqs, x_bcdt->nb[1], x_bcdt->nb[2],
                                        ggml_element_size(x_bcdt) * (2 * d_state));
        cb(B, "mamba_B_raw", il);
        cb(C, "mamba_C_raw", il);
        cb(dt, "mamba_dt_raw", il);

        // Apply RMS norm to dt, B, C (PLaMo-2 specific)
        B  = build_norm(B, model.layers[il].ssm_b_norm, NULL, LLM_NORM_RMS, il);
        C  = build_norm(C, model.layers[il].ssm_c_norm, NULL, LLM_NORM_RMS, il);
        dt = build_norm(dt, model.layers[il].ssm_dt_norm, NULL, LLM_NORM_RMS, il);
        cb(B, "mamba_B_normed", il);
        cb(C, "mamba_C_normed", il);
        cb(dt, "mamba_dt_normed", il);

        // dt_proj: {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = build_lora_mm(model.layers[il].ssm_dt, dt);
        dt = ggml_add(ctx0, dt, model.layers[il].ssm_dt_b);
        cb(dt, "mamba_dt_proj", il);

        ggml_tensor * A = ggml_reshape_2d(ctx0, model.layers[il].ssm_a, 1, n_heads);
        cb(A, "mamba_A", il);

        x = ggml_view_4d(ctx0, x, head_dim, n_heads, n_seq_tokens, n_seqs, head_dim * ggml_element_size(x),
                         head_dim * n_heads * ggml_element_size(x),
                         head_dim * n_heads * n_seq_tokens * ggml_element_size(x), 0);
        B = ggml_view_4d(ctx0, B, d_state, 1, n_seq_tokens, n_seqs, d_state * B->nb[0], B->nb[1], B->nb[2], 0);
        C = ggml_view_4d(ctx0, C, d_state, 1, n_seq_tokens, n_seqs, d_state * C->nb[0], C->nb[1], C->nb[2], 0);

        // use the states and the indices provided by build_recurrent_state
        // (this is necessary in order to properly use the states before they are overwritten,
        //  while avoiding to make unnecessary copies of the states)
        auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
            ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_heads, mctx_cur->get_size());

            // Custom operator to optimize the parallel associative scan
            // as described in the Annex D of the Mamba paper.
            // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
            return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
        };

        ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);
        cb(y_ssm, "mamba_ssm_scan", il);

        // store last states
        ggml_build_forward_expand(
            gf, ggml_cpy(
                    ctx0,
                    ggml_view_1d(ctx0, y_ssm, n_heads * head_dim * d_state * n_seqs,
                                 n_heads * head_dim * n_seq_tokens * n_seqs * ggml_element_size(y_ssm)),
                    ggml_view_1d(ctx0, ssm_states_all, n_heads * head_dim * d_state * n_seqs,
                                 kv_head * n_seqs * n_heads * head_dim * d_state * ggml_element_size(ssm_states_all))));
        cb(ssm_states_all, "mamba_ssm_states", il);

        ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, head_dim, n_heads, n_seq_tokens, n_seqs,
                                       head_dim * ggml_element_size(x), head_dim * n_heads * ggml_element_size(x),
                                       head_dim * n_heads * n_seq_tokens * ggml_element_size(x), 0);
        cb(y, "mamba_y_view", il);

        // Add D parameter and apply gating with z
        // {d_inner, n_seq_tokens, n_seqs} * {d_inner} => {d_inner, n_seq_tokens, n_seqs}
        ggml_tensor * D = ggml_reshape_2d(ctx0, model.layers[il].ssm_d, 1, n_heads);
        y               = ggml_add(ctx0, y, ggml_mul(ctx0, x, D));
        cb(y, "mamba_y_add_d", il);

        y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);
        cb(y, "mamba_y_swiglu_z", il);

        // out_proj: {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        y   = ggml_view_3d(ctx0, y, head_dim * n_heads, n_seq_tokens, n_seqs, y->nb[2], y->nb[3], 0);
        cur = build_lora_mm(model.layers[il].ssm_out, y);
        cb(cur, "mamba_out_proj", il);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
    cb(cur, "mamba_out", il);

    return cur;
}
