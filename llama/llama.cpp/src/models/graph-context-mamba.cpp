#include "models.h"

llm_graph_context_mamba::llm_graph_context_mamba(const llm_graph_params & params) : llm_graph_context(params) {}

ggml_tensor * llm_graph_context_mamba::build_mamba_layer(llm_graph_input_rs * inp,
                                                         ggml_tensor *        cur,
                                                         const llama_model &  model,
                                                         const llama_ubatch & ubatch,
                                                         int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const auto & layer = model.layers[il];

    const int64_t d_conv         = hparams.ssm_d_conv;
    const int64_t d_inner        = hparams.ssm_d_inner;
    const int64_t d_state        = hparams.ssm_d_state;
    const int64_t dt_rank        = hparams.ssm_dt_rank;
    const int64_t n_head         = d_inner;
    const int64_t head_dim       = 1;
    const int64_t n_seqs         = ubatch.n_seqs;
    // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
    const bool    ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    conv               = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * xz = build_lora_mm(layer.ssm_in, cur);
    // split the above in two
    // => {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * x  = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
    ggml_tensor * z =
        ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner * ggml_element_size(xz));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);

        // copy last (d_conv - 1) columns back into the state cache
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2],
                                               n_seq_tokens * (conv_x->nb[0]));

        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0, last_conv,
                         ggml_view_1d(ctx0, conv_states_all, (d_conv - 1) * (d_inner) * (n_seqs),
                                      kv_head * (d_conv - 1) * (d_inner) *ggml_element_size(conv_states_all))));

        // 1D convolution
        // The equivalent is to make a self-overlapping view of conv_x
        // over d_conv columns at each stride in the 3rd dimension,
        // then element-wise multiply that with the conv1d weight,
        // then sum the elements of each row,
        // (the last two steps are a dot product over rows (also doable with mul_mat))
        // then permute away the ne[0] dimension,
        // and then you're left with the resulting x tensor.
        // For simultaneous sequences, all sequences need to have the same length.
        x = ggml_ssm_conv(ctx0, conv_x, layer.ssm_conv1d);

        // bias
        x = ggml_add(ctx0, x, layer.ssm_conv1d_b);

        x = ggml_silu(ctx0, x);
    }

    // ssm
    {
        // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        ggml_tensor * x_db = build_lora_mm(layer.ssm_x, x);
        // split
        ggml_tensor * dt   = ggml_view_3d(ctx0, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0);
        ggml_tensor * B =
            ggml_view_4d(ctx0, x_db, d_state, /* n_group */ 1, n_seq_tokens, n_seqs, d_state * x_db->nb[0], x_db->nb[1],
                         x_db->nb[2], ggml_element_size(x_db) * dt_rank);
        ggml_tensor * C =
            ggml_view_4d(ctx0, x_db, d_state, /* n_group */ 1, n_seq_tokens, n_seqs, d_state * x_db->nb[0], x_db->nb[1],
                         x_db->nb[2], ggml_element_size(x_db) * (dt_rank + d_state));

        // Some Mamba variants (e.g. FalconMamba, Jamba) apply RMS norm in B, C & Dt layers
        if (ssm_dt_b_c_rms || (layer.ssm_dt_norm && layer.ssm_b_norm && layer.ssm_c_norm)) {
            dt = build_norm(dt, layer.ssm_dt_norm, NULL, LLM_NORM_RMS, il);
            B  = build_norm(B, layer.ssm_b_norm, NULL, LLM_NORM_RMS, il);
            C  = build_norm(C, layer.ssm_c_norm, NULL, LLM_NORM_RMS, il);
        }

        // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = build_lora_mm(layer.ssm_dt, dt);
        dt = ggml_add(ctx0, dt, layer.ssm_dt_b);

        cur = x;
        x   = ggml_reshape_4d(ctx0, x, head_dim, n_head, n_seq_tokens, n_seqs);

        ggml_tensor * A = layer.ssm_a;

        // use the states and the indices provided by build_recurrent_state
        // (this is necessary in order to properly use the states before they are overwritten,
        //  while avoiding to make unnecessary copies of the states)
        auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
            ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());

            // Custom operator to optimize the parallel associative scan
            // as described in the Annex D of the Mamba paper.
            // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
            return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
        };

        ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);

        // store last states
        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0, ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_seqs, x->nb[3] * x->ne[3]),
                         ggml_view_1d(ctx0, ssm_states_all, d_state * d_inner * n_seqs,
                                      kv_head * d_state * d_inner * ggml_element_size(ssm_states_all))));

        ggml_tensor * y = ggml_view_3d(ctx0, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[2], x->nb[3], 0);

        // TODO: skip computing output earlier for unused tokens

        y = ggml_add(ctx0, y, ggml_mul(ctx0, cur, layer.ssm_d));
        y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = build_lora_mm(layer.ssm_out, y);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_graph_context_mamba::build_mamba2_layer(llm_graph_input_rs * inp,
                                                          ggml_tensor *        cur,
                                                          const llama_model &  model,
                                                          const llama_ubatch & ubatch,
                                                          int                  il) const {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t d_inner  = hparams.ssm_d_inner;
    const int64_t d_state  = hparams.ssm_d_state;
    const int64_t n_head   = hparams.ssm_dt_rank;
    const int64_t head_dim = d_inner / n_head;
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

    // d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

    // {n_embd, d_in_proj} @ {n_embd, n_seq_tokens, n_seqs} => {d_in_proj, n_seq_tokens, n_seqs}
    ggml_tensor * zxBCdt = build_lora_mm(model.layers[il].ssm_in, cur);

    // split the above in three
    ggml_tensor * z   = ggml_view_4d(ctx0, zxBCdt, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * zxBCdt->nb[0],
                                     zxBCdt->nb[1], zxBCdt->nb[2], 0);
    ggml_tensor * xBC = ggml_view_3d(ctx0, zxBCdt, d_inner + 2 * n_group * d_state, n_seq_tokens, n_seqs, zxBCdt->nb[1],
                                     zxBCdt->nb[2], d_inner * ggml_element_size(zxBCdt));
    ggml_tensor * dt  = ggml_view_3d(ctx0, zxBCdt, n_head, n_seq_tokens, n_seqs, zxBCdt->nb[1], zxBCdt->nb[2],
                                     (2 * d_inner + 2 * n_group * d_state) * ggml_element_size(zxBCdt));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner + 2*n_group*d_state, n_seqs}
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);

        // copy last (d_conv - 1) columns back into the state cache
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs,
                                               conv_x->nb[1], conv_x->nb[2], n_seq_tokens * (conv_x->nb[0]));

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv,
                                               ggml_view_1d(ctx0, conv_states_all,
                                                            (d_conv - 1) * (d_inner + 2 * n_group * d_state) * (n_seqs),
                                                            kv_head * (d_conv - 1) * (d_inner + 2 * n_group * d_state) *
                                                                ggml_element_size(conv_states_all))));

        // 1D convolution
        // The equivalent is to make a self-overlapping view of conv_x
        // over d_conv columns at each stride in the 3rd dimension,
        // then element-wise multiply that with the conv1d weight,
        // then sum the elements of each row,
        // (the last two steps are a dot product over rows (also doable with mul_mat))
        // then permute away the ne[0] dimension,
        // and then you're left with the resulting x tensor.
        // For simultaneous sequences, all sequences need to have the same length.
        xBC = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);

        // bias
        xBC = ggml_add(ctx0, xBC, model.layers[il].ssm_conv1d_b);

        xBC = ggml_silu(ctx0, xBC);
    }

    // ssm
    {
        // These correspond to V K Q in SSM/attention duality
        ggml_tensor * x = ggml_view_4d(ctx0, xBC, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * xBC->nb[0],
                                       xBC->nb[1], xBC->nb[2], 0);
        ggml_tensor * B = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                       xBC->nb[1], xBC->nb[2], d_inner * ggml_element_size(xBC));
        ggml_tensor * C = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                       xBC->nb[1], xBC->nb[2], (d_inner + n_group * d_state) * ggml_element_size(xBC));

        // {n_head, n_seq_tokens, n_seqs}
        dt = ggml_add(ctx0, ggml_cont(ctx0, dt), model.layers[il].ssm_dt_b);

        ggml_tensor * A = model.layers[il].ssm_a;

        // use the states and the indices provided by build_recurrent_state
        // (this is necessary in order to properly use the states before they are overwritten,
        //  while avoiding to make unnecessary copies of the states)
        auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
            ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());

            // TODO: use semistructured matrices to implement state-space duality
            // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
            return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
        };

        ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);

        // store last states
        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0, ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_seqs, ggml_nelements(x) * x->nb[0]),
                         ggml_view_1d(ctx0, ssm_states_all, d_state * d_inner * n_seqs,
                                      kv_head * d_state * d_inner * ggml_element_size(ssm_states_all))));

        ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs, x->nb[1], n_head * x->nb[1],
                                       n_seq_tokens * n_head * x->nb[1], 0);

        // TODO: skip computing output earlier for unused tokens

        y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d));
        cb(y, "mamba2_y_add_d", il);
        y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

        // grouped RMS norm
        if (model.layers[il].ssm_norm) {
            y = ggml_reshape_4d(ctx0, y, d_inner / n_group, n_group, n_seq_tokens, n_seqs);
            y = build_norm(y, model.layers[il].ssm_norm, NULL, LLM_NORM_RMS, il);
        }

        y = ggml_reshape_3d(ctx0, y, d_inner, n_seq_tokens, n_seqs);

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = build_lora_mm(model.layers[il].ssm_out, y);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
    cb(cur, "mamba_out", il);

    return cur;
}
