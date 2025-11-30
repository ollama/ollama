#include "models.h"


llm_build_arwkv7::llm_build_arwkv7(const llama_model & model, const llm_graph_params & params) : llm_build_rwkv7_base(model, params) {
    GGML_ASSERT(n_embd == hparams.n_embd_r());

    ggml_tensor * cur;
    ggml_tensor * inpL;
    ggml_tensor * v_first = nullptr;

    inpL = build_inp_embd(model.tok_embd);

    auto * rs_inp = build_rs_inp();

    const auto n_embd = hparams.n_embd;
    const auto n_seq_tokens = ubatch.n_seq_tokens;
    const auto n_seqs = ubatch.n_seqs;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const llama_layer * layer = &model.layers[il];
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);

        ggml_tensor * token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il);

        ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM_RMS, il);
        cb(att_norm, "attn_norm", il);

        ggml_tensor * x_prev = ggml_concat(
                ctx0,
                token_shift,
                ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0),
                1
                );

        cur = build_rwkv7_time_mix(rs_inp, att_norm, x_prev, v_first, ubatch, il);

        token_shift = ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(att_norm));
        ggml_build_forward_expand(gf, build_rwkv_token_shift_store(token_shift, ubatch, il));

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur     = ggml_reshape_2d(ctx0, cur,     n_embd, n_tokens);
        ffn_inp = ggml_reshape_2d(ctx0, ffn_inp, n_embd, n_tokens);

        if (il == n_layer - 1 && inp_out_ids) {
            cur     = ggml_get_rows(ctx0, cur,     inp_out_ids);
            ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
        }
        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
