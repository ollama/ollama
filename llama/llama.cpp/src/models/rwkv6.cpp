#include "models.h"

llm_build_rwkv6::llm_build_rwkv6(const llama_model & model, const llm_graph_params & params) :
    llm_build_rwkv6_base(model, params) {
    GGML_ASSERT(hparams.token_shift_count == 2);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);

    auto * rs_inp = build_rs_inp();

    const auto n_embd       = hparams.n_embd;
    const auto n_seq_tokens = ubatch.n_seq_tokens;
    const auto n_seqs       = ubatch.n_seqs;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const llama_layer * layer = &model.layers[il];
        inpL                      = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);

        ggml_tensor * token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il);

        ggml_tensor * att_shift =
            ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], 0);
        ggml_tensor * ffn_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1],
                                               token_shift->nb[2], n_embd * ggml_element_size(token_shift));

        ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM, il);
        cb(att_norm, "attn_norm", il);

        ggml_tensor * x_prev = ggml_concat(
            ctx0, att_shift,
            ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0), 1);

        cur = build_rwkv6_time_mix(rs_inp, att_norm, x_prev, ubatch, il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        ggml_tensor * ffn_norm = build_norm(ffn_inp, layer->attn_norm_2, layer->attn_norm_2_b, LLM_NORM, il);
        cb(ffn_norm, "ffn_norm", il);

        x_prev = ggml_concat(
            ctx0, ffn_shift,
            ggml_view_3d(ctx0, ffn_norm, n_embd, n_seq_tokens - 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2], 0), 1);

        token_shift = ggml_concat(ctx0,
                                  ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2],
                                               (n_seq_tokens - 1) * n_embd * ggml_element_size(att_norm)),
                                  ggml_view_3d(ctx0, ffn_norm, n_embd, 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2],
                                               (n_seq_tokens - 1) * n_embd * ggml_element_size(ffn_norm)),
                                  1);
        ggml_build_forward_expand(gf, build_rwkv_token_shift_store(token_shift, ubatch, il));

        ffn_inp  = ggml_reshape_2d(ctx0, ffn_inp, n_embd, n_tokens);
        ffn_norm = ggml_reshape_2d(ctx0, ffn_norm, n_embd, n_tokens);
        x_prev   = ggml_reshape_2d(ctx0, x_prev, n_embd, n_tokens);
        cur      = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

        if (il == n_layer - 1 && inp_out_ids) {
            ffn_inp  = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            ffn_norm = ggml_get_rows(ctx0, ffn_norm, inp_out_ids);
            x_prev   = ggml_get_rows(ctx0, x_prev, inp_out_ids);
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        cur = build_rwkv6_channel_mix(layer, ffn_norm, x_prev, LLM_ARCH_RWKV6);
        cur = ggml_add(ctx0, cur, ffn_inp);

        if (hparams.rescale_every_n_layers != 0 && (il + 1) % hparams.rescale_every_n_layers == 0) {
            cur = ggml_scale(ctx0, cur, 0.5F);
        }
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
