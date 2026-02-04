#include "models.h"

llm_build_plm::llm_build_plm(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const float kq_scale = 1.0f/sqrtf(float(hparams.n_embd_head_k));

    const uint32_t n_embd_head_qk_rope = hparams.n_rot;
    const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * q = NULL;
            q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            cb(q, "q", il);

            // split into {n_head * n_embd_head_qk_nope, n_tokens}
            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(q->type, hparams.n_embd_head_k),
                    ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                    0);
            cb(q_nope, "q_nope", il);

            // and {n_head * n_embd_head_qk_rope, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                    ggml_row_size(q->type, hparams.n_embd_head_k),
                    ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                    ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
            ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_pe_compresseed, "kv_pe_compresseed", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_compressed = ggml_view_2d(ctx0, kv_pe_compresseed, kv_lora_rank, n_tokens,
                    kv_pe_compresseed->nb[1],
                    0);
            cb(kv_compressed, "kv_compressed", il);

            // and {n_embd_head_qk_rope, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_pe_compresseed, n_embd_head_qk_rope, 1, n_tokens,
                    kv_pe_compresseed->nb[1],
                    kv_pe_compresseed->nb[1],
                    ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            kv_compressed = build_norm(kv_compressed,
                    model.layers[il].attn_kv_a_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(kv_compressed, "kv_compressed", il);

            // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
            ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
            cb(kv, "kv", il);

            // split into {n_head * n_embd_head_qk_nope, n_tokens}
            ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                    ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                    0);
            cb(k_nope, "k_nope", il);

            // and {n_head * n_embd_head_v, n_tokens}
            ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v, n_head, n_tokens,
                    ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                    ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)*n_head),
                    ggml_row_size(kv->type, (n_embd_head_qk_nope)));
            cb(v_states, "v_states", il);

            v_states = ggml_cont(ctx0, v_states);
            cb(v_states, "v_states", il);

            v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v * n_head, n_tokens,
                    ggml_row_size(kv->type, hparams.n_embd_head_v * n_head),
                    0);
            cb(v_states, "v_states", il);

            q_pe = ggml_rope_ext(
                    ctx0, q_pe, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(q_pe, "q_pe", il);

            // shared RoPE key
            k_pe = ggml_rope_ext(
                    ctx0, k_pe, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(k_pe, "k_pe", il);

            ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
            cb(q_states, "q_states", il);

            ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
            cb(k_states, "k_states", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    q_states, k_states, v_states, nullptr, nullptr, nullptr, kq_scale, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

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

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
