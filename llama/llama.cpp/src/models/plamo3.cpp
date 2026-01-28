#include "models.h"

template <bool iswa>
llm_build_plamo3<iswa>::llm_build_plamo3(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t head_dim_q = hparams.n_embd_head_k;
    const int64_t head_dim_v = hparams.n_embd_head_v;

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>;
    inp_attn_type * inp_attn = nullptr;

    if constexpr (iswa) {
        inp_attn = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;

        float freq_base_l  = 0.0f;
        float freq_scale_l = 0.0f;
        if constexpr (iswa) {
            freq_base_l  = model.get_rope_freq_base (cparams, il);
            freq_scale_l = model.get_rope_freq_scale(cparams, il);
        } else {
            freq_base_l  = freq_base;
            freq_scale_l = freq_scale;
        }

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, cur);
        cb(cur, "wqkv", il);

        const int32_t n_head    = hparams.n_head(il);
        const int32_t n_head_kv = hparams.n_head_kv(il);

        const int64_t q_offset = 0;
        const int64_t k_offset = head_dim_q * n_head;
        const int64_t v_offset = k_offset + head_dim_q * n_head_kv;

        ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, head_dim_q, n_head, n_tokens,
                head_dim_q * sizeof(float), qkv->nb[1], q_offset * ggml_element_size(qkv));
        ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, head_dim_q, n_head_kv, n_tokens,
                head_dim_q * sizeof(float), qkv->nb[1], k_offset * ggml_element_size(qkv));
        ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, head_dim_v, n_head_kv, n_tokens,
                head_dim_v * sizeof(float), qkv->nb[1], v_offset * ggml_element_size(qkv));

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "attn_q_norm", il);
        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Kcur, "attn_k_norm", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow);

        const float attn_scale = 1.0f / sqrtf(float(head_dim_q));

        cur = build_attn(inp_attn,
                model.layers[il].wo, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, attn_scale, il);
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "attn_residual", il);

        residual = cur;

        cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                NULL,                      NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "ffn_residual", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// Explicit template instantiations
template struct llm_build_plamo3<false>;
template struct llm_build_plamo3<true>;
