#include "models.h"

llm_build_cogvlm::llm_build_cogvlm(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    const float   kq_scale    = 1.0f / sqrtf(float(n_embd_head));

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * inpL;
    ggml_tensor * cur;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    // check ubatch to see if we have input tokens (text)
    // or an input embedding vector (image)
    bool is_text;
    if (ubatch.token) {
        is_text = true;
    } else {
        is_text = false;
    }

    for (int il = 0; il < n_layer; ++il) {
        // get either the text or image weight tensors
        ggml_tensor *wqkv, *wo;
        ggml_tensor *ffn_gate, *ffn_down, *ffn_up;

        if (is_text) {
            wqkv     = model.layers[il].wqkv;
            wo       = model.layers[il].wo;
            ffn_gate = model.layers[il].ffn_gate;
            ffn_down = model.layers[il].ffn_down;
            ffn_up   = model.layers[il].ffn_up;
        } else {
            wqkv     = model.layers[il].visexp_attn_wqkv;
            wo       = model.layers[il].visexp_attn_wo;
            ffn_gate = model.layers[il].visexp_ffn_gate;
            ffn_down = model.layers[il].visexp_ffn_down;
            ffn_up   = model.layers[il].visexp_ffn_up;
        }

        ggml_tensor * inpSA = inpL;
        cur = build_norm(inpSA, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

        // build self attention
        {
            ggml_tensor * qkv = build_lora_mm(wqkv, cur);

            // split qkv into Q, K, V along the first dimension
            ggml_tensor * Qcur =
                ggml_view_3d(ctx0, qkv, n_embd_head, n_head, n_tokens, n_embd_head * sizeof(float), qkv->nb[1], 0);
            ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                              qkv->nb[1], n_embd * ggml_element_size(qkv));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                              qkv->nb[1], 2 * n_embd * ggml_element_size(qkv));

            Qcur = ggml_rope(ctx0, Qcur, inp_pos, n_embd_head, rope_type);
            Kcur = ggml_rope(ctx0, Kcur, inp_pos, n_embd_head, rope_type);

            cur = build_attn(inp_attn,
                wo, nullptr,
                Qcur, Kcur, Vcur,
                nullptr, nullptr, nullptr,
                kq_scale, il);
            cb(cur, "attn_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                ffn_up, NULL, NULL,
                ffn_gate, NULL, NULL,
                ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
