#include "models.h"

#include <float.h>

llm_build_chameleon::llm_build_chameleon(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        if (hparams.swin_norm) {
            cur = inpL;
        } else {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            if (model.layers[il].attn_q_norm) {
                Qcur = ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens,
                        ggml_element_size(Qcur) * n_embd_head,
                        ggml_element_size(Qcur) * n_embd_head * n_head,
                        0);
                cb(Qcur, "Qcur", il);

                Qcur = build_norm(Qcur,
                        model.layers[il].attn_q_norm,
                        model.layers[il].attn_q_norm_b,
                        LLM_NORM, il);
                cb(Qcur, "Qcur", il);
            }

            if (model.layers[il].attn_k_norm) {
                Kcur = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens,
                        ggml_element_size(Kcur) * n_embd_head,
                        ggml_element_size(Kcur) * n_embd_head * n_head_kv,
                        0);
                cb(Kcur, "Kcur", il);

                Kcur = build_norm(Kcur,
                        model.layers[il].attn_k_norm,
                        model.layers[il].attn_k_norm_b,
                        LLM_NORM, il);
                cb(Kcur, "Kcur", il);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        if (hparams.swin_norm) {
            cur = build_norm(cur,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        if (!hparams.swin_norm) {
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
        }

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (hparams.swin_norm) {
            cur = build_norm(cur,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

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

    // lm_head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output_with_img_logits", -1);

    // TODO: this suppresses the output of image tokens, which is required to enable text-only outputs.
    // Needs to be removed once image outputs are supported.
    int img_token_end_idx = 8196;
    int img_token_start_idx = 4;
    int num_img_tokens = img_token_end_idx - img_token_start_idx;
    // creates 1d tensor of size num_img_tokens and values -FLT_MAX,
    // which ensures that text token values are always at least larger than image token values
    ggml_tensor * img_logits = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_img_tokens);
    img_logits = ggml_clamp(ctx0, img_logits, -FLT_MAX, -FLT_MAX);
    cb(img_logits, "img_logits", -1);

    cur = ggml_set_1d(ctx0, cur, img_logits, ggml_element_size(cur) * img_token_start_idx);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
