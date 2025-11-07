#include "models.h"

llm_build_wavtokenizer_dec::llm_build_wavtokenizer_dec(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));

    cur = ggml_conv_1d_ph(ctx0, model.conv1d, cur, 1, 1);
    cur = ggml_add(ctx0, cur, model.conv1d_b);

    // posnet
    for (uint32_t il = 0; il < hparams.posnet.n_layer; ++il) {
        const auto & layer = model.layers[il].posnet;

        inpL = cur;

        switch (il) {
            case 0:
            case 1:
            case 3:
            case 4:
                {
                    cur = build_norm(cur,
                            layer.norm1,
                            layer.norm1_b,
                            LLM_NORM_GROUP, 0);

                    cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                    cur = ggml_conv_1d_ph(ctx0, layer.conv1, cur, 1, 1);
                    cur = ggml_add(ctx0, cur, layer.conv1_b);

                    cur = build_norm(cur,
                            layer.norm2,
                            layer.norm2_b,
                            LLM_NORM_GROUP, 0);

                    cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                    cur = ggml_conv_1d_ph(ctx0, layer.conv2, cur, 1, 1);
                    cur = ggml_add(ctx0, cur, layer.conv2_b);

                    cur = ggml_add(ctx0, cur, inpL);
                } break;
            case 2:
                {
                    cur = build_norm(cur,
                            layer.attn_norm,
                            layer.attn_norm_b,
                            LLM_NORM_GROUP, 0);

                    ggml_tensor * q;
                    ggml_tensor * k;
                    ggml_tensor * v;

                    q = ggml_conv_1d_ph(ctx0, layer.attn_q, cur, 1, 1);
                    k = ggml_conv_1d_ph(ctx0, layer.attn_k, cur, 1, 1);
                    v = ggml_conv_1d_ph(ctx0, layer.attn_v, cur, 1, 1);

                    q = ggml_add(ctx0, q, layer.attn_q_b);
                    k = ggml_add(ctx0, k, layer.attn_k_b);
                    v = ggml_add(ctx0, v, layer.attn_v_b);

                    q = ggml_cont(ctx0, ggml_transpose(ctx0, q));
                    k = ggml_cont(ctx0, ggml_transpose(ctx0, k));

                    ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);

                    kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f/sqrtf(float(hparams.posnet.n_embd)), 0.0f);

                    cur = ggml_mul_mat(ctx0, kq, v);

                    cur = ggml_conv_1d_ph(ctx0, layer.attn_o, cur, 1, 1);
                    cur = ggml_add(ctx0, cur, layer.attn_o_b);

                    cur = ggml_add(ctx0, cur, inpL);
                } break;
            case 5:
                {
                    cur = build_norm(cur,
                            layer.norm,
                            layer.norm_b,
                            LLM_NORM_GROUP, 0);
                } break;
            default: GGML_ABORT("unknown posnet layer");
        };
    }
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur = build_norm(cur,
            model.tok_norm,
            model.tok_norm_b,
            LLM_NORM, -1);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    inpL = cur;

    // convnext
    for (uint32_t il = 0; il < hparams.convnext.n_layer; ++il) {
        const auto & layer = model.layers[il].convnext;

        cur = inpL;

        cur = ggml_conv_1d_dw_ph(ctx0, layer.dw, cur, 1, 1);
        cur = ggml_add(ctx0, cur, layer.dw_b);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = build_norm(cur,
                layer.norm,
                layer.norm_b,
                LLM_NORM, -1);

        cur = build_ffn(cur,
                layer.pw1, layer.pw1_b, NULL,
                NULL,      NULL,        NULL,
                layer.pw2, layer.pw2_b, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_SEQ, il);

        cur = ggml_mul(ctx0, cur, layer.gamma);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        inpL = ggml_add(ctx0, cur, inpL);
    }
    cur = inpL;

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    cur = build_norm(cur,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, -1);

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cur = ggml_add(ctx0, cur, model.output_b);

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
