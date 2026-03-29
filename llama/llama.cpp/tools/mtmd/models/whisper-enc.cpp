#include "models.h"

ggml_cgraph * clip_graph_whisper_enc::build() {
    const int n_frames = img.nx;
    const int n_pos    = n_frames / 2;
    GGML_ASSERT(model.position_embeddings->ne[1] >= n_pos);

    ggml_tensor * inp = build_inp_raw(1);

    // conv1d block
    {
        // convolution + gelu
        ggml_tensor * cur = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_1_b);

        cur = ggml_gelu_erf(ctx0, cur);

        cur = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_2_b);

        cur = ggml_gelu_erf(ctx0, cur);
        // transpose
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cb(inp, "after_conv1d", -1);
    }

    // sanity check (only check one layer, but it should be the same for all)
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
    GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
    GGML_ASSERT(model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].v_b);
    GGML_ASSERT(!model.layers[0].k_b); // no bias for k

    ggml_tensor * pos_embd_selected = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], n_pos,
        model.position_embeddings->nb[1], 0
    );
    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            pos_embd_selected,
                            nullptr);

    cb(cur, "after_transformer", -1);

    if (model.audio_has_stack_frames()) {
        // StackAudioFrames
        // https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/blob/main/ultravox_model.py
        cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
        cb(cur, "after_stacked", -1);
    }

    if (proj_type == PROJECTOR_TYPE_ULTRAVOX) {
        // UltravoxProjector
        // pre-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);

        // ffn in
        cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);

        // swiglu
        // see SwiGLU in ultravox_model.py, the second half passed through is silu, not the first half
        cur = ggml_swiglu_swapped(ctx0, cur);

        // mid-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_mid_w);

        // ffn out
        cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);

    } else if (proj_type == PROJECTOR_TYPE_QWEN2A) {
        // projector
        cur = ggml_mul_mat(ctx0, model.mm_fc_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_fc_b);

    } else if (proj_type == PROJECTOR_TYPE_VOXTRAL) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_MUSIC_FLAMINGO) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_GLMA) {
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
            cur = ggml_add(ctx0, cur, model.mm_norm_pre_b);
            cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
            cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_2_w, model.mm_2_b, hparams.ffn_op, 0);
            cur = ggml_concat(ctx0, model.mm_boi, cur, 1);
            cur = ggml_concat(ctx0, cur, model.mm_eoi, 1);
    } else {
        GGML_ABORT("%s: unknown projector type", __func__);
    }

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
