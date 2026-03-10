#include "models.h"

ggml_cgraph * clip_graph_paddleocr::build() {
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, 10000, 1, 0, 1, 32, 1);
    };

    ggml_tensor * learned_pos_embd = resize_position_embeddings();
    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        // mlp_AR paddleocr projector
        float proj_norm_eps = 1e-5;
        cur = build_norm(cur,
                    model.mm_input_norm_w, model.mm_input_norm_b,
                    NORM_TYPE_NORMAL, proj_norm_eps, -1);

        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);
        cur = build_ffn(cur,
                    model.mm_1_w, model.mm_1_b,
                    nullptr, nullptr,
                    model.mm_2_w, model.mm_2_b,
                    hparams.ffn_op, -1);
        cb(cur, "mlp_out", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
