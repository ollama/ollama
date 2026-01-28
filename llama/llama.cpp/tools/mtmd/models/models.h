#pragma once

#include "../clip-graph.h"

/*
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

struct clip_graph_siglip : clip_graph {
    clip_graph_siglip(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_pixtral : clip_graph {
    clip_graph_pixtral(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen2vl : clip_graph {
    clip_graph_qwen2vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen3vl : clip_graph {
    clip_graph_qwen3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_youtuvl : clip_graph {
    clip_graph_youtuvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_minicpmv : clip_graph {
    clip_graph_minicpmv(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_internvl : clip_graph {
    clip_graph_internvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llama4 : clip_graph {
    clip_graph_llama4(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_kimivl : clip_graph {
    clip_graph_kimivl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_cogvlm : clip_graph {
    clip_graph_cogvlm(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llava : clip_graph {
    clip_graph_llava(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_whisper_enc : clip_graph {
    clip_graph_whisper_enc(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_conformer : clip_graph {
    clip_graph_conformer(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_glm4v : clip_graph {
    clip_graph_glm4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_mobilenetv5 : clip_graph {
    clip_graph_mobilenetv5(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * rms_norm_2d(
        ggml_tensor * inp,
        ggml_tensor * weight,
        float eps = 1e-6f);

    ggml_tensor* pad_same_2d(
        ggml_tensor* inp,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int dilation_h = 1,
        int dilation_w = 1);

    ggml_tensor * build_edge_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_inverted_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_mobilenet_attn(
        ggml_tensor * inp,
        const mobilenetv5_block & block);
};
