#pragma once

#include "../clip-graph.h"

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
