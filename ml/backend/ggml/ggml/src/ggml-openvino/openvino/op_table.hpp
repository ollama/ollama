#pragma once

#include "node_context.hpp"

namespace ov {
namespace frontend {
namespace ggml {

namespace op {

#define GGML_OP_CONVERTER(op) OutputVector op(const NodeContext& context)

GGML_OP_CONVERTER(translate_add);
GGML_OP_CONVERTER(translate_cont);
GGML_OP_CONVERTER(translate_get_rows);
GGML_OP_CONVERTER(translate_mul);
GGML_OP_CONVERTER(translate_mulmat);
GGML_OP_CONVERTER(translate_permute);
GGML_OP_CONVERTER(translate_reshape);
GGML_OP_CONVERTER(translate_rms_norm);
GGML_OP_CONVERTER(translate_rope);
GGML_OP_CONVERTER(translate_scale);
GGML_OP_CONVERTER(translate_unary_silu);
GGML_OP_CONVERTER(translate_soft_max);
GGML_OP_CONVERTER(translate_transpose);
GGML_OP_CONVERTER(translate_view);
GGML_OP_CONVERTER(translate_glu_swiglu);
GGML_OP_CONVERTER(translate_set_rows);

} // namespace op

std::unordered_map<std::string, CreatorFunction> get_supported_ops();

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
