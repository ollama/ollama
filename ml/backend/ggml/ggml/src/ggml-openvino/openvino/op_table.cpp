#include "op_table.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/subtract.hpp>

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {

std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
    using namespace ov::op;
    return {
        {"GGML_OP_ADD",        op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_ADD1",       op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_CONT",       op::translate_cont                             },
        {"GGML_OP_DIV",        op::translate_1to1_match_2_inputs<v1::Divide>  },
        {"GGML_OP_GET_ROWS",   op::translate_get_rows                         },
        {"GGML_OP_MUL",        op::translate_1to1_match_2_inputs<v1::Multiply>},
        {"GGML_OP_MUL_MAT",    op::translate_mulmat                           },
        {"GGML_OP_PERMUTE",    op::translate_permute                          },
        {"GGML_OP_RESHAPE",    op::translate_reshape                          },
        {"GGML_OP_RMS_NORM",   op::translate_rms_norm                         },
        {"GGML_OP_ROPE",       op::translate_rope                             },
        {"GGML_OP_SCALE",      op::translate_scale                            },
        {"GGML_OP_SOFT_MAX",   op::translate_soft_max                         },
        {"GGML_OP_SUB",        op::translate_1to1_match_2_inputs<v1::Subtract>},
        {"GGML_OP_TRANSPOSE",  op::translate_transpose                        },
        {"GGML_UNARY_OP_SILU", op::translate_unary_silu                       },
        {"GGML_OP_VIEW",       op::translate_view                             },
        {"GGML_GLU_OP_SWIGLU", op::translate_glu_swiglu                       },
        {"GGML_OP_SET_ROWS",   op::translate_set_rows                         },
    };
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
