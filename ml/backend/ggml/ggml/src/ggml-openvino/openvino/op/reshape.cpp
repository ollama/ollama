#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_reshape(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    if (context.get_input_shape(0) == context.get_output_shape(0)) {
        return {context.get_input(0)};
    }

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported RESHAPE case");

    auto output_shape = context.get_output_shape(0).to_shape();
    std::shared_ptr<ov::Node> new_shape_node;
    if (op_case == 1) {
        new_shape_node =
            ov::op::v0::Constant::create(ov::element::i64,
                                         {3},
                                         std::vector<int64_t>{-1, (int64_t)output_shape[1], (int64_t)output_shape[2]});
    } else if (op_case == 2) {
        new_shape_node =
            ov::op::v0::Constant::create(ov::element::i64,
                                         {3},
                                         std::vector<int64_t>{(int64_t)output_shape[0], -1, (int64_t)output_shape[2]});
    } else {
        new_shape_node =
            ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{(int64_t) output_shape[0], -1, 1});
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
