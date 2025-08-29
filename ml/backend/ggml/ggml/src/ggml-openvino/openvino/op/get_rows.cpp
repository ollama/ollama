#include <cstdint>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_get_rows(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2, "Unsupported CONT case");

    Output<Node> res;
    auto data = context.get_input(0);
    auto indices = context.get_input(1);

    if (op_case == 2) {
        // The input comes from a VIEW
        indices = process_view_input(context, 1);
    }

    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    if (indices.get_partial_shape()[1].get_length() == 1) {
        indices =
            std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    } else {
        indices =
            std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
    }

    if (res.get_element_type() != context.get_output_type(0)) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type(0));
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
