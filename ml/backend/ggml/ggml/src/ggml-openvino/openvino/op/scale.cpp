#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_scale(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    float scale;
    memcpy(&scale, context.get_output_op_params(0), sizeof(float));
    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});

    auto res = std::make_shared<ov::op::v1::Multiply>(context.get_input(0), scale_node);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
