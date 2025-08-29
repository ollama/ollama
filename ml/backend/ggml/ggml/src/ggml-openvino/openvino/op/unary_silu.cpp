#include <openvino/core/node_output.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_unary_silu(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(input);
    auto res = std::make_shared<ov::op::v1::Multiply>(input, sigmoid);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
