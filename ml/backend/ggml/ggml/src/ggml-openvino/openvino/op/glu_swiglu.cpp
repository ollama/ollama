#include <cstdint>
#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_glu_swiglu(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    if (context.get_input_size() == 2) {
        src0 = context.get_input(0);
        src1 = context.get_input(1);
    } else {
        auto combined = context.get_input(0);
        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {2});
        auto split = std::make_shared<ov::op::v1::Split>(combined, split_axis, 2);
        src0 = split->output(0);
        src1 = split->output(1);
    }
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(src0);
    auto silu = std::make_shared<ov::op::v1::Multiply>(src0, sigmoid);
    auto res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
