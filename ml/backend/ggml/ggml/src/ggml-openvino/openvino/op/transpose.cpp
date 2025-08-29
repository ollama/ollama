#include <openvino/op/transpose.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_transpose(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto perm = argsort_descend(context.get_output_stride(0));
    auto res = std::make_shared<ov::op::v1::Transpose>(context.get_input(0),
                                                       ov::op::v0::Constant::create(ov::element::i64, {3}, perm));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
