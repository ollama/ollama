#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/transpose.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_transpose(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 1, 3, 2}));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
