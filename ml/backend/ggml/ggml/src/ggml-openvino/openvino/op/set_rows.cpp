#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_set_rows(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto data = context.get_input(0);
    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type(0));

    auto dst_shape = context.get_output_shape(0).to_shape();
    FRONT_END_OP_CONVERSION_CHECK(dst_shape[0] == 1, "Unsupported shape in SET_ROWS");

    if (context.is_static() && context.is_first_token()) {
        Output<Node> res;
        if (context.get_op_case() == 2) {
            res = std::make_shared<ov::op::v1::Reshape>(
                data,
                ov::op::v0::Constant::create(
                    ov::element::i64,
                    {3},
                    {context.get_context_size(), context.get_num_heads_kv(), context.get_head_size()}),
                false);
            res = std::make_shared<ov::op::v1::Transpose>(
                res, ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 0}));
        } else {
            res = data;
        }
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    auto indices = context.get_input(1);
    auto dst = context.get_input(context.get_output_name());

    auto zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto dst_reshaped = std::make_shared<ov::op::v1::Reshape>(
        dst,
        ov::op::v0::Constant::create(ov::element::i64, {2}, {(int64_t) dst_shape[1], (int64_t) dst_shape[2]}),
        false);
    auto indices_reshaped =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
    auto data_reshaped = std::make_shared<ov::op::v0::Squeeze>(data, zero);
    auto updated = std::make_shared<ov::op::v3::ScatterUpdate>(dst_reshaped, indices_reshaped, data_reshaped, zero);
    auto res = std::make_shared<ov::op::v1::Reshape>(updated, std::make_shared<ov::op::v0::ShapeOf>(dst), false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
