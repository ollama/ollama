#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/util/op_types.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mulmat(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    ov::Output<Node> res;
    ov::Output<ov::Node> B = context.get_input(0);
    ov::Output<ov::Node> A = context.get_input(1);

    bool convert_out_type = false;
    if (ov::op::util::is_constant(B.get_node()) && context.get_input_type(0) != context.get_input_type(1)) {
        B = std::make_shared<ov::op::v0::Convert>(context.get_input(0), context.get_input_type(1));
    } else if (context.get_input_type(0) != context.get_input_type(1)) {
        A = std::make_shared<ov::op::v0::Convert>(context.get_input(1), context.get_input_type(0));
        convert_out_type = true;
    }

    auto B_shape = context.get_input_shape(0).to_shape();
    auto A_shape = context.get_input_shape(1).to_shape();
    int64_t A_batch = A_shape[0];
    int64_t B_batch = B_shape[0];
    auto A_batch_larger = A_batch > B_batch;
    Output<Node> Z = A_batch_larger ? B : A;
    int64_t factor = A_batch_larger ? A_batch / B_batch : B_batch / A_batch;
    if (factor > 1) {
        auto A_batch_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{A_batch});
        auto B_batch_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{B_batch});
        auto factor_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto Z_last_two_dim = get_dimensions(Z.get_node_shared_ptr(), {1, 2});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {1});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        Output<Node> batch_small = A_batch_larger ? B_batch_node : A_batch_node;
        Output<Node> batch_large = A_batch_larger ? A_batch_node : B_batch_node;
        auto broadcast_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch_small, factor_node, Z_last_two_dim}, 0);
        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape);

        auto new_Z_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch_large, Z_last_two_dim}, 0);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, false);
        }
        if (A_batch_larger) {
            B = Z;
        } else {
            A = Z;
        }

        if (convert_out_type) {
            auto result_lp = std::make_shared<ov::op::v0::MatMul>(A, B, false, true);
            res = std::make_shared<ov::op::v0::Convert>(result_lp, context.get_output_type(0));
        } else {
            res = std::make_shared<ov::op::v0::MatMul>(A, B, false, true);
        }

        return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
