#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rope(const NodeContext& context) {
    num_inputs_check(context, 2, 3);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2, "Unsupported CONT case");

    ov::Output<Node> res;

    auto data_node = context.get_input(0).get_node_shared_ptr();
    auto output_shape = context.get_output_shape(0).to_shape();
    int32_t* op_params = context.get_output_op_params(0);

    Output<Node> cos_theta_node;
    Output<Node> sin_theta_node;
    if (context.has_input("rope_cos")) {
        cos_theta_node = context.get_input("rope_cos");
        sin_theta_node = context.get_input("rope_sin");
    } else {
        auto inp_pos = context.get_input(1).get_node_shared_ptr();
        std::shared_ptr<ov::Node> rope_freqs_weight;
        if (context.get_input_size() == 3) {
            rope_freqs_weight = context.get_input(2).get_node_shared_ptr();
        }
        auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight);
        sin_theta_node = sin_cos.first;
        cos_theta_node = sin_cos.second;
    }

    if (op_case == 2) {
        // The input comes from a VIEW
        int slice_len = output_shape[1] * output_shape[2];
        data_node = process_view_input(context, 0, slice_len).get_node_shared_ptr();
        auto data_shape = ov::op::v0::Constant::create(
            ov::element::i64, {3}, std::vector<int64_t>{-1, (int64_t) output_shape[1], (int64_t) output_shape[2]});
        data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
    }

    const int mode = op_params[2];
    constexpr int ROPE_TYPE_NEOX = 2;
    constexpr int ROPE_TYPE_NORM = 0;

    if (mode == ROPE_TYPE_NORM) {
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        auto even_slice = std::make_shared<ov::op::v8::Slice>(data_node, zero, end, two, two);
        auto odd_slice = std::make_shared<ov::op::v8::Slice>(data_node, one, end, two, two);

        Output<Node> first_half =
            std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Multiply>(even_slice, cos_theta_node),
                                                   std::make_shared<ov::op::v1::Multiply>(odd_slice, sin_theta_node));
        Output<Node> second_half =
            std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(even_slice, sin_theta_node),
                                              std::make_shared<ov::op::v1::Multiply>(odd_slice, cos_theta_node));

        first_half = std::make_shared<ov::op::v0::Unsqueeze>(first_half,
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
        second_half = std::make_shared<ov::op::v0::Unsqueeze>(second_half,
                                                              ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
        auto stack = std::make_shared<ov::op::v0::Concat>(OutputVector{first_half, second_half}, 3);
        res = std::make_shared<ov::op::v1::Reshape>(stack, std::make_shared<ov::op::v0::ShapeOf>(data_node), false);
    } else if (mode == ROPE_TYPE_NEOX) {
        auto data_split = std::make_shared<ov::op::v1::Split>(
            data_node, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2}), 2);
        Output<Node> slice_data_node_0 = data_split->outputs()[0];
        Output<Node> slice_data_node_1 = data_split->outputs()[1];

        auto first_half_node = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, cos_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, sin_theta_node));

        auto second_half_node = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, sin_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, cos_theta_node));

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_half_node, second_half_node}, 2);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
