#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <stdexcept>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_reshape(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    if (context.get_input_shape(0) == context.get_output_shape()) {
        return {context.get_input(0)};
    }

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(
        op_case == 1 || op_case == 2 || op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6,
        "Unsupported RESHAPE case");

    auto output_shape = context.get_output_shape().to_shape();
    std::shared_ptr<ov::Node> new_shape_node;
    if (op_case == 1) {
        if (context.is_stateful()) {
            new_shape_node = ov::op::v0::Constant::create(
                ov::element::i64, {3},
                std::vector<int64_t>{-1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        } else {
            new_shape_node = ov::op::v0::Constant::create(
                ov::element::i64, {4},
                std::vector<int64_t>{(int64_t) output_shape[0], -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        }
    } else if (op_case == 2) {
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{(int64_t) output_shape[0], (int64_t) output_shape[1], -1, (int64_t) output_shape[3]});

    } else if (op_case == 3) {
        throw std::runtime_error("might be outdated RESHAPE case");
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64, {4}, std::vector<int64_t>{(int64_t) output_shape[0], (int64_t) output_shape[1], -1, 1});

    } else if (op_case == 4) {
        return {context.get_input(0).get_node_shared_ptr()->input_value(0)};

    } else if (op_case == 5) {
        if (context.is_stateful()) {
            std::vector<int64_t> shape_vec = {1, -1, (int64_t) context.get_output_shape().to_shape()[3]};
            new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {3}, shape_vec);
        } else {
            std::vector<int64_t> shape_vec = {1, 1, -1, (int64_t) context.get_output_shape().to_shape()[3]};
            new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {4}, shape_vec);
        }

        // // Alternative
        // auto token_len = context.get_input("token_len");
        // auto emb_size =
        //     ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) context.get_output_shape().to_shape()[3]});
        // auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        // new_shape_node = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, one, token_len, emb_size}, 0);

    } else if (op_case == 6) {
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {4}, context.get_output_shape().to_shape());
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
