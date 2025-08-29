#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_permute(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported CONT case");
    ov::Output<Node> res;

    if (op_case == 1) {
        auto perm = argsort_descend(context.get_output_stride(0));
        res = std::make_shared<ov::op::v1::Transpose>(context.get_input(0),
                                                      ov::op::v0::Constant::create(ov::element::i64, {3}, perm));
    } else {
        auto src = context.get_input(0);
        auto attention_size = context.get_input("attention_size");
        if (context.is_static()) {
            attention_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        }

        auto src_shape_ = context.get_input_shape(0).to_shape();
        std::vector<int64_t> src_shape(src_shape_.begin(), src_shape_.end());

        std::shared_ptr<ov::Node> src_reshaped;
        if (op_case == 2) {
            src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src,
                ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{-1, src_shape[1], src_shape[2]}),
                false);
        } else {
            src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src,
                ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{src_shape[1], src_shape[0], -1}),
                false);
        }

        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        std::shared_ptr<ov::Node> slice_axis;
        if (op_case == 2) {
            slice_axis = zero;
        } else {
            slice_axis = two;
        }
        auto src_slice = std::make_shared<ov::op::v8::Slice>(src_reshaped, zero, attention_size, one, slice_axis);

        if (op_case == 2) {
            res = std::make_shared<ov::op::v1::Transpose>(src_slice, ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 0, 2}));
        } else {
            res = src_slice;
        }
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
