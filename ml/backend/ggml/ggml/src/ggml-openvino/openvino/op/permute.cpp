#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_permute(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3 || op_case == 4,
                                "Unsupported PERMUTE case");

    ov::Output<Node> res;
    auto src = context.get_input(0);
    auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});

    if (op_case == 1 || context.is_stateful()) {
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } else if (op_case == 4) {
        auto output_shape = context.get_output_shape().to_shape();
        auto n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        auto head_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        auto n_seq_active = context.has_input("n_seq_active") ?
                                context.get_input("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        auto new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, n_heads, head_size}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        auto reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, true);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } else {
        auto cache_shape = src.get_partial_shape();
        auto output_shape = context.get_output_shape().to_shape();
        int64_t head_size = output_shape[3];
        int64_t n_heads = output_shape[1];
        int64_t ctx_per_seq = cache_shape[2].is_static() ? cache_shape[2].get_length() : -1;
        int64_t n_seq = cache_shape[1].get_length();

        Output<Node> attention_size;
        if (!context.has_input("attention_size")) {
            attention_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } else if (op_case == 2) {
            attention_size = context.get_input("attention_size");
        } else {
            attention_size = context.get_input("attention_size_swa");
        }

        Output<Node> seq_active_start;
        Output<Node> seq_active_end;
        if (context.has_input("seq_active_start")) {
            seq_active_start = context.get_input("seq_active_start");
            seq_active_end = context.get_input("seq_active_end");
        } else {
            int64_t n_seq_active = output_shape[0];
            size_t offset = *((size_t *) context.get_input_op_params(0));
            int64_t seq_active_start_val = offset / context.get_input_stride(0)[0];
            int64_t seq_active_end_val = seq_active_start_val + n_seq_active;
            seq_active_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_start_val});
            seq_active_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_end_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size]
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        auto src_reshaped = std::make_shared<ov::op::v1::Reshape>(
            src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_per_seq, n_heads, head_size}), false);
        auto slice1 = std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
        auto slice2 = std::make_shared<ov::op::v8::Slice>(slice1, zero, attention_size, one, one);
        res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
