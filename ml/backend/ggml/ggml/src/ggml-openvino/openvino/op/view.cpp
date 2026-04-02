#include "../op_table.h"
#include "../utils.h"
#include <openvino/op/reshape.hpp>
namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    if (context.get_op_case() == 2) {
        auto dst_shape = context.get_output_shape().to_shape();
        return rename_outputs_with_suffix({process_view_input(context, 0, dst_shape[2] * dst_shape[3])},
                                          context.get_name());
    }
    // op_case 3
    if (context.get_op_case() == 3) {
        auto input = context.get_input(0);
        auto input_ov_shape = input.get_partial_shape();

        auto input_llama_shape = context.get_input_shape(0).to_shape();

        // if the input ov shape size is different from the input llama shape size, it means the input is already reshaped and we need to reshape it back to the original shape before slicing
        if (input_ov_shape.size() != input_llama_shape.size()) {
            input = std::make_shared<ov::op::v1::Reshape>(input, ov::op::v0::Constant::create(ov::element::i64, {input_llama_shape.size()}, input_llama_shape), false);
        }

        auto dst_shape = context.get_output_shape().to_shape();

        // find the index of dst_shape that is different from input shape, and use that index to slice the input
        int slice_dim = -1;
        for (size_t i = 0; i < dst_shape.size(); ++i) {
            if (dst_shape[i] != input_llama_shape[i]) {
                slice_dim = i;
                break;
            }
        }

        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {dst_shape[slice_dim]});
        auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim});
        auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
        return {sliced};
    }
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
