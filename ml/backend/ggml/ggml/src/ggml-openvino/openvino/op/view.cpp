#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    return {context.get_input(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
