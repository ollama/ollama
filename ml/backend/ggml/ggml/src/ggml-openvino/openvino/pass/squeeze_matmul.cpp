#include "squeeze_matmul.h"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/pattern/op/label.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace opp = ov::pass::pattern;

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    auto m_act = opp::any_input();
    auto m_wei = opp::any_input();
    auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    const auto callback = [=](ov::pass::pattern::Matcher & m) {
        const auto & pattern_map = m.get_pattern_value_map();
        auto matmul_node =
            std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_node_shared_ptr());
        auto act = pattern_map.at(m_act);
        auto wei = pattern_map.at(m_wei);
        auto act_shape = act.get_partial_shape();
        auto wei_shape = wei.get_partial_shape();
        if (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            return false;
        }
        if (act_shape.rank().get_length() == 4 && wei_shape.rank().get_length() == 2) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            auto new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_node->get_transpose_a(),
                                                                   matmul_node->get_transpose_b());
            auto unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_node->get_friendly_name());
            ov::copy_runtime_info(matmul_node, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_node(matmul_node, unsqueezed_output);
            return true;
        }
        return false;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
