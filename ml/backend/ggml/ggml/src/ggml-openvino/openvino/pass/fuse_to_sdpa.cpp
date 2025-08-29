#include "fuse_to_sdpa.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/pass/pattern/op/label.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

FuseToSDPA::FuseToSDPA() {
    const auto m_k = ov::pass::pattern::any_input();
    const auto m_q = ov::pass::pattern::any_input();
    const auto m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    const auto m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    const auto m_scale = ov::pass::pattern::any_input();
    const auto m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    const auto m_mask = ov::pass::pattern::any_input();
    const auto m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_scaled_qk, m_mask});
    const auto m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    const auto m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    const auto m_v = ov::pass::pattern::any_input();
    const auto m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    const auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto k = pattern_to_output[m_k];
        auto q = pattern_to_output[m_q];
        auto v = pattern_to_output[m_v];
        auto mask = pattern_to_output[m_mask];
        auto scale = pattern_to_output[m_scale];

        auto v_trans =
            register_new_node<ov::op::v1::Transpose>(v, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 2, 1}));
        auto mask_f16 = register_new_node<ov::op::v0::Convert>(mask, ov::element::f16);
        auto scale_f16 = register_new_node<ov::op::v0::Convert>(scale, ov::element::f16);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v_trans, mask_f16, scale_f16, false);

        ov::replace_node(m.get_match_root(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
