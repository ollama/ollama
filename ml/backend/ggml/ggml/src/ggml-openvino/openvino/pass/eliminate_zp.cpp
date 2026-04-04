#include "eliminate_zp.h"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/parallel.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/pass/pattern/op/label.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

EliminateZeroPoints::EliminateZeroPoints() {
    // Find pattern:
    // (Multiply Any(scale)
    //           (Subtract (Convert Constant(data)))
    //                     (Convert Constant(zero_point)))
    // where zero_point is a scalar
    // If data is u4 and zp value is 8 (q4_0), Replace the Subtract with an i4 Constant whose value is data - zp_val
    // If data is u8 and zp value is 128 (q8_0) or 32 (q6_k), Replace the Subtract with an i8 Constant

    auto m_data_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_data_convert = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_data_constant});

    auto m_zp_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_zp_convert = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_zp_constant});

    auto m_subtract = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({m_data_convert, m_zp_convert});
    auto m_scale = ov::pass::pattern::any_input();
    auto m_multiply = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_scale, m_subtract});

    const auto callback = [=](ov::pass::pattern::Matcher & m) {
        const auto & pattern_map = m.get_pattern_value_map();

        auto multiply_node =
            std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(m_multiply).get_node_shared_ptr());
        auto subtract_node =
            std::dynamic_pointer_cast<ov::op::v1::Subtract>(pattern_map.at(m_subtract).get_node_shared_ptr());
        auto data_constant =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(m_data_constant).get_node_shared_ptr());
        auto zp_constant =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(m_zp_constant).get_node_shared_ptr());

        if (!multiply_node || !subtract_node || !data_constant || !zp_constant) {
            return false;
        }

        if (ov::shape_size(zp_constant->get_shape()) != 1) {
            return false;
        }

        auto data_type = data_constant->get_element_type();
        auto zp_data = zp_constant->cast_vector<int>();

        if (zp_data.empty()) {
            return false;
        }

        int zp_value = zp_data[0];

        bool should_eliminate = false;
        ov::element::Type target_type;

        if (data_type == ov::element::u4 && zp_value == 8) {
            should_eliminate = true;
            target_type = ov::element::i4;
        } else if (data_type == ov::element::u8 && (zp_value == 128 || zp_value == 32)) {
            should_eliminate = true;
            target_type = ov::element::i8;
        }

        if (!should_eliminate) {
            return false;
        }

        auto data_shape = data_constant->get_shape();
        size_t total_elements = ov::shape_size(data_shape);

        std::shared_ptr<ov::op::v0::Constant> new_constant;

        // TODO improve performance
        if (data_type == ov::element::u4) {
            auto data_values = data_constant->cast_vector<uint8_t>();
            std::vector<int8_t> adjusted_values(total_elements);

            ov::parallel_for(total_elements, [&](size_t i) {
                adjusted_values[i] = static_cast<int8_t>(static_cast<int>(data_values[i]) - 8);
            });

            new_constant = std::make_shared<ov::op::v0::Constant>(target_type, data_shape, adjusted_values);
        } else if (data_type == ov::element::u8) {
            auto data_values = data_constant->cast_vector<uint8_t>();
            std::vector<int8_t> adjusted_values(total_elements);

            ov::parallel_for(total_elements, [&, zp_value](size_t i) {
                adjusted_values[i] = static_cast<int8_t>(static_cast<int>(data_values[i]) - zp_value);
            });

            new_constant = std::make_shared<ov::op::v0::Constant>(target_type, data_shape, adjusted_values);
        }

        auto new_convert =
            std::make_shared<ov::op::v0::Convert>(new_constant, subtract_node->get_output_element_type(0));
        ov::replace_node(subtract_node, new_convert);

        return true;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(m_multiply, "ov::frontend::ggml::pass::EliminateZeroPoints"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
