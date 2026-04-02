#include "frontend.h"

#include "input_model.h"
#include "op_table.h"
#include "translate_session.h"

namespace ov {
namespace frontend {
namespace ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr & model, bool naive) {
    auto ggml_model = std::dynamic_pointer_cast<ggml::InputModel>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    const auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    return converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
