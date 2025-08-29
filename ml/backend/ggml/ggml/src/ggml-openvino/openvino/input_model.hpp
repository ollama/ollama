#pragma once

#include <openvino/frontend/input_model.hpp>

#include "decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::ggml::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgmlDecoder>& gdecoder);

    const std::shared_ptr<GgmlDecoder>& get_model_decoder() const;

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
