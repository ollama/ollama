#include "input_model.hpp"

#include "decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

InputModel::InputModel(const std::shared_ptr<GgmlDecoder>& gdecoder) : m_decoder(gdecoder) {}

const std::shared_ptr<GgmlDecoder>& InputModel::get_model_decoder() const {
    return m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
