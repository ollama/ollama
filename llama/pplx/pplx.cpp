// Port of the Apache-2.0 perplexity-ai/pplx-decider-v1-27b readout.
#include "pplx.h"
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
}

pplx_head::pplx_head(const std::string & path) {
    ggml_context * tensors = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
        gguf_init_from_file(path.c_str(), {true, &tensors}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(tensors, ggml_free);
    require(meta && ctx, "PPLX: cannot read model metadata");
    const int arch = gguf_find_key(meta.get(), "general.architecture");
    require(arch >= 0 && gguf_get_kv_type(meta.get(), arch) == GGUF_TYPE_STRING,
            "PPLX: missing model architecture");
    const auto key = std::string(gguf_get_val_str(meta.get(), arch)) + ".decision.temperature";
    const int temp = gguf_find_key(meta.get(), key.c_str());
    require(temp >= 0 && gguf_get_kv_type(meta.get(), temp) == GGUF_TYPE_FLOAT32,
            "PPLX: missing calibration temperature");
    temperature = gguf_get_val_f32(meta.get(), temp);
    require(std::isfinite(temperature) && temperature > 0, "PPLX: invalid calibration temperature");
    const int index = gguf_find_tensor(meta.get(), "pplx.readout.weight");
    const auto tensor = ggml_get_tensor(ctx.get(), "pplx.readout.weight");
    require(index >= 0 && tensor && tensor->type == GGML_TYPE_F32 &&
            tensor->ne[0] > 0 && tensor->ne[0] <= 16384 && tensor->ne[1] == 255 &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1, "PPLX: invalid readout tensor");
    width = tensor->ne[0];
    weights.resize(size_t(width) * 255);
    std::ifstream file(path, std::ios::binary);
    file.seekg(gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), index));
    file.read(reinterpret_cast<char *>(weights.data()), weights.size() * sizeof(float));
    require(file.good(), "PPLX: truncated readout tensor");
}

std::vector<float> pplx_head::score(const std::vector<float> & hidden, int options) const {
    require(hidden.size() == size_t(width), "PPLX: hidden state width mismatch");
    require(options >= 1 && options <= 255, "PPLX: expected 1-255 options");
    std::vector<float> logits(options);
    for (int i = 0; i < options; ++i) {
        double sum = 0;
        for (int j = 0; j < width; ++j) sum += double(hidden[j]) * weights[size_t(i) * width + j];
        logits[i] = sum / temperature;
        require(std::isfinite(logits[i]), "PPLX: non-finite logit");
    }
    return logits;
}
