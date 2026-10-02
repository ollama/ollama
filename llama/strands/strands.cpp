// Strands Decider pointer head, following the Apache-2.0 reference:
// https://github.com/strands-labs/strands-decider/blob/main/src/strands_decider/modeling.py
#include "strands.h"
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
}

struct strands_head::impl {
    int hidden, dim;
    std::vector<float> norm_weight, norm_bias, q_weight, q_bias, k_weight, k_bias;
    float temperatures[3];

    explicit impl(const std::string &path) {
        ggml_context *tensors = nullptr;
        auto meta = gguf_init_from_file(path.c_str(), {true, &tensors});
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(meta, gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(tensors, ggml_free);
        std::ifstream file(path, std::ios::binary);
        require(meta && tensors && file.good(), "Strands: cannot open model");
        const int arch = gguf_find_key(meta, "general.architecture");
        require(arch >= 0 && gguf_get_kv_type(meta, arch) == GGUF_TYPE_STRING, "Strands: missing architecture");
        const std::string prefix = std::string(gguf_get_val_str(meta, arch)) + ".decision.";
        auto integer = [&](const char *name) {
            const int key = gguf_find_key(meta, (prefix + name).c_str());
            require(key >= 0 && gguf_get_kv_type(meta, key) == GGUF_TYPE_UINT32, "Strands: missing head dimensions");
            return gguf_get_val_u32(meta, key);
        };
        hidden = integer("hidden_size");
        dim = integer("pointer_dim");
        require(hidden > 0 && hidden <= 16384 && dim > 0 && dim <= 4096, "Strands: invalid head dimensions");
        auto read = [&](const char *name, int rows, int cols) {
            std::string full = std::string("strands.") + name;
            int index = gguf_find_tensor(meta, full.c_str());
            require(index >= 0, "Strands: missing head tensor");
            auto t = ggml_get_tensor(tensors, full.c_str());
            require(t && t->type == GGML_TYPE_F32 && t->ne[0] == cols && t->ne[1] == rows &&
                    t->ne[2] == 1 && t->ne[3] == 1, "Strands: invalid head tensor shape or type");
            std::vector<float> data(size_t(rows) * cols);
            file.seekg(gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, index));
            file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
            require(file.good(), "Strands: truncated head tensor");
            for (float v : data) require(std::isfinite(v), "Strands: non-finite head weight");
            return data;
        };
        norm_weight = read("norm.weight", 1, hidden);
        norm_bias = read("norm.bias", 1, hidden);
        q_weight = read("q.weight", dim, hidden);
        q_bias = read("q.bias", 1, dim);
        k_weight = read("k.weight", dim, hidden);
        k_bias = read("k.bias", 1, dim);
        const char *names[] = {"temperature.noul", "temperature.choice", "temperature.score"};
        for (int i = 0; i < 3; ++i) {
            const int key = gguf_find_key(meta, (prefix + names[i]).c_str());
            require(key >= 0 && gguf_get_kv_type(meta, key) == GGUF_TYPE_FLOAT32, "Strands: missing temperature");
            temperatures[i] = gguf_get_val_f32(meta, key);
            require(std::isfinite(temperatures[i]) && temperatures[i] > 0, "Strands: invalid temperature");
        }
    }

    std::vector<float> project(const std::vector<float> &state, bool query) const {
        require(state.size() == size_t(hidden), "Strands: hidden size mismatch");
        double mean = 0, variance = 0;
        for (float v : state) mean += v;
        mean /= hidden;
        for (float v : state) variance += (v-mean)*(v-mean);
        const double scale = 1 / std::sqrt(variance/hidden + 1e-5);
        std::vector<float> normalized(hidden);
        for (int j = 0; j < hidden; ++j)
            normalized[j] = (state[j]-mean)*scale*norm_weight[j] + norm_bias[j];
        const auto &weight = query ? q_weight : k_weight;
        auto result = query ? q_bias : k_bias;
        for (int i = 0; i < dim; ++i) {
            float sum = 0;
            for (int j = 0; j < hidden; ++j) sum += weight[size_t(i)*hidden+j]*normalized[j];
            result[i] += sum;
        }
        return result;
    }

    std::vector<std::vector<float>> score(const std::vector<std::vector<float>> &states, const common_json &fields) const {
        require(!states.empty() && fields.is_array() && fields.size() == 1, "Strands: expected one independent question");
        auto last = [&](const common_json &span) {
            require(span.is_array() && span.size() == 2 && span[0].is_number_integer() && span[1].is_number_integer(),
                    "Strands: malformed token span");
            const int start = span[0].get<int>(), end = span[1].get<int>();
            require(start >= 0 && start < end && size_t(end) <= states.size(), "Strands: invalid token span");
            return end - 1;
        };
        const auto &field = fields[0];
        const int type = field.at("type").get<int>();
        require(type >= 0 && type < 3, "Strands: invalid question type");
        const int answer = last(field.at("question"));
        require(size_t(answer + 1) == states.size(), "Strands: answer must be the final token");
        const auto query = project(states[answer], true);
        const auto &options = field.at("options");
        require(options.is_array() && options.size() >= 2 && options.size() <= 255, "Strands: expected 2-255 options");
        std::vector<float> logits;
        for (const auto &option : options) {
            const int index = last(option);
            require(index < answer, "Strands: option must precede answer");
            const auto key = project(states[index], false);
            float value = 0;
            for (int j = 0; j < dim; ++j) value += query[j]*key[j];
            value /= std::sqrt(float(dim))*temperatures[type];
            require(std::isfinite(value), "Strands: non-finite logit");
            logits.push_back(value);
        }
        return {std::move(logits)};
    }
};

strands_head::strands_head(const std::string &path) : p(new impl(path)) {}
strands_head::~strands_head() = default;
std::vector<std::vector<float>> strands_head::score(const std::vector<std::vector<float>> &hidden, const common_json &fields) {
    return p->score(hidden, fields);
}
