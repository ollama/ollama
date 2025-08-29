#include <algorithm>
#include <openvino/runtime/core.hpp>

#include "ggml-backend-impl.h"
#include "ggml-decoder.h"
#include "ggml-impl.h"

enum ggml_status openvino_frontend_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph);

std::shared_ptr<GgmlOvDecoder> get_ggml_decoder(struct ggml_cgraph* cgraph, bool is_static, bool is_first_token);

ov::Tensor convert_ggml_input_to_ov(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string& name);

std::map<std::string, void*> get_ggml_graph_output_dst(std::shared_ptr<GgmlOvDecoder> ggml_decoder);

size_t checksum(const void* data, size_t size);

void print_input_tensor_info(const std::string& name, const ov::Tensor& tensor);

void print_output_tensor_info(const std::string& name,
                              const ov::Tensor& tensor,
                              std::map<std::string, void*>& output_dst);

template <typename T>
std::vector<T> pad_input(const ggml_tensor* tensor, size_t padded_rows, size_t padded_cols, T pad_value) {
    std::vector<T> padded_data(padded_rows * padded_cols, pad_value);
    size_t rows = tensor->ne[1];
    size_t cols = tensor->ne[0];
    T* data = static_cast<T*>(tensor->data);

    for (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        for (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded_data[i * padded_cols + j] = data[i * cols + j];
        }
    }
    return padded_data;
}

void set_zero_diagonal(std::vector<float>& matrix, size_t dim);

bool is_prefill(struct ggml_cgraph * cgraph);

ov::AnyMap get_npu_config();

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string& param_name);

bool is_naive(struct ggml_cgraph* cgraph);

enum ggml_status naive_compute(struct ggml_cgraph* cgraph, ov::Core& core, const std::string& device,
                               const ov::AnyMap& config);
