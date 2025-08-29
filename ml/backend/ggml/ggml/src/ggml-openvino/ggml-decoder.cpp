#include "ggml-decoder.h"

#include <ggml-impl.h>
#include <ggml.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <openvino/core/dimension.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/runtime/tensor.hpp>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

GgmlOvDecoder::GgmlOvDecoder(struct ggml_tensor* node, struct ggml_cgraph* cgraph, bool is_static, bool is_first_token,
                             int context_size, int num_heads, int num_heads_kv, int head_size) :
    m_cgraph(cgraph),
    m_node(node),
    m_op_name(std::string(node->name)),
    m_context_size(context_size),
    m_num_heads(num_heads),
    m_num_heads_kv(num_heads_kv),
    m_head_size(head_size),
    m_is_static(is_static),
    m_is_first_token(is_first_token) {
    set_input_output(node);
}

GgmlOvDecoder::GgmlOvDecoder(struct ggml_cgraph* cgraph,
                             std::map<std::string, std::shared_ptr<ov::Node>>& model_weights, bool is_static,
                             bool is_first_token) :
    m_cgraph(cgraph),
    m_op_name(m_node ? std::string(m_node->name) : ""),
    m_model_weights(model_weights),
    m_is_static(is_static),
    m_is_first_token(is_first_token) {
    if (is_first_token && getenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
        print_tensor_address_map(cgraph);
    }

    if (getenv("GGML_OPENVINO_DUMP_CGRAPH")) {
        std::string filename = "cgraph.txt";
        dump_cgraph(cgraph, filename);
    }

    set_llm_params();

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto* cur_node = cgraph->nodes[node_n];
        m_nodes.push_back(cur_node);
        set_input_output(cur_node);
    }

    add_extra_inputs();
}

GgmlOvDecoder::GgmlOvDecoder(struct ggml_cgraph* cgraph) {
    m_cgraph = cgraph;
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto* cur_node = cgraph->nodes[node_n];
        if (cur_node->op == GGML_OP_NONE) {
            continue;
        }
        m_nodes.push_back(cur_node);
        set_input_output(cur_node, true);
    }
}

// Called in GgmlOvDecoder constructor. Two cases: 1. constructing a decoder for the whole graph;
// 2. constructing a decoder for a node;
// 3. constructing a decoder for the whole graph naively (op test case)
void GgmlOvDecoder::set_input_output(ggml_tensor* node, bool naive) {
    std::string node_name;
    if (node->op == GGML_OP_SET_ROWS) {
        // SET_ROWS updates the tensor in place. For later ov op that uses the
        // the view_src of SET_ROWS, we need to make sure they get the updated tensor
        // by putting the view_src name in the tensor_map in
        // <openvino>/src/frontends/ggml/src/translate_session.cpp
        node_name = std::string(node->view_src->name);
    } else {
        node_name = std::string(node->name);
    }

    m_output_names.push_back(node_name);
    m_outputs[node_name] = node;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto* src = node->src[i];
        if (src == nullptr) {
            continue;
        }
        std::string src_name = std::string(src->name);
        m_input_names.push_back(src_name);
        m_inputs[src_name] = src;
        m_op_node_name.emplace_back(src_name, ggml_op_name(node->op));

        // Add model inputs and weights constants, if called for the whole graph
        if (naive) {
            auto param_node = std::make_shared<ov::op::v0::Parameter>(get_ov_type(src), get_graph_input_shape(src));
            param_node->set_friendly_name(src_name);
            param_node->output(0).get_tensor().set_names({src_name});
            m_model_inputs[src_name] = param_node;

        } else if (!m_node && !src->view_src) {
            ggml_backend_buffer* buffer = src->buffer;

            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY || src->flags & GGML_TENSOR_FLAG_INPUT) {
                // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
                if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                    assert(src_name.find("cache_k") == 0 || src_name.find("cache_v") == 0);
                }
                if (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                    continue;
                }
                auto param_node = std::make_shared<ov::op::v0::Parameter>(get_ov_type(src), get_graph_input_shape(src));
                param_node->set_friendly_name(src_name);
                param_node->output(0).get_tensor().set_names({src_name});
                m_model_inputs[src_name] = param_node;
            }
        }
    }

    // Add model outputs, if called for the whole graph
    if (naive) {
        m_model_output_names.push_back(node->name);
    } else if (!m_node) {
        static std::set<std::string> debug_output_names = {};
        // Workaround: the final tensor "result_output" does not have GGML_TENSOR_FLAG_OUTPUT flag set in cgraph
        if (node->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY || node->flags & GGML_TENSOR_FLAG_OUTPUT ||
            std::string(node->name).find("result") == 0 || debug_output_names.count(node->name)) {
            auto name = node->view_src ? std::string(node->view_src->name) : std::string(node->name);
            if (node->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                assert(name.find("cache_k") == 0 || name.find("cache_v") == 0);
            }
            if (auto it = std::find(m_model_output_names.begin(), m_model_output_names.end(), name);
                it == m_model_output_names.end()) {
                m_model_output_names.push_back(name);
            }
            if (auto it = std::find(m_kv_names.begin(), m_kv_names.end(), name); it == m_kv_names.end()) {
                m_kv_names.push_back(name);
            }
        }
    }

    if (m_node) {
        switch (node->op) {
        case GGML_OP_RESHAPE: {
            if (node->ne[0] * node->ne[1] == node->src[0]->ne[0]) {
                m_op_case = 1;
            } else if (node->src[0]->ne[0] * node->src[0]->ne[1] == node->ne[0]) {
                m_op_case = 2;
            } else if (node->src[0]->ne[0] * node->src[0]->ne[1] == node->ne[1]) {
                m_op_case = 3;
            }
            break;
        }
        case GGML_OP_CONT: {
            if (ggml_nelements(node->src[0]) == ggml_nelements(node->src[0]->view_src)) {
                // The input comes from a PERMUTE
                m_op_case = 1;
            } else {
                // The input comes from a VIEW which is subtensor
                m_op_case = 2;
            }
            break;
        }
        case GGML_OP_SET_ROWS: {
            if (std::string(node->name).find("cache_k") == 0) {
                m_op_case = 1;
            } else {
                m_op_case = 2;
            }
            break;
        }
        case GGML_OP_PERMUTE: {
            if (node->src[0]->view_src == nullptr) {
                // Permute Qcur
                m_op_case = 1;
            } else if (ggml_is_contiguous(node->src[0])) {
                // Permute cache_k (view)
                m_op_case = 2;
            } else {
                // Permute cache_v (view)
                m_op_case = 3;
            }
            break;
        }
        case GGML_OP_GET_ROWS: {
            if (node->src[1]->op == GGML_OP_VIEW) {
                m_op_case = 2;
            } else {
                m_op_case = 1;
            }
            break;
        }
        case GGML_OP_ROPE: {
            if (node->src[0]->op == GGML_OP_VIEW) {
                m_op_case = 2;
            } else {
                m_op_case = 1;
            }
            break;
        }
        default:
            break;
        }
    }
}

void GgmlOvDecoder::set_llm_params() {
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        auto* node = m_cgraph->nodes[i];
        if (node->op == GGML_OP_VIEW && std::string(node->name) == "cache_k_l0 (view)") {
            auto* cache_k = node->src[0];
            m_context_size = cache_k->ne[1];
        } else if (node->op == GGML_OP_ROPE && std::string(node->name) == "Qcur-0") {
            m_head_size = node->ne[0];
            m_num_heads = node->ne[1];
            m_rope_params = node->op_params;
        } else if (node->op == GGML_OP_ROPE && std::string(node->name) == "Kcur-0") {
            m_num_heads_kv = node->ne[1];
        }
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_input_shape(const ggml_tensor* src) const {
    auto name = std::string(src->name);
    ov::PartialShape input_shape;
    if (name == "inp_tokens" || name == "inp_pos") {
        if (m_is_static) {
            if (m_is_first_token) {
                input_shape = ov::PartialShape{1, 1, m_context_size};
            } else {
                input_shape = ov::PartialShape{1, 1, 1};
            }
        } else {
            input_shape = ov::PartialShape{1, 1, ov::Dimension(1, m_context_size)};
        }
    } else if (name == "inp_out_ids" && !m_is_static) {
        input_shape = ov::PartialShape{1, 1, ov::Dimension(1, m_context_size)};
    } else if (name == "KQ_mask") {
        if (m_is_static) {
            if (m_is_first_token) {
                input_shape = ov::PartialShape{1, m_context_size, m_context_size};
            } else {
                input_shape = ov::PartialShape{1, 1, m_context_size};
            }
        } else {
            auto max_mask_size = GGML_PAD(m_context_size, GGML_KQ_MASK_PAD);
            input_shape = ov::PartialShape{1, ov::Dimension(1, max_mask_size), ov::Dimension(1, max_mask_size)};
        }
    } else if (name.find("cache_k") == 0) {
        input_shape = ov::PartialShape{m_context_size, m_num_heads_kv, m_head_size};
    } else if (name.find("cache_v") == 0) {
        input_shape = ov::PartialShape{m_num_heads_kv, m_head_size, m_context_size};
    } else if (const auto* op = get_tensor_used_op(src); op && op->op == GGML_OP_SET_ROWS) {
        input_shape = ov::PartialShape{1, 1, -1};
        if (m_is_static) {
            if (m_is_first_token) {
                // Dummy static shape, since the indices are not used in this case
                input_shape = ov::PartialShape{1};
            } else if (std::string(op->name).find("cache_k") == 0) {
                input_shape = ov::PartialShape{1, 1, 1};
            } else {
                input_shape = ov::PartialShape{1, 1, m_num_heads_kv * m_head_size};
            }
        }
    } else if (src->op == GGML_OP_VIEW) {
        // This case is added to make test-backend-ops work
        input_shape = ov::PartialShape{get_shape(src->view_src)};
    } else {
        input_shape = ov::PartialShape{get_shape(src)};
    }
    return input_shape;
}

void GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in matmul's in the attention block. The shape of those matmul's are 32 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    //     Not used for NPU
    int64_t attention_size = -1;
    for (const auto& node : m_nodes) {
        if (node->op == GGML_OP_SOFT_MAX) {
            auto* mask = node->src[1];
            if (std::string(mask->name).find("KQ_mask") != 0) {
                throw std::runtime_error("Unexpected softmax node: " + std::string(mask->name));
            }
            attention_size = mask->ne[0];
            break;
        }
    }

    {
        std::string name = "attention_size";
        auto param_node = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        param_node->set_friendly_name(name);
        param_node->output(0).get_tensor().set_names({name});
        m_model_extra_inputs[name] = param_node;

        auto tensor = std::make_shared<ov::Tensor>(ov::element::i64, ov::Shape{1});
        *tensor->data<int64_t>() = attention_size;
        m_model_extra_input_values[name] = tensor;
    }
}

const ggml_tensor* GgmlOvDecoder::get_tensor_used_op(const ggml_tensor* tensor) const {
    if (tensor == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto* node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] == tensor) {
                return node;
            }
        }
    }
    return nullptr;
}

const ggml_tensor* GgmlOvDecoder::get_tensor_from_name(const std::string& name) const {
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto* node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const auto* src = node->src[j];
            if (src == nullptr) {
                break;
            }
            if (std::string(src->name) == name) {
                return src;
            }
        }
    }
    return nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() const {
    std::map<std::string, std::string> kv_param_res_names;
    for (const auto& name : m_kv_names) {
        if (name.find("cache_k") == 0 || name.find("cache_v") == 0) {
            kv_param_res_names[name] = name;
        }
    }
    return kv_param_res_names;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(struct ggml_cgraph* cgraph) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    static std::mutex weights_mutex;
    auto* nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    std::for_each(std::execution::par, nodes, nodes + n_nodes, [&](ggml_tensor* node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto* src = node->src[i];
            if (src == nullptr) {
                continue;
            }

            std::string src_name(src->name);
            if (!src->view_src) {
                ggml_backend_buffer* buffer = src->buffer;
                if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    bool should_create = false;
                    {
                        std::lock_guard<std::mutex> lock(weights_mutex);
                        if (model_weights.find(src_name) == model_weights.end()) {
                            model_weights[src_name] = nullptr;
                            should_create = true;
                        }
                    }
                    if (should_create) {
                        auto weight_node = create_weight_node(src);
                        weight_node->set_friendly_name(src_name);
                        {
                            std::lock_guard<std::mutex> lock(weights_mutex);
                            model_weights[src_name] = weight_node;
                        }
                    }
                }
            }
        }
    });
    return model_weights;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_node(ggml_tensor* tensor) {
    auto node_type = get_ov_type(tensor);
    auto node_shape = get_shape(tensor);
    auto ne_total = ggml_nelements(tensor);
    ov::Tensor weights(node_type, node_shape);
    memcpy(weights.data(), tensor->data, ne_total * node_type.size());
    return std::make_shared<ov::op::v0::Constant>(weights);
}

void GgmlOvDecoder::dump_cgraph(const struct ggml_cgraph* cgraph, std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    file << "=== GRAPH ===\n";

    // clang-format off
    file << "n_nodes = " << cgraph->n_nodes << "\n";
    file << " " << std::setw(3) << "nodes"
                <<  std::setw(15) << "shape"
                << std::setw(20) << "op"
                << std::setw(20) << "name"
                << std::setw(3) << "    "
                << std::setw(50) << "stride"
                << "\n";
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << ", "
             << std::setw(5) << node->ne[2] << ", "
             << std::setw(5) << node->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(node->op) << std::right << " "
             << std::left << std::setw(45) << node->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << node->nb[0] << ", "
             << std::setw(5) << node->nb[1] << ", "
             << std::setw(5) << node->nb[2] << ", "
             << std::setw(5) << node->nb[3] << "] "
             << "\n";

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (auto* src = node->src[i]) {
                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::left << std::setw(12) << ggml_op_name(src->op) << std::right;
                file << std::left << std::setw(30) << src->name << std::right
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * node = cgraph->leafs[i];

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << "] "
             << std::setw(8) << ggml_op_name(node->op) << " "
             << std::setw(16) << ggml_get_name(node) << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

void print_tensor_address_map(const struct ggml_cgraph* cgraph) {
    std::map<void*, std::vector<std::string>> address_map;
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto* node = cgraph->nodes[node_n];
        if (node->data) {
            auto it = address_map.find(node->data);
            if (it == address_map.end()) {
                address_map[node->data] = std::vector<std::string>();
            }
            address_map[node->data].push_back(node->name);
        }
    }
    for (const auto& pair : address_map) {
        std::cout << "Address: " << pair.first << std::endl;
        for (const auto& name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << std::endl << std::endl;
    }
}

std::vector<size_t> GgmlOvDecoder::get_shape(const ggml_tensor* tensor) {
    std::vector<size_t> shape;
    for (int i = GGML_MAX_DIMS - 2; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    return shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(const ggml_tensor* tensor) {
    std::vector<size_t> stride;
    for (int i = GGML_MAX_DIMS - 2; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    return stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(const ggml_tensor* tensor) {
    switch (tensor->type) {
    case GGML_TYPE_F64:
        return ov::element::f64;
    case GGML_TYPE_F32:
        return ov::element::f32;
    case GGML_TYPE_F16:
        return ov::element::f16;
    case GGML_TYPE_BF16:
        return ov::element::bf16;
    case GGML_TYPE_I8:
        return ov::element::i8;
    case GGML_TYPE_I16:
        return ov::element::i16;
    case GGML_TYPE_I32:
        return ov::element::i32;
    case GGML_TYPE_I64:
        return ov::element::i64;
    default:
        throw std::runtime_error("Unsupported tensor type");
    }
}

ov::PartialShape GgmlOvDecoder::get_input_shape(const std::string& name) const {
    return ov::PartialShape(get_shape(m_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_input_stride(const std::string& name) const {
    return get_stride(m_inputs.at(name));
}

ov::element::Type GgmlOvDecoder::get_input_type(const std::string& name) const {
    return get_ov_type(m_inputs.at(name));
}

size_t GgmlOvDecoder::get_input_size() const {
    return m_input_names.size();
}

std::string& GgmlOvDecoder::get_input_name(size_t index) const {
    m_name = m_input_names[index];
    return m_name;
}

std::vector<std::string> GgmlOvDecoder::get_input_names() const {
    return m_input_names;
}

std::vector<size_t> GgmlOvDecoder::get_output_stride(const std::string& name) const {
    return get_stride(m_outputs.at(name));
}

ov::PartialShape GgmlOvDecoder::get_output_shape(const std::string& name) const {
    return ov::PartialShape(get_shape(m_outputs.at(name)));
}

ov::element::Type GgmlOvDecoder::get_output_type(const std::string& name) const {
    return get_ov_type(m_outputs.at(name));
}

std::string& GgmlOvDecoder::get_output_name(size_t index) const {
    m_name = std::string(m_output_names[index]);
    return m_name;
}

std::vector<std::string> GgmlOvDecoder::get_output_names() const {
    return m_output_names;
}

const std::string& GgmlOvDecoder::get_op_name() const {
    return m_op_name;
}

int32_t* GgmlOvDecoder::get_input_op_params(const std::string& name) const {
    return m_inputs.at(name)->op_params;
}

int32_t* GgmlOvDecoder::get_output_op_params(const std::string& name) const {
    return m_outputs.at(name)->op_params;
}

void GgmlOvDecoder::visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>)> node_visitor) const {
    for (const auto& node : m_nodes) {
        auto decoder = std::make_shared<GgmlOvDecoder>(
            node, m_cgraph, m_is_static, m_is_first_token, m_context_size, m_num_heads, m_num_heads_kv, m_head_size);
        node_visitor(decoder);
    }
}

const std::string& GgmlOvDecoder::get_op_type() const {
    static const std::map<ggml_op, std::string> ops = {
        {GGML_OP_NONE,      "GGML_OP_NONE"     },
        {GGML_OP_ACC,       "GGML_OP_ACC"      },
        {GGML_OP_ADD,       "GGML_OP_ADD"      },
        {GGML_OP_ADD1,      "GGML_OP_ADD1"     },
        {GGML_OP_CONT,      "GGML_OP_CONT"     },
        {GGML_OP_DIV,       "GGML_OP_DIV"      },
        {GGML_OP_DUP,       "GGML_OP_DUP"      },
        {GGML_OP_GET_ROWS,  "GGML_OP_GET_ROWS" },
        {GGML_OP_MUL,       "GGML_OP_MUL"      },
        {GGML_OP_MUL_MAT,   "GGML_OP_MUL_MAT"  },
        {GGML_OP_PERMUTE,   "GGML_OP_PERMUTE"  },
        {GGML_OP_RESHAPE,   "GGML_OP_RESHAPE"  },
        {GGML_OP_RMS_NORM,  "GGML_OP_RMS_NORM" },
        {GGML_OP_ROPE,      "GGML_OP_ROPE"     },
        {GGML_OP_SCALE,     "GGML_OP_SCALE"    },
        {GGML_OP_SOFT_MAX,  "GGML_OP_SOFT_MAX" },
        {GGML_OP_SUB,       "GGML_OP_SUB"      },
        {GGML_OP_TRANSPOSE, "GGML_OP_TRANSPOSE"},
        {GGML_OP_VIEW,      "GGML_OP_VIEW"     },
        {GGML_OP_SET_ROWS,  "GGML_OP_SET_ROWS" },
    };
    static const std::map<ggml_unary_op, std::string> unary_ops = {
        {GGML_UNARY_OP_ABS,         "GGML_UNARY_OP_ABS"        },
        {GGML_UNARY_OP_SGN,         "GGML_UNARY_OP_SGN"        },
        {GGML_UNARY_OP_NEG,         "GGML_UNARY_OP_NEG"        },
        {GGML_UNARY_OP_STEP,        "GGML_UNARY_OP_STEP"       },
        {GGML_UNARY_OP_TANH,        "GGML_UNARY_OP_TANH"       },
        {GGML_UNARY_OP_ELU,         "GGML_UNARY_OP_ELU"        },
        {GGML_UNARY_OP_RELU,        "GGML_UNARY_OP_RELU"       },
        {GGML_UNARY_OP_SIGMOID,     "GGML_UNARY_OP_SIGMOID"    },
        {GGML_UNARY_OP_GELU,        "GGML_UNARY_OP_GELU"       },
        {GGML_UNARY_OP_GELU_QUICK,  "GGML_UNARY_OP_GELU_QUICK" },
        {GGML_UNARY_OP_SILU,        "GGML_UNARY_OP_SILU"       },
        {GGML_UNARY_OP_HARDSWISH,   "GGML_UNARY_OP_HARDSWISH"  },
        {GGML_UNARY_OP_HARDSIGMOID, "GGML_UNARY_OP_HARDSIGMOID"},
        {GGML_UNARY_OP_EXP,         "GGML_UNARY_OP_EXP"        },
        {GGML_UNARY_OP_COUNT,       "GGML_UNARY_OP_COUNT"      }
    };
    static const std::map<ggml_glu_op, std::string> glu_ops = {
        {GGML_GLU_OP_SWIGLU, "GGML_GLU_OP_SWIGLU"},
        {GGML_GLU_OP_GEGLU,  "GGML_GLU_OP_GEGLU" },
        {GGML_GLU_OP_REGLU,  "GGML_GLU_OP_REGLU" }
    };

    switch (m_node->op) {
    case GGML_OP_UNARY:
        return unary_ops.at(ggml_get_unary_op(m_node));
    case GGML_OP_GLU:
        return glu_ops.at(ggml_get_glu_op(m_node));
    default:
        return ops.at(m_node->op);
    }
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return unknown_op;
}
