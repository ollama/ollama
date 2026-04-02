#include "ggml-decoder.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-openvino-extra.h"
#include "ggml-openvino.h"
#include "ggml-quants.h"

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
#include <mutex>
#include <openvino/core/dimension.hpp>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             bool is_static,
                             bool is_stateful,
                             bool is_prefill,
                             int prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(false),
    m_prefill_chunk_size(prefill_chunk_size),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    if (auto * env = getenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS"); env && std::string(env) != "0") {
#ifdef _WIN32
        _putenv_s("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS", "");
#else
        unsetenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS");
#endif
        print_tensor_address_map(cgraph);
    }

    validate_cgraph();

    set_input_output();
    compute_model_inputs();
    compute_model_outputs();

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }

    add_extra_inputs();
}

void GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_node_info_list.clear();
    set_input_output();
    compute_model_inputs();
    compute_model_outputs();
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = true;
    set_input_output();
    compute_model_inputs();
    compute_model_outputs();
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }
}

void GgmlOvDecoder::set_input_output() {
    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto node = m_cgraph->nodes[node_n];

        NodeInfo current_node_info;
        auto node_name = std::string(node->name);
        auto node_output_name = node_name;
        auto * node_output = node;
        if (node->op == GGML_OP_SET_ROWS) {
            // SET_ROWS updates the tensor in place. For later ov op that uses the
            // the view_src of SET_ROWS, we need to make sure they get the updated tensor
            // by putting the view_src name in the tensor_map in
            // <openvino>/src/frontends/ggml/src/translate_session.cpp
            node_output_name = std::string(node->view_src->name);
            node_output = node->view_src;
        }

        current_node_info.node = node;
        current_node_info.node_name = node_name;
        current_node_info.node_output = node_output;
        current_node_info.node_output_name = node_output_name;
        current_node_info.node_op_case = 0;
        current_node_info.data_addr = node->data;

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            auto src_name = std::string(src->name);
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_graph_input_ov_name(src, node);
            }
            current_node_info.node_inputs[src_name] = src;
            current_node_info.node_inputs_names.push_back(src_name);
        }

        m_node_info_list.push_back(current_node_info);
    }
}

int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {
    int op_case = 0;
    switch (node->op) {
    case GGML_OP_RESHAPE: {
        auto * src = node->src[0];
        if (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == node->ne[0] && src->src[0]->ne[1] == node->ne[1]) {
            op_case = 4;
        } else if (node->ne[0] * node->ne[1] == src->ne[0]) {
            op_case = 1;
        } else if (src->ne[0] * src->ne[1] == node->ne[0]) {
            op_case = 2;
            if (src->ne[2] * src->ne[3] == node->ne[1]) {
                op_case = 5;
            }
        } else if (src->ne[0] * src->ne[1] == node->ne[1]) {
            op_case = 3;
        } else if (src->ne[1] * src->ne[2] == node->ne[1]) {
            op_case = 6;
        }
        break;
    }
    case GGML_OP_CONT: {
        if (node->src[0]->op == GGML_OP_PERMUTE) {
            op_case = 1;
        } else if (node->src[0]->op == GGML_OP_TRANSPOSE) {
            op_case = 2;
        } else if (node->src[0]->op == GGML_OP_VIEW) {
            op_case = 3;
        }
        break;
    }
    case GGML_OP_PERMUTE: {
        if (node->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } else if (node->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(node->view_src->name);
            int layer = extract_layer_from_name(src_name);
            if (!is_swa_layer(layer)) {
                op_case = 2;
            } else {
                op_case = 3;
            }
        } else {
            // rope'ed query tensor
            op_case = 4;
        }
        break;
    }
    case GGML_OP_MUL_MAT: {
        if (node->src[0]->op == GGML_OP_CONT && node->src[0]->src[0]->op == GGML_OP_TRANSPOSE) {
            op_case = 2;
        } else if (node->src[0]->op == GGML_OP_VIEW && node->src[1]->op == GGML_OP_VIEW) {
            op_case = 3;
        }
        break;
    }
    case GGML_OP_GET_ROWS: {
        if (node->src[1]->op == GGML_OP_VIEW) {
            op_case = 2;
        }
        break;
    }
    case GGML_OP_ROPE: {
        if (node->src[0]->op == GGML_OP_VIEW) {
            op_case = 2;
        }
        break;
    }
    case GGML_OP_VIEW: {
        if (node->src[0]->op == GGML_OP_VIEW) {
            auto * src = node->src[0];
            if (ggml_nelements(node) != ggml_nelements(src)) {
                throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 2;
        }
        {
            auto * src = node->src[0];
            if ((ggml_nelements(node) != ggml_nelements(src)) && m_naive) {
                // Compare each dimension of node and src, if only one dimension differs then op_case=3
                int diff_count = 0;
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->ne[i] != src->ne[i]) {
                        diff_count++;
                    }
                }
                if (diff_count == 1) {
                    op_case = 3;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    return op_case;
}

int extract_layer_from_name(const std::string & name) {
    size_t pos1 = name.find("_l");
    assert(pos1 != std::string::npos);
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    if (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    int layer = std::stoi(layer_str);
    return layer;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, bool is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto * node = cgraph->nodes[i];
        std::string name = std::string(node->name);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            model_params.n_heads = node->src[0]->ne[2];
            model_params.n_heads_kv = node->src[1]->ne[2];
            model_params.head_size = node->src[0]->ne[0];
            compute_params.input_len = node->src[0]->ne[1];

            auto * cache_k_perm = node->src[1];
            if (cache_k_perm->op == GGML_OP_CPY) {
                cache_k_perm = cache_k_perm->src[0];
            }
            assert(cache_k_perm->op == GGML_OP_PERMUTE);
            auto * cache_k_view = cache_k_perm->src[0];
            assert(cache_k_view->op == GGML_OP_VIEW);

            auto * cache_k = cache_k_view->src[0];
            int layer = extract_layer_from_name(cache_k->name);
            auto * mask = node->src[3];
            std::string mask_name(mask->name);

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            if (mask_name.find("swa") != std::string::npos) {
                model_params.swa_layers.push_back(layer);
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } else {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, sizeof(size_t));
            compute_params.seq_active_start = offset / seq_size;
            compute_params.token_len_per_seq = node->ne[2];

            if (mask_name.find("swa") != std::string::npos) {
                compute_params.attention_size_swa = mask->ne[0];
            } else {
                compute_params.attention_size = mask->ne[0];
            }
            if (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
            break;
        }
        if (node->op == GGML_OP_ROPE) {
            memcpy(model_params.rope_params, node->op_params, sizeof(int32_t) * 15);
        }
    }
    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    // for NPU, output_len is always 1 except for llama-perplexity
    if (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    model_params.ctx_swa = model_params.ctx_per_seq_swa * model_params.n_seq;
    return {model_params, compute_params};
}

void GgmlOvDecoder::validate_cgraph() const {
    if (m_model_params.n_seq > 1 && m_is_static == true) {
        throw std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_input_shape(const ggml_tensor * op, const ggml_tensor * input) const {
    if (m_naive) {
        return input!= nullptr ? ov::PartialShape{get_shape(input)} : ov::PartialShape{get_shape(op)};
    }
    auto name = std::string(input->name);
    ov::PartialShape input_shape;

    if (is_inp_tok(input, op) || is_inp_pos(input, op)) {
        // tokens or positions
        int len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else if (is_output_idx(input, op)) {
        // output index
        input_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } else if (is_inp_mask(input, op)) {
        // mask
        if (m_is_static) {
            input_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } else if (m_is_stateful) {
            input_shape = ov::PartialShape{1, 1, -1, -1};
        } else {
            input_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } else if (is_kvcache(input, op)) {
        // kvcache
        input_shape = ov::PartialShape{get_shape(input)};
        if (!m_is_static) {
            // do not fix ctx size to make llama-bench work across test params
            input_shape[2] = -1;
        }
        if (is_stateful()) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            assert(input_shape.size() == 4 && input_shape[0] == 1 && input_shape[1] == 1 &&
                   input_shape[2].is_dynamic() &&
                   input_shape[3] == (m_model_params.n_heads_kv * m_model_params.head_size));
            input_shape = {input_shape[0], ov::Dimension::dynamic(), m_model_params.n_heads_kv,
                           m_model_params.head_size};
        }

    } else if (is_kv_idx(input, op)) {
        // kv update index
        int len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else {
        input_shape = ov::PartialShape{get_shape(input)};
    }
    return input_shape;
}

void GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_input = [this](const std::string & name, int64_t value) {
        if (m_is_static) {
            auto constant =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{value});
            constant->set_friendly_name(name);
            m_model_extra_inputs[name] = constant;
        } else {
            auto param_node = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
            param_node->set_friendly_name(name);
            param_node->output(0).get_tensor().set_names({name});
            m_model_extra_inputs[name] = param_node;

            auto tensor = std::make_shared<ov::Tensor>(ov::element::i64, ov::Shape{1});
            *tensor->data<int64_t>() = value;
            m_model_extra_input_values[name] = tensor;
        }
    };

    create_1d_input("attention_size", m_compute_params.attention_size);
    if (m_compute_params.attention_size_swa != -1) {
        create_1d_input("attention_size_swa", m_compute_params.attention_size_swa);
    }
    create_1d_input("n_seq_active", m_compute_params.n_seq_active);
    create_1d_input("seq_active_start", m_compute_params.seq_active_start);
    create_1d_input("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    create_1d_input("token_len_per_seq", m_compute_params.token_len_per_seq);
    // create_1d_input("token_len", m_token_len_per_seq * m_n_seq_active);
}

bool GgmlOvDecoder::node_is_used_as_src(const int node_idx) {
    ggml_tensor * node = m_cgraph->nodes[node_idx];
    for (int i = node_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (other_node->src[j] == node) {
                return true;
            }
        }
    }
    return false;
}

void GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * node = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        if (node->op == GGML_OP_NONE && node_is_used_as_src(i)) {
            std::string node_name(node->name);
            if (m_model_weights.find(node_name) == m_model_weights.end()) {
                m_inputs[node_name] = node;
                auto param_node =
                    std::make_shared<ov::op::v0::Parameter>(get_ov_type(node), get_graph_input_shape(node, nullptr));
                param_node->set_friendly_name(node_name);
                param_node->output(0).get_tensor().set_names({node_name});
                m_model_inputs[node_name] = param_node;
            }
            continue;
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            std::string src_name = std::string(src->name);
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_graph_input_ov_name(src, node);
            }
            if (m_model_weights.find(src_name) != m_model_weights.end()) {
                continue;
            }

            bool is_intermediate_node = false;
            for (const auto & node_info : m_node_info_list) {
                if (node_info.node == src) {
                    is_intermediate_node = true;
                    break;
                }
            }
            if (is_intermediate_node) {
                continue;
            }
            if (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                continue;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                if (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            ov::PartialShape param_shape = get_graph_input_shape(node, src);
            auto param_node = std::make_shared<ov::op::v0::Parameter>(get_ov_type(src), param_shape);
            param_node->set_friendly_name(src_name);
            param_node->output(0).get_tensor().set_names({src_name});
            m_model_inputs[src_name] = param_node;
        }
    }
}

void GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto * cur_node = m_cgraph->nodes[node_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        if (cur_node->op == GGML_OP_NONE) {
            continue;
        }
        auto cur_node_use_count = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_node)];
        if (cur_node_use_count == 0) {
            // The output of SET_ROWS is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            if (cur_node != nullptr && cur_node->op == GGML_OP_SET_ROWS) {
                cur_node = cur_node->view_src;
            }
        } else {
            int input_use_count = 0;
            for (int i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * node = m_cgraph->nodes[i];
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != NULL && node->src[j] == cur_node) {
                        input_use_count++;
                    }
                }
            }
            if (input_use_count == cur_node_use_count) {
                cur_node = nullptr;
            }
        }
        if (cur_node != nullptr) {
            std::string node_output_name(cur_node->name);
            m_model_outputs[node_output_name] = cur_node;
            m_model_output_names.push_back(node_output_name);
        }
    }
}

const ggml_tensor * GgmlOvDecoder::get_tensor_used_op(const ggml_tensor * tensor) const {
    if (tensor == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] == tensor) {
                return node;
            }
        }
    }
    return nullptr;
}

const ggml_tensor * GgmlOvDecoder::get_tensor_from_name(const std::string & name) const {
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const auto * src = node->src[j];
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
    for (const auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    return kv_param_res_names;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, bool naive) {
    static std::mutex weights_mutex;
    std::lock_guard<std::mutex> lock(weights_mutex);

    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    for (int node_i = 0; node_i < n_nodes; node_i++) {
        auto * node = nodes[node_i];
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }

            std::string src_name(src->name);
            if (is_rope_freqs_weight(src, node)) {
                src_name = "rope_freqs.weight";
            }
            if (!src->view_src) {
                ggml_backend_buffer * buffer = src->buffer;
                if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type)) {
                    if (model_weights.find(src_name) == model_weights.end()) {
                        auto weight_node = create_weight_node(src, naive);
                        weight_node->set_friendly_name(src_name);
                        model_weights[src_name] = weight_node;
                    }
                }
            }
        }
    }
    return model_weights;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_node(ggml_tensor * tensor, bool naive) {
    const bool is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    if (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        if (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            if (weight_extra->weight_node) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                return weight_extra->weight_node;
            }
        } else if (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            if (quant_extra->weight_node) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                return quant_extra->weight_node;
            }
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    static const std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16,
                                                     GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                                                     GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
    if (weight_types.find(tensor->type) == weight_types.end()) {
        throw std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    if (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        if (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } else {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } else {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_node->set_friendly_name(tensor->name);
    if (!is_ov_buffer) {
        return ov_weight.weight_node;
    }

    ggml_openvino_extra_base * extra;
    if (ov_weight.is_quantized()) {
        extra = new ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_node);
    } else {
        extra = new ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_node);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    return ov_weight.weight_node;
}

void GgmlOvDecoder::dump_cgraph(const ggml_cgraph * cgraph, std::string & filename) {
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
                << std::setw(62) << "stride"
                << std::setw(20) << "buffer_type"
                << "\n";
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // Get buffer type name
        const char * buf_name = "none";
        ggml_backend_buffer_t buf = node->view_src ? node->view_src->buffer : node->buffer;
        if (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

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
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (auto* src = node->src[i]) {
                // Get buffer type name for source
                const char * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                if (src_buf) {
                    src_buf_name = ggml_backend_buffer_name(src_buf);
                }

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
                << std::right << std::setw(15) << src_buf_name << std::right
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    for (int i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * node = cgraph->leafs[i];

        // Get buffer type name for leaf
        const char * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = node->view_src ? node->view_src->buffer : node->buffer;
        if (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << "] "
             << std::setw(8) << ggml_op_name(node->op) << " "
             << std::setw(16) << ggml_get_name(node)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

void print_tensor_address_map(const ggml_cgraph * cgraph) {
    std::map<void *, std::vector<std::string>> address_map;
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto * node = cgraph->nodes[node_n];
        if (node->data) {
            auto it = address_map.find(node->data);
            if (it == address_map.end()) {
                address_map[node->data] = std::vector<std::string>();
            }
            address_map[node->data].push_back(node->name);
        }
    }
    for (const auto & pair : address_map) {
        std::cout << "Address: " << pair.first << std::endl;
        for (const auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << std::endl << std::endl;
    }
}

ov::Shape GgmlOvDecoder::get_shape(const ggml_tensor * tensor) {
    std::vector<size_t> shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    return shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(const ggml_tensor * tensor) {
    std::vector<size_t> stride;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    return stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(const ggml_tensor * tensor) {
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
        return ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_input_shape(int node_idx, const std::string & name) const {
    return ov::PartialShape(get_shape(m_node_info_list[node_idx].node_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_input_stride(int node_idx, const std::string & name) const {
    return get_stride(m_node_info_list[node_idx].node_inputs.at(name));
}

ov::element::Type GgmlOvDecoder::get_input_type(int node_idx, const std::string & name) const {
    return get_ov_type(m_node_info_list[node_idx].node_inputs.at(name));
}

size_t GgmlOvDecoder::get_input_size() const {
    return m_model_inputs.size();
}

size_t GgmlOvDecoder::get_input_size(int node_idx) const {
    return m_node_info_list[node_idx].node_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_input_names(int node_idx) const {
    return m_node_info_list[node_idx].node_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(int node_idx) const {
    auto * ggml_tensor = m_node_info_list[node_idx].node_output;
    return ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(const int node_idx) const {
    return get_ov_type(m_node_info_list[node_idx].node);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(int node_idx) const {
    return {m_node_info_list[node_idx].node_output_name};
}

const std::string & GgmlOvDecoder::get_op_name() const {
    static const std::string unknown_name = "UNKNOWN_OP_NAME";
    return unknown_name;
}

const std::string & GgmlOvDecoder::get_op_name(int node_idx) const {
    return m_node_info_list[node_idx].node_name;
}

int32_t * GgmlOvDecoder::get_input_op_params(int node_idx, const std::string & name) const {
    return m_node_info_list[node_idx].node_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(int node_idx) const {
    return m_node_info_list[node_idx].node->op_params;
}

void GgmlOvDecoder::visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const {
    for (int node_idx = 0; node_idx < m_cgraph->n_nodes; node_idx++) {
        if (m_cgraph->nodes[node_idx]->op == GGML_OP_NONE) {
            continue;
        }
        node_visitor(std::make_shared<GgmlOvDecoder>(*this), node_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(const ggml_tensor * node) {
    static const std::map<ggml_op, std::string> ops = {
        {GGML_OP_NONE,           "GGML_OP_NONE"          },
        {GGML_OP_ACC,            "GGML_OP_ACC"           },
        {GGML_OP_ADD,            "GGML_OP_ADD"           },
        {GGML_OP_ADD1,           "GGML_OP_ADD1"          },
        {GGML_OP_CONT,           "GGML_OP_CONT"          },
        {GGML_OP_DIV,            "GGML_OP_DIV"           },
        {GGML_OP_DUP,            "GGML_OP_DUP"           },
        {GGML_OP_GET_ROWS,       "GGML_OP_GET_ROWS"      },
        {GGML_OP_MUL,            "GGML_OP_MUL"           },
        {GGML_OP_MUL_MAT,        "GGML_OP_MUL_MAT"       },
        {GGML_OP_PERMUTE,        "GGML_OP_PERMUTE"       },
        {GGML_OP_RESHAPE,        "GGML_OP_RESHAPE"       },
        {GGML_OP_RMS_NORM,       "GGML_OP_RMS_NORM"      },
        {GGML_OP_ROPE,           "GGML_OP_ROPE"          },
        {GGML_OP_SCALE,          "GGML_OP_SCALE"         },
        {GGML_OP_SOFT_MAX,       "GGML_OP_SOFT_MAX"      },
        {GGML_OP_SUB,            "GGML_OP_SUB"           },
        {GGML_OP_TRANSPOSE,      "GGML_OP_TRANSPOSE"     },
        {GGML_OP_VIEW,           "GGML_OP_VIEW"          },
        {GGML_OP_SET_ROWS,       "GGML_OP_SET_ROWS"      },
        {GGML_OP_CPY,            "GGML_OP_CPY"           },
        {GGML_OP_FLASH_ATTN_EXT, "GGML_OP_FLASH_ATTN_EXT"},
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

    switch (node->op) {
    case GGML_OP_UNARY:
        return unary_ops.at(ggml_get_unary_op(node));
    case GGML_OP_GLU:
        return glu_ops.at(ggml_get_glu_op(node));
    default:
        return ops.at(node->op);
    }
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return unknown_op;
}

const std::string & GgmlOvDecoder::get_op_type(int node_idx) const {
    return m_node_info_list[node_idx].node_op_type;
}

const std::string & GgmlOvDecoder::get_op_type() const {
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return unknown_op;
}
