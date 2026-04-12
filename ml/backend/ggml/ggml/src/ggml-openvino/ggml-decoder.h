#pragma once

#include "ggml-quants.h"
#include "ggml.h"
#include "openvino/decoder.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <openvino/core/partial_shape.hpp>
#include <optional>
#include <vector>

struct ModelParams {
    int ctx = -1;
    int ctx_swa = -1;
    int ctx_per_seq = -1;
    int ctx_per_seq_swa = -1;
    int n_seq = 1;
    int n_heads = -1;
    int n_heads_kv = -1;
    int head_size = -1;
    int32_t rope_params[15];
    std::vector<int> swa_layers;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(const ModelParams & other) const {
        return memcmp(rope_params, other.rope_params, sizeof(int32_t) * 15) == 0;
    }

    bool can_reuse_dynamically(const ModelParams & other) const { return same_rope_params(other); }

    bool can_reuse_statically(const ModelParams & other) const { return same_rope_params(other) && ctx == other.ctx; }

    bool kv_buffer_changed(const ModelParams & other) const { return kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

struct ComputeParams {
    int n_seq_active = 1;
    int seq_active_start = 0;
    int attention_size = -1;
    int attention_size_swa = -1;
    int input_len = -1;
    int token_len_per_seq = -1;
    int past_kv_len = -1;
    int output_len = 1;
};

class GgmlOvDecoder : public ov::frontend::ggml::GgmlDecoder {
public:
    struct NodeInfo {
        ggml_tensor * node;
        std::string node_name;
        std::string node_op_type;
        std::map<std::string, ggml_tensor *> node_inputs;
        std::vector<std::string> node_inputs_names;
        ggml_tensor * node_output;
        std::string node_output_name;
        int node_op_case = 0;
        void * data_addr;
    };
    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool is_prefill = false,
                  int prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights);

    virtual ov::Any get_attribute(const std::string & name) const override {
        return nullptr;
        GGML_UNUSED(name);
    }

    virtual ov::PartialShape get_input_shape(int node_idx, const std::string & name) const override;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string & name) const override;

    virtual ov::element::Type get_input_type(int node_idx, const std::string & name) const override;

    virtual size_t get_input_size() const override;

    virtual size_t get_input_size(int node_idx) const override;

    virtual void get_input_node(size_t input_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) const override {
        GGML_UNUSED(input_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    virtual std::vector<std::string> get_input_names(int node_idx) const override;

    virtual ov::PartialShape get_output_shape(int node_idx) const override;

    virtual ov::element::Type get_output_type(int node_idx) const override;

    virtual int32_t * get_input_op_params(int node_idx, const std::string & name) const override;

    virtual int32_t * get_output_op_params(int node_idx) const override;

    virtual std::vector<std::string> get_output_names(int node_idx) const override;

    virtual const std::string & get_op_type() const override;

    virtual const std::string & get_op_type(int node_idx) const override;

    virtual const std::string & get_op_name() const override;

    virtual const std::string & get_op_name(int node_idx) const override;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const override;

    ggml_tensor * get_input_ggml_tensor(const std::string & name) const { return m_inputs.at(name); }

    virtual int get_op_case(int node_idx) const override { return m_node_info_list[node_idx].node_op_case; }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_inputs() const override {
        return m_model_inputs;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_extra_inputs() const override {
        return m_model_extra_inputs;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Tensor>> & get_model_extra_input_values() const {
        return m_model_extra_input_values;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_weights() const override {
        return m_model_weights;
    }

    virtual std::vector<std::string> get_model_output_names() const override {
        return m_model_output_names;
    }

    const std::map<std::string, ggml_tensor *> & get_model_outputs() const { return m_model_outputs; }

    virtual int get_ctx_size() const { return m_model_params.ctx; }

    virtual int get_ctx_swa_size() const { return m_model_params.ctx_swa; }

    virtual int get_ctx_per_seq() const { return m_model_params.ctx_per_seq; }

    virtual int get_ctx_per_seq_swa() const { return m_model_params.ctx_per_seq_swa; }

    virtual int get_n_seq() const { return m_model_params.n_seq; }

    virtual int is_swa_layer(int layer) const override {
        return std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    int get_past_kv_len() const { return m_compute_params.past_kv_len; }

    int get_input_len() const { return m_compute_params.input_len; }

    virtual int32_t * get_rope_params() const override { return const_cast<int32_t *>(m_model_params.rope_params); }

    virtual std::map<std::string, std::string> get_kv_param_res_names() const override;

    virtual bool is_static() const override { return m_is_static; }

    virtual bool is_stateful() const override { return m_is_stateful; }

    ov::PartialShape get_graph_input_shape(const ggml_tensor * op, const ggml_tensor * input) const;

    static void dump_cgraph(const ggml_cgraph * cgraph, std::string & filename);

    static std::shared_ptr<ov::Node> create_weight_node(ggml_tensor * tensor, bool naive = false);

    static std::map<std::string, std::shared_ptr<ov::Node>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    const ggml_tensor * get_tensor_used_op(const ggml_tensor * tensor) const;

    const ggml_tensor * get_tensor_from_name(const std::string & name) const;

    void clear_model_weights() { m_model_weights.clear(); }

    static std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() const { return m_model_params; }

    ComputeParams get_compute_params() const { return m_compute_params; }

    void set_model_params(const ModelParams & model_params) { m_model_params = model_params; }

    void set_compute_params(const ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    int m_prefill_chunk_size = 0;

    static ov::Shape get_shape(const ggml_tensor * tensor);
    static std::vector<size_t> get_stride(const ggml_tensor * tensor);
    static ov::element::Type get_ov_type(const ggml_tensor * tensor);
    static std::string compute_op_type(const ggml_tensor * node);
    void add_extra_inputs();

    void update_io(ggml_cgraph * cgraph);

    inline static bool is_inp_tok(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    inline static bool is_inp_pos(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    inline static bool is_inp_emb(const ggml_tensor * tensor, const ggml_tensor * op) {
        return tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    inline static bool is_inp_mask(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]);
    }

    inline static bool is_rope_freqs_weight(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    inline static bool is_kvcache(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_SET_ROWS && op->src[2] == tensor;
    }

    inline static bool is_kv_idx(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    inline static bool is_output_idx(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE;
    }

    static std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) {
        if (is_inp_tok(tensor, op)) {
            return "inp_tokens";
        }
        if (is_inp_pos(tensor, op)) {
            return "inp_pos";
        }
        if (is_inp_emb(tensor, op)) {
            return "embd";
        }
        if (is_output_idx(tensor, op)) {
            return "inp_out_ids";
        }
        if (is_inp_mask(tensor, op)) {
            return std::string(tensor->name).find("swa") == std::string::npos ? "self_kq_mask" : "self_kq_mask_swa";
        }
        return tensor->name;
    }

private:
    void set_input_output();
    int compute_op_case(const ggml_tensor * node) const;
    bool node_is_used_as_src(const int node_idx);
    void compute_model_inputs();
    void compute_model_outputs();

    void validate_cgraph() const;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::Tensor>> m_model_extra_input_values;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::vector<std::string> m_model_output_names;
    std::vector<NodeInfo> m_node_info_list;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

void print_tensor_address_map(const ggml_cgraph * cgraph);

int extract_layer_from_name(const std::string & name);
