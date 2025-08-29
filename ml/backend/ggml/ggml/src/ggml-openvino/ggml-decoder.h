#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <openvino/core/partial_shape.hpp>
#include <vector>

#include "ggml.h"
#include "openvino/decoder.hpp"

class GgmlOvDecoder : public ov::frontend::ggml::GgmlDecoder {
public:
    // Graph decoder
    GgmlOvDecoder(struct ggml_cgraph* cgraph, std::map<std::string, std::shared_ptr<ov::Node>>& model_weights,
                  bool is_static, bool is_first_token);

    // Node decoder, called in GgmlOvDecoder::visit_subgraph
    GgmlOvDecoder(struct ggml_tensor* node, struct ggml_cgraph* cgraph, bool is_static, bool is_first_token,
                  int context_size, int num_heads, int num_heads_kv, int head_size);

    // Naive graph decoder
    GgmlOvDecoder(struct ggml_cgraph* cgraph);

    virtual ov::Any get_attribute(const std::string& name) const override {
        return nullptr;
        GGML_UNUSED(name);
    }

    virtual ov::PartialShape get_input_shape(const std::string& name) const override;

    virtual std::vector<size_t> get_input_stride(const std::string& name) const override;

    virtual ov::element::Type get_input_type(const std::string& name) const override;

    virtual size_t get_input_size() const override;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const override {
        GGML_UNUSED(input_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    virtual std::string& get_input_name(size_t index) const override;

    virtual std::vector<std::string> get_input_names() const override;

    virtual ov::PartialShape get_output_shape(const std::string& name) const override;

    virtual std::vector<size_t> get_output_stride(const std::string& name) const override;

    virtual ov::element::Type get_output_type(const std::string& name) const override;

    virtual int32_t* get_input_op_params(const std::string& name) const override;

    virtual int32_t* get_output_op_params(const std::string& name) const override;

    virtual std::string& get_output_name(size_t index) const override;

    virtual std::vector<std::string> get_output_names() const override;

    virtual const std::string& get_op_type() const override;

    virtual const std::string& get_op_name() const override;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>)> node_visitor) const override;

    const ggml_tensor* get_input_ggml_tensor(const std::string& name) const {
        return m_inputs.at(name);
    }

    const ggml_tensor* get_output_ggml_tensor(const std::string& name) const {
        return m_outputs.at(name);
    }

    virtual int get_op_case() const override {
        return m_op_case;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const override {
        return m_model_inputs;
    }
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const override {
        return m_model_extra_inputs;
    }
    virtual const std::map<std::string, std::shared_ptr<ov::Tensor>>& get_model_extra_input_values() const {
        return m_model_extra_input_values;
    }
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const override {
        return m_model_weights;
    }
    virtual const std::vector<std::string>& get_model_output_names() const override {
        return m_model_output_names;
    }

    virtual int get_context_size() const override { return m_context_size; }

    virtual int get_num_heads() const override { return m_num_heads; }

    virtual int get_num_heads_kv() const override { return m_num_heads_kv; }

    virtual int get_head_size() const override { return m_head_size; }

    virtual int32_t* get_rope_params() const override { return m_rope_params; }

    virtual std::map<std::string, std::string> get_kv_param_res_names() const override;

    virtual bool is_static() const override { return m_is_static; }

    virtual bool is_first_token() const override { return m_is_first_token; }

    ov::PartialShape get_graph_input_shape(const ggml_tensor* src) const;

    static std::shared_ptr<ov::Node> create_weight_node(ggml_tensor* tensor);
    static std::map<std::string, std::shared_ptr<ov::Node>> create_weight_nodes(struct ggml_cgraph* cgraph);

    const ggml_tensor* get_tensor_used_op(const ggml_tensor* tensor) const;
    const ggml_tensor* get_tensor_from_name(const std::string& name) const;

    void clear_model_weights() { m_model_weights.clear(); }

private:
    void set_input_output(ggml_tensor* node, bool naive = false);
    void add_extra_inputs();
    static void dump_cgraph(const struct ggml_cgraph* cgraph, std::string& filename);
    static std::vector<size_t> get_shape(const ggml_tensor* tensor);
    static std::vector<size_t> get_stride(const ggml_tensor* tensor);
    static ov::element::Type get_ov_type(const ggml_tensor* tensor);

    // set context_size, num_heads, etc
    void set_llm_params();

    struct ggml_cgraph* m_cgraph = nullptr;
    ggml_tensor* m_node = nullptr;
    std::vector<ggml_tensor*> m_nodes;
    std::map<std::string, ggml_tensor*> m_inputs;
    std::vector<std::string> m_input_names;
    std::map<std::string, ggml_tensor*> m_outputs;
    std::vector<std::string> m_output_names;
    std::string m_op_name;
    mutable std::string m_name;
    int m_op_case = 0;
    std::vector<std::pair<std::string, std::string>> m_op_node_name;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::Tensor>> m_model_extra_input_values;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_weights;
    std::vector<std::string> m_model_output_names;
    int m_context_size;
    int m_num_heads;
    int m_num_heads_kv;
    int m_head_size;
    int32_t* m_rope_params;
    std::vector<std::string> m_kv_names;
    bool m_is_static;
    bool m_is_first_token;
};

void print_tensor_address_map(const struct ggml_cgraph* cgraph);
