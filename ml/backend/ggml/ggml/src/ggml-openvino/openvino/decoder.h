#pragma once

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
