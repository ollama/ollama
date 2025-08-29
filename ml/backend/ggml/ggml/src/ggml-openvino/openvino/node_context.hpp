#pragma once

#include <cstdint>
#include <openvino/frontend/node_context.hpp>

#include "decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type()),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names();
        m_output_names = decoder->get_output_names();
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    size_t get_input_size() const override {
        return m_decoder->get_input_size();
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_input_names[index]);
    }

    PartialShape get_input_shape(size_t index) const {
        return m_decoder->get_input_shape(m_input_names[index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape(size_t index) const {
        return m_decoder->get_output_shape(m_output_names[index]);
    }

    std::vector<size_t> get_output_stride(size_t index) const {
        return m_decoder->get_output_stride(m_output_names[index]);
    }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_input_names[index]);
    }

    int32_t* get_output_op_params(size_t index) const {
        return m_decoder->get_output_op_params(m_output_names[index]);
    }

    ov::element::Type get_output_type(size_t index) const {
        return m_decoder->get_output_type(m_output_names[index]);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_decoder->get_input_name(idx));
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name();
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case();
    }
    bool is_static() const {
        return m_decoder->is_static();
    }
    bool is_first_token() const {
        return m_decoder->is_first_token();
    }

    int get_num_heads() const { return m_decoder->get_num_heads(); }

    int get_num_heads_kv() const { return m_decoder->get_num_heads_kv(); }

    int get_head_size() const { return m_decoder->get_head_size(); }

    int get_context_size() const { return m_decoder->get_context_size(); }

  private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
