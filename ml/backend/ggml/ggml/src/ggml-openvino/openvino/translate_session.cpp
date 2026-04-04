#include "translate_session.h"

#include "ggml-openvino/openvino/node_context.h"
#include "ggml-openvino/openvino/utils.h"
#include "input_model.h"
#include "pass/eliminate_zp.h"
#include "pass/mark_decompression_convert_constant_folding.h"
#include "pass/squeeze_matmul.h"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/convert_like.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/make_stateful.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

namespace ov {
namespace frontend {
namespace ggml {

using namespace ov::op;

namespace {

ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    const std::shared_ptr<ov::Model> & model,
    const std::map<std::string, std::string> & kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    const auto & params = model->get_parameters();
    const auto & results = model->get_results();

    for (const auto & param_res : kv_param_res_names) {
        const auto & param_name = param_res.first;
        const auto & res_name = param_res.second;

        auto param_it = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<v0::Parameter> & node) {
            return node->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        auto res_it = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<v0::Result> & node) {
            return node->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::Result> res = *res_it;
        pairs.emplace_back(param, res);
    }
    return pairs;
}

void add_sliced_mask(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {

    auto create_sliced_mask = [&](const std::string & mask_name, const std::string & sliced_name, bool is_static) {
        if ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end())) {
            auto token_len_per_seq = tensor_map.at("token_len_per_seq").get_node_shared_ptr();
            auto mask = tensor_map.at(mask_name).get_node_shared_ptr();
            std::shared_ptr<ov::Node> mask_sliced;
            if (is_static) {
                mask_sliced = mask;
            } else if (ggml_model_decoder.is_stateful()) {
                auto zero_2d = ov::op::v0::Constant::create(ov::element::i64, {2}, {0,0});
                auto one_2d = ov::op::v0::Constant::create(ov::element::i64, {2}, {1,1});
                auto zero_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto three_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
                auto neg_one_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
                auto axes = ov::op::v0::Constant::create(ov::element::i64, {2}, {-2,-1});
                auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
                auto gather_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one_1d, three_1d);
                auto reshaped_inp_pos = std::make_shared<ov::op::v1::Reshape>(gather_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), false);
                auto inp_pos_incremented = std::make_shared<ov::op::v1::Add>(reshaped_inp_pos, ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}));
                auto stop = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{token_len_per_seq, std::make_shared<v1::ConvertLike>(inp_pos_incremented, token_len_per_seq)}, 0);
                mask_sliced =
                    std::make_shared<ov::op::v8::Slice>(mask, zero_2d, stop, one_2d, axes);
                mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
                mask_sliced->set_friendly_name(sliced_name);
            } else {
                auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
                mask_sliced = std::make_shared<ov::op::v8::Slice>(mask, zero, token_len_per_seq, one, two);
                mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
                mask_sliced->set_friendly_name(sliced_name);
            }
            tensor_map.insert({sliced_name, mask_sliced->output(0)});
        }
    };

    create_sliced_mask("self_kq_mask", "KQ_mask_sliced", ggml_model_decoder.is_static());
    create_sliced_mask("self_kq_mask_swa", "KQ_mask_swa_sliced", ggml_model_decoder.is_static());
}

void add_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    if (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == nullptr) {
        return;
    }
    auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    if (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_node_shared_ptr();
    }

    auto sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_weight);
    auto sin_theta = sin_cos.first;
    auto cos_theta = sin_cos.second;

    cos_theta.get_node_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_node_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
void preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    add_sliced_mask(tensor_map, ggml_model_decoder);
    add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(const frontend::InputModel::Ptr & input_model,
                                   const std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   bool naive) :
    m_input_model(input_model),
    m_translator_map(translator_map),
    m_ov_model(nullptr),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    return m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(const frontend::InputModel::Ptr & input_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    const auto & ggml_model = std::dynamic_pointer_cast<InputModel>(input_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    for (const auto & it : ggml_model_decoder->get_model_inputs()) {
        params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second));
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto & it : ggml_model_decoder->get_model_extra_inputs()) {
        if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second)) {
            params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second));
        }
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    auto node_visitor = [&](std::shared_ptr<GgmlDecoder> decoder, int node_idx) {
        auto operation_type = decoder->get_op_type(node_idx);
        if (operation_type == "GGML_OP_NONE") {
            return;
        }

        ov::OutputVector converted_outputs;
        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        NodeContext node_context(decoder, tensor_map, node_idx, this);
        converted_outputs = it->second(node_context);

        const auto & node_output_names = decoder->get_output_names(node_idx);
        FRONT_END_OP_CONVERSION_CHECK(node_output_names.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      node_output_names.size(), " and ", converted_outputs.size(), " respectively.");

        for (size_t i = 0; i < node_output_names.size(); ++i) {
            auto output_name = node_output_names[i];
            if (i < converted_outputs.size() && converted_outputs[i].get_node_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
    };

    if (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->visit_subgraph(node_visitor);

    for (const auto & name : ggml_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        auto result = std::make_shared<v0::Result>(tensor_map->at(name));
        result->set_friendly_name(name);
        results.push_back(result);
    }

    ov::ParameterVector used_params;
    for (const auto & param : params) {
        if (!param->output(0).get_target_inputs().empty()) {
            used_params.push_back(param);
        }
    }
    // if (auto diff = params.size() - used_params.size()) {
    //     GGML_LOG_INFO("%zu parameters are not used in the model.", diff);
    // }
    resulting_model = std::make_shared<Model>(results, used_params);

    apply_transformations(resulting_model);
    return resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    auto ggml_model_decoder = std::dynamic_pointer_cast<InputModel>(m_input_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(true);
        manager.register_pass<ov::pass::MarkCompressedFloatConstants>();

        if (ggml_model_decoder->is_stateful()) {
            const auto kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
            const auto kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
            manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
        }

        if (ggml_model_decoder->is_static()) {
            manager.register_pass<pass::EliminateZeroPoints>();
            manager.register_pass<pass::SqueezeMatmul>();
        }
        manager.run_passes(model);
        if (ggml_model_decoder->is_stateful()) {
            auto output_names = ggml_model_decoder->get_model_output_names();
            std::map<std::string, int> model_output_indexes;
            for (size_t i=0; i<output_names.size(); i++) {
                model_output_indexes.insert(std::make_pair(output_names[i], i));
            }
            ov::preprocess::PrePostProcessor ppp(model);
            for (size_t i=0; i<model->get_output_size(); i++) {
                auto output_friendly_name = model->output(i).get_node_shared_ptr()->get_friendly_name();
                auto output_id = model_output_indexes[output_friendly_name];
                auto model_output_shape = model->output(i).get_partial_shape();
                auto decoder_output_shape = ggml_model_decoder->get_output_shape(output_id);
                if (model_output_shape.rank().is_static() && decoder_output_shape.rank().is_static()
                    && model_output_shape.rank().get_length() + 1 == decoder_output_shape.rank().get_length()
                    && decoder_output_shape[0].is_static() && decoder_output_shape[0].get_length() == 1) {
                    ppp.output(i).postprocess().custom([](const ov::Output<ov::Node>& node) {
                        auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        return std::make_shared<ov::op::v0::Unsqueeze>(node, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    return model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
