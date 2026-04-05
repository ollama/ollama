#include "chat.h"
#include "chat-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "json-partial.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "regex-partial.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <algorithm>
#include <cstdio>
#include <cctype>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto time = std::chrono::system_clock::to_time_t(now);
    auto local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    auto res = ss.str();
    return res;
}

static std::string string_diff(const std::string & last, const std::string & current) {
    if (last.empty()) {
        return current;
    }
    if (!string_starts_with(current, last)) {
        if (string_starts_with(last, current)) {
            // This happens if the last generation ended on a partial stop word (not erased),
            // and the current ended on a stop word (erased).
            return "";
        }
        throw std::runtime_error("Invalid diff: '" + last + "' not found at start of '" + current + "'");
    }
    return current.substr(last.size());
}

static bool has_content_or_tool_calls(const common_chat_msg & msg) {
    return !msg.content.empty() || !msg.tool_calls.empty();
}

template <>
json common_chat_msg::to_json_oaicompat() const
{
    json message {
        {"role", "assistant"},
    };
    if (!reasoning_content.empty()) {
        message["reasoning_content"] = reasoning_content;
    }
    if (content.empty() && !tool_calls.empty()) {
        message["content"] = json();
    } else {
        message["content"] = content;
    }
    if (!tool_calls.empty()) {
        auto arr = json::array();
        for (const auto & tc : tool_calls) {
            arr.push_back({
                {"type", "function"},
                {"function", {
                    {"name", tc.name},
                    {"arguments", tc.arguments},
                }},
                {"id", tc.id},
                // // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
                // // We only generate a random id for the ones that don't generate one by themselves
                // // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
                // {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
            });
        }
        message["tool_calls"] = arr;
    }
    return message;
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::compute_diffs(const common_chat_msg & msg_prv, const common_chat_msg & msg_new) {
    std::vector<common_chat_msg_diff> diffs;
    if (msg_new.tool_calls.size() > msg_prv.tool_calls.size()) {
        diffs.reserve(msg_new.tool_calls.size() - msg_prv.tool_calls.size() + 3);
    } else {
        diffs.reserve(3);
    }

    // TODO: these can become expensive for long messages - how to optimize?
    if (msg_prv.reasoning_content != msg_new.reasoning_content) {
        auto & diff = diffs.emplace_back();
        diff.reasoning_content_delta = string_diff(msg_prv.reasoning_content, msg_new.reasoning_content);
    }
    if (msg_prv.content != msg_new.content) {
        auto & diff = diffs.emplace_back();
        diff.content_delta = string_diff(msg_prv.content, msg_new.content);
    }

    if (msg_new.tool_calls.size() < msg_prv.tool_calls.size()) {
        throw std::runtime_error("Invalid diff: now finding less tool calls!");
    }

    if (!msg_prv.tool_calls.empty()) {
        const auto idx = msg_prv.tool_calls.size() - 1;
        const auto & pref = msg_prv.tool_calls[idx];
        const auto & newf = msg_new.tool_calls[idx];
        if (pref.name != newf.name) {
            throw std::runtime_error("Invalid diff: tool call mismatch!");
        }
        const auto args_diff = string_diff(pref.arguments, newf.arguments);
        if (!args_diff.empty() || pref.id != newf.id) {
            auto & diff = diffs.emplace_back();
            diff.tool_call_index = idx;
            if (pref.id != newf.id) {
                diff.tool_call_delta.id = newf.id;
                diff.tool_call_delta.name = newf.name;
            }
            diff.tool_call_delta.arguments = args_diff;
        }
    }
    for (size_t idx = msg_prv.tool_calls.size(); idx < msg_new.tool_calls.size(); ++idx) {
        auto & diff = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = msg_new.tool_calls[idx];
    }

    return diffs;
}

typedef minja::chat_template common_chat_template;

struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template; // Model had builtin template or template overridde was specified.
    std::unique_ptr<common_chat_template> template_default; // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};

struct templates_params {
    json messages;
    json tools;
    common_chat_tool_choice tool_choice;
    json json_schema;
    bool parallel_tool_calls;
    common_reasoning_format reasoning_format;
    bool stream;
    std::string grammar;
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    json extra_context;
    bool add_bos;
    bool add_eos;
    bool is_inference = true;
};

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice) {
    if (tool_choice == "auto") {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    if (tool_choice == "none") {
        return COMMON_CHAT_TOOL_CHOICE_NONE;
    }
    if (tool_choice == "required") {
        return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    }
    throw std::invalid_argument("Invalid tool_choice: " + tool_choice);
}

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates) {
    common_chat_templates_inputs dummy_inputs;
    common_chat_msg msg;
    msg.role = "user";
    msg.content = "test";
    dummy_inputs.messages = {msg};
    dummy_inputs.enable_thinking = false;
    const auto rendered_no_thinking = common_chat_templates_apply(chat_templates, dummy_inputs);
    dummy_inputs.enable_thinking = true;
    const auto rendered_with_thinking = common_chat_templates_apply(chat_templates, dummy_inputs);
    return rendered_no_thinking.prompt != rendered_with_thinking.prompt;
}

template <>
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {

        if (!messages.is_array()) {
            throw std::invalid_argument("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::invalid_argument("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::invalid_argument("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role");

            auto has_content = message.contains("content");
            auto has_tool_calls = message.contains("tool_calls");
            if (has_content) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content;
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::invalid_argument("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text") {
                            throw std::invalid_argument("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::invalid_argument("Invalid 'content' type: expected string or array, got " + content.dump() + " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    common_chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::invalid_argument("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::invalid_argument("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::invalid_argument("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::invalid_argument("Missing tool call name: " + tool_call.dump());
                    }
                    tc.name = fc.at("name");
                    tc.arguments = fc.at("arguments");
                    if (tool_call.contains("id")) {
                        tc.id = tool_call.at("id");
                    }
                    msg.tool_calls.push_back(tc);
                }
            }
            if (!has_content && !has_tool_calls) {
                throw std::invalid_argument("Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & https://github.com/ggml-org/llama.cpp/issues/12279)");
            }
            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("name")) {
                msg.tool_name = message.at("name");
            }
            if (message.contains("tool_call_id")) {
                msg.tool_call_id = message.at("tool_call_id");
            }

            msgs.push_back(msg);
        }
    } catch (const std::exception & e) {
        // @ngxson : disable otherwise it's bloating the API response
        // printf("%s\n", std::string("; messages = ") + messages.dump(2));
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()));
    }

    return msgs;
}

template <>
json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    json messages = json::array();
    for (const auto & msg : msgs) {
        if (!msg.content.empty() && !msg.content_parts.empty()) {
            throw std::runtime_error("Cannot specify both content and content_parts");
        }
        json jmsg {
            {"role", msg.role},
        };
        if (!msg.content.empty()) {
            jmsg["content"] = msg.content;
        } else if (!msg.content_parts.empty()) {
            if (concat_typed_text) {
                std::string text;
                for (const auto & part : msg.content_parts) {
                    if (part.type != "text") {
                        LOG_WRN("Ignoring content part type: %s\n", part.type.c_str());
                        continue;
                    }
                    if (!text.empty()) {
                        text += '\n';
                    }
                    text += part.text;
                }
                jmsg["content"] = text;
            } else {
                auto & parts = jmsg["content"] = json::array();
                for (const auto & part : msg.content_parts) {
                    parts.push_back({
                        {"type", part.type},
                        {"text", part.text},
                    });
                }
            }
        } else {
            jmsg["content"] = json(); // null
        }
        if (!msg.reasoning_content.empty()) {
            jmsg["reasoning_content"] = msg.reasoning_content;
        }
        if (!msg.tool_name.empty()) {
            jmsg["name"] = msg.tool_name;
        }
        if (!msg.tool_call_id.empty()) {
            jmsg["tool_call_id"] = msg.tool_call_id;
        }
        if (!msg.tool_calls.empty()) {
            auto & tool_calls = jmsg["tool_calls"] = json::array();
            for (const auto & tool_call : msg.tool_calls) {
                json tc {
                    {"type", "function"},
                    {"function", {
                        {"name", tool_call.name},
                        {"arguments", tool_call.arguments},
                    }},
                };
                if (!tool_call.id.empty()) {
                    tc["id"] = tool_call.id;
                }
                tool_calls.push_back(tc);
            }
        }
        messages.push_back(jmsg);
    }
    return messages;
}

template <>
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const std::string & messages) {
    return common_chat_msgs_parse_oaicompat(json::parse(messages));
}

template <>
std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
    std::vector<common_chat_tool> result;

    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::invalid_argument("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::invalid_argument("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::invalid_argument("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::invalid_argument("Missing tool function: " + tool.dump());
                }

                const auto & function = tool.at("function");
                result.push_back({
                    /* .name = */ function.at("name"),
                    /* .description = */ function.at("description"),
                    /* .parameters = */ function.at("parameters").dump(),
                });
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }

    return result;
}

template <>
std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const std::string & tools) {
    return common_chat_tools_parse_oaicompat(json::parse(tools));
}

template <>
json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            {"type", "function"},
            {"function", {
                {"name", tool.name},
                {"description", tool.description},
                {"parameters", json::parse(tool.parameters)},
            }},
        });
    }
    return result;
}

template <> json common_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"] = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        json function = json::object();
        if (!diff.tool_call_delta.name.empty()) {
            function["name"] = diff.tool_call_delta.name;
        }
        function["arguments"] = diff.tool_call_delta.arguments;
        tool_call["function"] = function;
        delta["tool_calls"] = json::array({tool_call});
    }
    return delta;
}

bool common_chat_verify_template(const std::string & tmpl, bool use_jinja) {
    if (use_jinja) {
        try {
            common_chat_msg msg;
            msg.role = "user";
            msg.content = "test";

            auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl);

            common_chat_templates_inputs inputs;
            inputs.messages = {msg};

            common_chat_templates_apply(tmpls.get(), inputs);
            return true;
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to apply template: %s\n", __func__, e.what());
            return false;
        }
    }
    llama_chat_message chat[] = {{"user", "test"}};
    const int res = llama_chat_apply_template(tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string common_chat_format_single(
        const struct common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & past_msg,
        const common_chat_msg & new_msg,
        bool add_ass,
        bool use_jinja) {

    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos = tmpls->add_bos;
    inputs.add_eos = tmpls->add_eos;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg = common_chat_templates_apply(tmpls, inputs).prompt;
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg = common_chat_templates_apply(tmpls, inputs).prompt;
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string common_chat_format_example(const struct common_chat_templates * tmpls, bool use_jinja, const std::map<std::string, std::string> & chat_template_kwargs) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos = tmpls->add_bos;
    inputs.add_eos = tmpls->add_eos;
    inputs.chat_template_kwargs = chat_template_kwargs;
    auto add_simple_msg = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system",    "You are a helpful assistant");
    add_simple_msg("user",      "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user",      "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt;
}

#define CHATML_TEMPLATE_SRC \
    "{%- for message in messages -%}\n" \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n" \
    "{%- if add_generation_prompt -%}\n" \
    "  {{- '<|im_start|>assistant\n' -}}\n" \
    "{%- endif -%}"

void common_chat_templates_free(struct common_chat_templates * tmpls) {
    delete tmpls;
}

bool common_chat_templates_was_explicit(const struct common_chat_templates * tmpls) {
    return tmpls->has_explicit_template;
}

const char * common_chat_templates_source(const struct common_chat_templates * tmpls, const char * variant) {
    if (variant != nullptr) {
        if (strcmp(variant, "tool_use") == 0) {
            if (tmpls->template_tool_use) {
                return tmpls->template_tool_use->source().c_str();
            }
            return nullptr;
        } else {
            LOG_DBG("%s: unknown template variant: %s\n", __func__, variant);
        }
    }
    return tmpls->template_default->source().c_str();
}

common_chat_templates_ptr common_chat_templates_init(
    const struct llama_model * model,
    const std::string & chat_template_override,
    const std::string & bos_token_override,
    const std::string & eos_token_override)
{
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src = str;
            has_explicit_template = true;
        }
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
    } else {
        default_template_src = chat_template_override;
    }
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }

    // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
    // Ref: https://github.com/ggml-org/llama.cpp/pull/15230#issuecomment-3173959633
    if (default_template_src.find("<|channel|>") != std::string::npos
            // search for the error message and patch it
            && default_template_src.find("in message.content or") != std::string::npos) {
        string_replace_all(default_template_src,
            "{%- if \"<|channel|>analysis<|message|>\" in message.content or \"<|channel|>final<|message|>\" in message.content %}",
            "{%- if false %}");
    }

    // TODO @aldehir : this is a temporary fix, pending Minja changes
    // Ref: https://github.com/ggml-org/llama.cpp/pull/17713#issuecomment-3631342664
    if (default_template_src.find("[TOOL_CALLS]") != std::string::npos
            // search for the error message and patch it
            && default_template_src.find("if (message['content'] is none or") != std::string::npos) {
        string_replace_all(default_template_src,
            "{%- if (message['content'] is none or message['content'] == '' or message['content']|length == 0) and (message['tool_calls'] is not defined or message['tool_calls'] is none or message['tool_calls']|length == 0) %}",
            "{%- if false %}");
    }

    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    bool add_bos = false;
    bool add_eos = false;
    if (model) {
        const auto * vocab = llama_model_get_vocab(model);
        const auto get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                if (default_template_src.find(jinja_variable_name) != std::string::npos
                    || template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                    LOG_WRN("common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't work as intended.\n", name);
                }
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
        add_bos = llama_vocab_get_add_bos(vocab);
        add_eos = llama_vocab_get_add_eos(vocab);
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
    tmpls->add_bos = add_bos;
    tmpls->add_eos = add_eos;
    try {
        tmpls->template_default = std::make_unique<minja::chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to parse chat template (defaulting to chatml): %s \n", __func__, e.what());
        tmpls->template_default = std::make_unique<minja::chat_template>(CHATML_TEMPLATE_SRC, token_bos, token_eos);
    }
    if (!template_tool_use_src.empty()) {
        try {
            tmpls->template_tool_use = std::make_unique<minja::chat_template>(template_tool_use_src, token_bos, token_eos);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, e.what());
        }
    }
    return tmpls;
}

const char * common_chat_format_name(common_chat_format format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY: return "Content-only";
        case COMMON_CHAT_FORMAT_GENERIC: return "Generic";
        case COMMON_CHAT_FORMAT_MISTRAL_NEMO: return "Mistral Nemo";
        case COMMON_CHAT_FORMAT_MAGISTRAL: return "Magistral";
        case COMMON_CHAT_FORMAT_LLAMA_3_X: return "Llama 3.x";
        case COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS: return "Llama 3.x with builtin tools";
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1: return "DeepSeek R1";
        case COMMON_CHAT_FORMAT_FIREFUNCTION_V2: return "FireFunction v2";
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2: return "Functionary v3.2";
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1: return "Functionary v3.1 Llama 3.1";
        case COMMON_CHAT_FORMAT_DEEPSEEK_V3_1: return "DeepSeek V3.1";
        case COMMON_CHAT_FORMAT_HERMES_2_PRO: return "Hermes 2 Pro";
        case COMMON_CHAT_FORMAT_COMMAND_R7B: return "Command R7B";
        case COMMON_CHAT_FORMAT_GRANITE: return "Granite";
        case COMMON_CHAT_FORMAT_GPT_OSS: return "GPT-OSS";
        case COMMON_CHAT_FORMAT_SEED_OSS: return "Seed-OSS";
        case COMMON_CHAT_FORMAT_NEMOTRON_V2: return "Nemotron V2";
        case COMMON_CHAT_FORMAT_APERTUS: return "Apertus";
        case COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS: return "LFM2 with JSON tools";
        case COMMON_CHAT_FORMAT_MINIMAX_M2: return "MiniMax-M2";
        case COMMON_CHAT_FORMAT_GLM_4_5: return "GLM 4.5";
        case COMMON_CHAT_FORMAT_KIMI_K2: return "Kimi K2";
        case COMMON_CHAT_FORMAT_QWEN3_CODER_XML: return "Qwen3 Coder";
        case COMMON_CHAT_FORMAT_APRIEL_1_5: return "Apriel 1.5";
        case COMMON_CHAT_FORMAT_XIAOMI_MIMO: return "Xiaomi MiMo";
        case COMMON_CHAT_FORMAT_PEG_SIMPLE: return "peg-simple";
        case COMMON_CHAT_FORMAT_PEG_NATIVE: return "peg-native";
        case COMMON_CHAT_FORMAT_PEG_CONSTRUCTED: return "peg-constructed";
        default:
            throw std::runtime_error("Unknown chat format");
    }
}

const char * common_reasoning_format_name(common_reasoning_format format) {
    switch (format) {
        case COMMON_REASONING_FORMAT_NONE:     return "none";
        case COMMON_REASONING_FORMAT_AUTO:     return "auto";
        case COMMON_REASONING_FORMAT_DEEPSEEK: return "deepseek";
        case COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY: return "deepseek-legacy";
        default:
            throw std::runtime_error("Unknown reasoning format");
    }
}

common_reasoning_format common_reasoning_format_from_name(const std::string & format) {
    if (format == "none") {
        return COMMON_REASONING_FORMAT_NONE;
    } else if (format == "auto") {
        return COMMON_REASONING_FORMAT_AUTO;
    } else if (format == "deepseek") {
        return COMMON_REASONING_FORMAT_DEEPSEEK;
    } else if (format == "deepseek-legacy") {
        return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    }
    throw std::runtime_error("Unknown reasoning format: " + format);
}

static void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

static void foreach_parameter(const json & function, const std::function<void(const std::string &, const json &, bool)> & fn) {
    if (!function.contains("parameters") || !function.at("parameters").is_object()) {
        return;
    }
    const auto & params = function.at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        return;
    }
    const auto & props = params.at("properties");
    std::set<std::string> required;
    if (params.contains("required") && params.at("required").is_array()) {
        params.at("required").get_to(required);
    }
    for (const auto & [name, prop] : props.items()) {
        bool is_required = (required.find(name) != required.end());
        fn(name, prop, is_required);
    }
}

static std::string apply(
    const common_chat_template & tmpl,
    const struct templates_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt)
{
    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages = messages_override ? *messages_override : inputs.messages;
    if (tools_override) {
        tmpl_inputs.tools = *tools_override;
    } else {
        tmpl_inputs.tools = inputs.tools.empty() ? json() : inputs.tools;
    }
    tmpl_inputs.add_generation_prompt = inputs.add_generation_prompt;
    tmpl_inputs.extra_context = inputs.extra_context;
    tmpl_inputs.extra_context["enable_thinking"] = inputs.enable_thinking;
    if (additional_context) {
        tmpl_inputs.extra_context.merge_patch(*additional_context);
    }
    // TODO: add flag to control date/time, if only for testing purposes.
    // tmpl_inputs.now = std::chrono::system_clock::now();

    minja::chat_template_options tmpl_opts;
    // To avoid double BOS / EOS tokens, we're manually removing begining / trailing tokens
    // instead of using `chat_template_options.use_bos_token = false`, since these tokens
    // may be needed inside the template / between messages too.
    auto result = tmpl.apply(tmpl_inputs, tmpl_opts);
    if (inputs.add_bos && string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(result, tmpl.eos_token())) {
        result = result.substr(0, result.size() - tmpl.eos_token().size());
    }
    return result;
}

static common_chat_params common_chat_params_init_generic(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    auto tool_call_schemas = json::array();
    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & function = tool.at("function");
        auto tool_schema = json {
            {"type", "object"},
            {"properties", {
                {"name", {
                    {"type", "string"},
                    {"const", function.at("name")},
                }},
                {"arguments", function.at("parameters")},
            }},
            {"required", json::array({"name", "arguments"})},
        };
        if (function.contains("description")) {
            tool_schema["description"] = function.at("description");
        }
        if (inputs.parallel_tool_calls) {
            tool_schema.at("properties")["id"] = {
                {"type", "string"},
                {"minLength", 4},
            };
            tool_schema.at("required").push_back("id");
        }
        tool_call_schemas.emplace_back(tool_schema);
    });
    const auto tool_call =
        inputs.parallel_tool_calls
            ? json {
                {"type", "object"},
                {"properties", {
                    {"tool_calls", {
                        {"type", "array"},
                        {"items", tool_call_schemas.size() == 1 ? tool_call_schemas[0] : json {
                            {"anyOf", tool_call_schemas},
                        }},
                        {"minItems", 1},
                    }},
                }},
                {"required", json::array({"tool_calls"})},
            }
            : json {
                {"type", "object"},
                {"properties", {
                    {"tool_call", tool_call_schemas.size() == 1 ? tool_call_schemas[0] : json {
                        {"anyOf", tool_call_schemas},
                    }},
                }},
                {"required", json::array({"tool_call"})},
            };
    const auto schema =
        inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED
            ? json {
                {"anyOf", json::array({
                    tool_call,
                    {
                        {"type", "object"},
                        {"properties", {
                            {"response", inputs.json_schema.is_null()
                                ? json {{"type", "string"}}
                                : inputs.json_schema
                            },
                        }},
                        {"required", json::array({"response"})},
                    },
                })}
            }
            : tool_call;

    data.grammar_lazy = false;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        builder.add_schema("root", schema);
    });

    auto tweaked_messages = common_chat_template::add_system(
        inputs.messages,
        "Respond in JSON format, either with `tool_call` (a request to call tools) or with `response` reply to the user's request");

    data.prompt = apply(tmpl, inputs, /* messages_override= */ tweaked_messages);
    data.format = COMMON_CHAT_FORMAT_GENERIC;
    return data;
}

static common_chat_params common_chat_params_init_mistral_nemo(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        auto schemas = json::array();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            schemas.push_back({
                {"type", "object"},
                {"properties", {
                    // Important note: the model is probably trained to take a JSON stringified arguments value.
                    // It's hard to constrain that for now (while reusing the JSON schema conversion), so we're just expecting a plain object.
                    {"name", {
                        {"type", "string"},
                        {"const", function.at("name")},
                    }},
                    {"arguments", function.at("parameters")},
                    {"id", {
                        {"type", "string"},
                        // Nemo's template expects a 9-character alphanumeric ID.
                        {"pattern", "^[a-zA-Z0-9]{9}$"},
                    }},
                }},
                {"required", json::array({"name", "arguments", "id"})},
            });
        });
        auto schema = json {
            {"type", "array"},
            {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
            {"minItems", 1},
        };
        if (!inputs.parallel_tool_calls) {
            schema["maxItems"] = 1;
        }
        builder.add_rule("root", "\"[TOOL_CALLS]\" " + builder.add_schema("tool_calls", schema));
    });
    data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
    data.preserved_tokens = {
        "[TOOL_CALLS]",
    };
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_MISTRAL_NEMO;
    return data;
}


// Case-insensitive find
static size_t ifind_string(const std::string & haystack, const std::string & needle, size_t pos = 0) {
    auto it = std::search(
        haystack.begin() + pos, haystack.end(),
        needle.begin(), needle.end(),
        [](char a, char b) { return std::tolower(a) == std::tolower(b); }
    );
    return (it == haystack.end()) ? std::string::npos : std::distance(haystack.begin(), it);
}

static common_chat_params common_chat_params_init_lfm2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    const auto is_json_schema_provided = !inputs.json_schema.is_null();
    const auto is_grammar_provided = !inputs.grammar.empty();
    const auto are_tools_provided = inputs.tools.is_array() && !inputs.tools.empty();

    // the logic requires potentially modifying the messages
    auto tweaked_messages = inputs.messages;

    auto replace_json_schema_marker = [](json & messages) -> bool {
        static std::string marker1 = "force json schema.\n";
        static std::string marker2 = "force json schema.";

        if (messages.empty() || messages.at(0).at("role") != "system") {
            return false;
        }

        std::string content = messages.at(0).at("content");

        for (const auto & marker : {marker1, marker2}) {
            const auto pos = ifind_string(content, marker);
            if (pos != std::string::npos) {
                content.replace(pos, marker.length(), "");
                // inject modified content back into the messages
                messages.at(0).at("content") = content;
                return true;
            }
        }

        return false;
    };

    // Lfm2 model does not natively work with json, but can generally understand the tools structure
    //
    // Example of the pytorch dialog structure:
    //     <|startoftext|><|im_start|>system
    //     List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
    //     <|im_start|>user
    //     What is the current status of candidate ID 12345?<|im_end|>
    //     <|im_start|>assistant
    //     <|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
    //     <|im_start|>tool
    //     <|tool_response_start|>{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}<|tool_response_end|><|im_end|>
    //     <|im_start|>assistant
    //     The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
    //
    // For the llama server compatibility with json tools semantic,
    // the client can add "Follow json schema." line into the system message prompt to force the json output.
    //
    if (are_tools_provided && (is_json_schema_provided || is_grammar_provided)) {
        // server/utils.hpp prohibits that branch for the custom grammar anyways
        throw std::runtime_error("Tools call must not use \"json_schema\" or \"grammar\", use non-tool invocation if you want to use custom grammar");
    } else if (are_tools_provided && replace_json_schema_marker(tweaked_messages)) {
        LOG_INF("%s: Using tools to build a grammar\n", __func__);

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    {"type", "object"},
                    {"properties", {
                        {"name", {
                            {"type", "string"},
                            {"const", function.at("name")},
                        }},
                        {"arguments", function.at("parameters")},
                    }},
                    {"required", json::array({"name", "arguments", "id"})},
                });
            });
            auto schema = json {
                {"type", "array"},
                {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
                {"minItems", 1},
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }

            builder.add_rule("root", "\"<|tool_call_start|>\"" + builder.add_schema("tool_calls", schema) + "\"<|tool_call_end|>\"");
        });
        // model has no concept of tool selection mode choice,
        // if the system prompt rendered correctly it will produce a tool call
        // the grammar goes inside the tool call body
        data.grammar_lazy = true;
        data.grammar_triggers = {{COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, "\\s*<\\|tool_call_start\\|>\\s*\\["}};
        data.preserved_tokens = {"<|tool_call_start|>", "<|tool_call_end|>"};
        data.format = COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS;
    } else if (are_tools_provided && (!is_json_schema_provided && !is_grammar_provided)) {
        LOG_INF("%s: Using tools without json schema or grammar\n", __func__);
        // output those tokens
        data.preserved_tokens = {"<|tool_call_start|>", "<|tool_call_end|>"};
    } else if (is_json_schema_provided) {
        LOG_INF("%s: Using provided json schema to build a grammar\n", __func__);
        data.grammar = json_schema_to_grammar(inputs.json_schema);
    } else if (is_grammar_provided) {
        LOG_INF("%s: Using provided grammar\n", __func__);
        data.grammar = inputs.grammar;
    } else {
        LOG_INF("%s: Using content relying on the template\n", __func__);
    }

    data.prompt = apply(tmpl, inputs, /* messages_override= */ tweaked_messages);
    LOG_DBG("%s: Prompt: %s\n", __func__, data.prompt.c_str());

    return data;
}

static common_chat_params common_chat_params_init_ministral_3(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Build up messages to follow the format: https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512/blob/main/chat_template.jinja
    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto role = msg.value("role", "");
        if (role != "system" && role != "assistant") {
            // Only adjust system and assistant messages. Interestingly, the system message may contain thinking.
            adjusted_messages.push_back(msg);
            continue;
        }

        auto content = json::array();

        // If message contains `reasoning_content`, add it as a block of type `thinking`
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            content.push_back({
                {"type", "thinking"},
                {"thinking", msg.at("reasoning_content").get<std::string>()},
            });
        }

        // If message contains `content`, add it as a block of type `text`
        if (msg.contains("content")) {
            if (msg.at("content").is_string()) {
                content.push_back({
                    {"type", "text"},
                    {"text", msg.at("content").get<std::string>()},
                });
            } else if (msg.at("content").is_array()) {
                auto blocks = msg.at("content");
                content.insert(content.end(), blocks.begin(), blocks.end());
            }
        }

        auto adjusted = msg;
        adjusted["content"] = content;
        adjusted.erase("reasoning_content");
        adjusted_messages.push_back(adjusted);
    }

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

    data.prompt = apply(tmpl, inputs, /* messages_override = */ adjusted_messages);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = {
        "[THINK]",
        "[/THINK]",
        "[TOOL_CALLS]",
        "[ARGS]",
    };

    auto parser = build_chat_peg_native_parser([&](common_chat_peg_native_builder & p) {
        auto reasoning = extract_reasoning ? p.optional("[THINK]" + p.reasoning(p.until("[/THINK]")) + "[/THINK]") : p.eps();

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            // Ministral wants to emit json surrounded by code fences
            return reasoning << "```json" << p.content(p.schema(p.json(), "response-format", inputs.json_schema)) << "```";
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                const auto & schema = function.at("parameters");

                tool_choice |= p.rule("tool-" + name,
                    p.tool_open(p.tool_name(p.literal(name)) + "[ARGS]")
                    + p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema))
                );
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call", p.repeat("[TOOL_CALLS]" + tool_choice, min_calls, max_calls));

            return reasoning << p.content(p.until("[TOOL_CALLS]")) << tool_calls;
        }

        // Content only parser
        include_grammar = false;
        return reasoning << p.content(p.rest());
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto schema = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"}
        };
    }

    return data;
}

static common_chat_params common_chat_params_init_magistral(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_MAGISTRAL;
    data.preserved_tokens = {
        "[THINK]",
        "[/THINK]",
    };

    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    {"type", "object"},
                    {"properties", {
                        {"name", {
                            {"type", "string"},
                            {"const", function.at("name")},
                        }},
                        {"arguments", function.at("parameters")},
                        {"id", {
                            {"type", "string"},
                            {"pattern", "^[a-zA-Z0-9]{9}$"},
                        }},
                    }},
                    {"required", json::array({"name", "arguments", "id"})},
                });
            });
            auto schema = json {
                {"type", "array"},
                {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
                {"minItems", 1},
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }
            builder.add_rule("root", "\"[TOOL_CALLS]\" " + builder.add_schema("tool_calls", schema));
        });
        data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
        data.preserved_tokens.push_back("[TOOL_CALLS]");
    } else {
        data.grammar_lazy = false;
        if (!inputs.json_schema.is_null()) {
            if (!inputs.grammar.empty()) {
                throw std::runtime_error("Either \"json_schema\" or \"grammar\" can be specified, but not both");
            }
            data.grammar = json_schema_to_grammar(inputs.json_schema);
        } else {
            data.grammar = inputs.grammar;
        }
    }

    return data;
}

static common_chat_params common_chat_params_init_command_r7b(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto has_reasoning_content = msg.contains("reasoning_content") && msg.at("reasoning_content").is_string();
        auto has_tool_calls = msg.contains("tool_calls") && msg.at("tool_calls").is_array();
        if (has_reasoning_content && has_tool_calls) {
            auto adjusted_message = msg;
            adjusted_message["tool_plan"] = msg.at("reasoning_content");
            adjusted_message.erase("reasoning_content");
            adjusted_messages.push_back(adjusted_message);
        } else {
            adjusted_messages.push_back(msg);
        }
    }
    data.prompt = apply(tmpl, inputs, /* messages_override= */ adjusted_messages);
    data.format = COMMON_CHAT_FORMAT_COMMAND_R7B;
    if (string_ends_with(data.prompt, "<|START_THINKING|>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "<|END_THINKING|>";
        } else {
            data.thinking_forced_open = true;
        }
    } else if (!inputs.enable_thinking && string_ends_with(data.prompt, "<|CHATBOT_TOKEN|>")) {
        data.prompt += "<|START_THINKING|><|END_THINKING|>";
    }

    data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        auto schemas = json::array();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            schemas.push_back({
                {"type", "object"},
                {"properties", {
                    {"tool_call_id", {
                        {"type", "string"},
                        // Command-R's template expects an integer string.
                        {"pattern", "^[0-9]{1,10}$"},
                    }},
                    {"tool_name", {
                        {"type", "string"},
                        {"const", function.at("name")},
                    }},
                    {"parameters", function.at("parameters")},
                }},
                {"required", json::array({"tool_call_id", "tool_name", "parameters"})},
            });
        });
        auto schema = json {
            {"type", "array"},
            {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
            {"minItems", 1},
        };
        if (!inputs.parallel_tool_calls) {
            schema["maxItems"] = 1;
        }
        builder.add_rule("root",
            std::string(data.thinking_forced_open ? "( \"<|END_THINKING|>\" space )? " : "") +
            "\"<|START_ACTION|>\" " + builder.add_schema("tool_calls", schema) + " \"<|END_ACTION|>\"");
    });
    data.grammar_triggers.push_back({
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
        // If thinking_forced_open, then we capture the </think> tag in the grammar,
        // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
        std::string(data.thinking_forced_open ? "[\\s\\S]*?(<\\|END_THINKING\\|>\\s*)" : "(?:<\\|START_THINKING\\|>[\\s\\S]*?<\\|END_THINKING\\|>\\s*)?") +
            "(<\\|START_ACTION\\|>)[\\s\\S]*"
    });
    data.preserved_tokens = {
        "<|START_ACTION|>",
        "<|END_ACTION|>",
        "<|START_RESPONSE|>",
        "<|END_RESPONSE|>",
        "<|START_THINKING|>",
        "<|END_THINKING|>",
    };
    return data;
}

static void expect_tool_parameters(const std::string & name, const json & parameters, const std::vector<std::string> & expected_properties) {
    if (!parameters.is_object() || !parameters.contains("type") || parameters.at("type") != "object" || !parameters.contains("properties") || !parameters.contains("required")) {
        throw std::runtime_error("Parameters of tool " + name + " must be an object w/ required properties");
    }
    const auto & parameters_properties = parameters.at("properties");
    const auto & parameters_required = parameters.at("required");
    for (const auto & prop : expected_properties) {
        if (!parameters_properties.contains(prop)) {
            throw std::runtime_error("Parameters of tool " + name + " is missing property: " + prop); // NOLINT
        }
        if (std::find(parameters_required.begin(), parameters_required.end(), json(prop)) == parameters_required.end()) {
            throw std::runtime_error("Parameters of tool " + name + " must have property marked as required: " + prop); // NOLINT
        }
    }
    if (parameters_properties.size() != expected_properties.size()) {
        throw std::runtime_error("Parameters of tool " + name + " must only have these properties:" + string_join(expected_properties, ", "));
    }
}

static common_chat_params common_chat_params_init_llama_3_x(const common_chat_template & tmpl, const struct templates_params & inputs, bool allow_python_tag_builtin_tools) {
    auto builtin_tools = json::array();
    common_chat_params data;
    if (!inputs.tools.is_null()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;

            auto handle_builtin_tool = [&](const std::string & name, const json & parameters) {
                if (name == "wolfram_alpha" || name == "web_search" || name == "brave_search") {
                    // https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/remote/tool_runtime/wolfram_alpha/wolfram_alpha.py
                    // https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/remote/tool_runtime/brave_search/brave_search.py
                    expect_tool_parameters(name, parameters, {"query"});
                } else if (name == "python" || name == "code_interpreter") {
                    // https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/inline/tool_runtime/code_interpreter/code_interpreter.py
                    expect_tool_parameters(name, parameters, {"code"});
                } else {
                    return false;
                }

                std::vector<std::string> kvs;
                for (const auto & [key, value] : parameters.at("properties").items()) {
                    kvs.push_back("\"" + key + "=\" " + builder.add_schema(name + "-args-" + key, value)); // NOLINT
                }

                tool_rules.push_back(
                    builder.add_rule(
                        name + "-call",
                        "\"<|python_tag|>" + name + ".call(\" " + string_join(kvs, " \", \" ") + " \")\""));
                builtin_tools.push_back(name);

                return true;
            };

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);

                // https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/remote/tool_runtime
                if (allow_python_tag_builtin_tools) {
                    handle_builtin_tool(name, parameters);
                }
                tool_rules.push_back(
                    builder.add_rule(
                        name + "-call",
                        "\"{\" space "
                        "( \"\\\"type\\\"\"       space \":\" space \"\\\"function\\\"\"     space \",\" space )? "
                        "  \"\\\"name\\\"\"       space \":\" space \"\\\"" + name + "\\\"\" space \",\" space "
                        "  \"\\\"parameters\\\"\" space \":\" space " + builder.add_schema(name + "-args", parameters) + " "
                        "\"}\" space"));
            });
            // Small models may hallucinate function names so we match anything (*at the start*) that looks like the JSON of a function call, regardless of the name.
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                "(\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\")[\\s\\S]*", // + name + "\"[\\s\\S]*",
            });
            if (!builtin_tools.empty()) {
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|python_tag|>"});
                data.preserved_tokens.push_back("<|python_tag|>");
            }
            // Allow a few empty lines on top of the usual constrained json schema space rule.
            builder.add_rule("root", string_join(tool_rules, " | "));
            data.additional_stops.push_back("<|eom_id|>");
        });
        data.format = allow_python_tag_builtin_tools && !builtin_tools.empty()
            ? COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS
            : COMMON_CHAT_FORMAT_LLAMA_3_X;
    } else {
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    }
    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, /* tools_override= */ std::nullopt, json {
        {"date_string", format_time(inputs.now, "%d %b %Y")},
        {"tools_in_user_message", false},
        {"builtin_tools", builtin_tools.empty() ? json() : builtin_tools},
    });
    return data;
}

static common_chat_params common_chat_params_init_nemotron_v2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Generate the prompt using the apply() function with the template
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_NEMOTRON_V2;

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    // When tools are present, build grammar for the <TOOLCALL> format, similar to CommandR, but without tool call ID
    if (!inputs.tools.is_null() && inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = true;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    { "type",       "object"                                                   },
                    { "properties",
                        {
                            { "name",
                            {
                                { "type", "string" },
                                { "const", function.at("name") },
                            } },
                            { "arguments", function.at("parameters") },
                        }                                                                        },
                    { "required",   json::array({ "name", "arguments" }) },
                });
            });
            auto schema = json{
                        { "type",     "array"                                                         },
                        { "items",    schemas.size() == 1 ? schemas[0] : json{ { "anyOf", schemas } } },
                        { "minItems", 1                                                               },
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }
            builder.add_rule("root",
                                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                                    "\"<TOOLCALL>\" " + builder.add_schema("tool_calls", schema) +
                                    " \"</TOOLCALL>\"");
        });
        data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
            // If thinking_forced_open, then we capture the </think> tag in the grammar,
            // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
            std::string(data.thinking_forced_open ?
                            "[\\s\\S]*?(</think>\\s*)" :
                            "(?:<think>[\\s\\S]*?</think>\\s*)?") +
                "(<TOOLCALL>)[\\s\\S]*" });
    }
    return data;
}

static common_chat_params common_chat_params_init_nemotron_v3(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar = true;

    auto parser = build_chat_peg_constructed_parser([&](auto & p) {
        auto reasoning = p.eps();
        if (inputs.enable_thinking && extract_reasoning) {
            auto reasoning_content = p.reasoning(p.until("</think>")) + ("</think>" | p.end());
            if (data.thinking_forced_open) {
                reasoning = reasoning_content;
            }
        }

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return reasoning << p.content(p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");

                auto schema_info = common_schema_info();
                schema_info.resolve_refs(parameters);

                auto tool_open = "<function=" + p.tool_name(p.literal(name)) + ">\n";
                auto tool_close = p.literal("</function>\n");
                auto args = p.sequence();
                auto arg_string = p.rule("xml-arg-string", p.until_one_of({
                    "\n</parameter>",
                    "\n<parameter=",
                    "\n</function>"
                }));

                foreach_parameter(function, [&](const auto & param_name, const json & param_schema, bool is_required) {
                    auto rule_name = "tool-" + name + "-arg-" + param_name;

                    auto arg_open = "<parameter=" + p.tool_arg_name(p.literal(param_name)) + ">\n";
                    auto arg_close = p.literal("</parameter>\n");
                    auto arg_value = p.eps();

                    if (schema_info.resolves_to_string(param_schema)) {
                        arg_value = p.tool_arg_string_value(arg_string) + "\n";
                    } else {
                        arg_value = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", param_schema));
                    }

                    // Model may or my not close with </parameter>
                    auto arg_rule = p.rule(rule_name, p.tool_arg_open(arg_open) + arg_value + p.optional(p.tool_arg_close(arg_close)));
                    args += p.repeat(arg_rule, /* min = */ is_required ? 1 : 0, /* max = */ 1);
                });

                tool_choice |= p.rule("tool-" + name, p.tool_open(tool_open) + args + p.tool_close(tool_close));
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_call = p.rule("tool-call", "<tool_call>\n" + tool_choice + "</tool_call>" + p.space());
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, /* min = */ min_calls, /* max = */ max_calls));

            return reasoning << p.content(p.until("<tool_call>")) << tool_calls;
        }

        // Content only parser
        include_grammar = false;
        return reasoning << p.content(p.rest());
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto schema = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>"}
        };
    }

    return data;
}


static common_chat_params common_chat_params_init_apertus(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Generate the prompt using the apply() function with the template
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_APERTUS;

    // Handle thinking tags appropriately based on inputs.enable_thinking
    if (string_ends_with(data.prompt, "<|inner_prefix|>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "<|inner_suffix|>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    // When tools are present, build grammar for the <|tools_prefix|> format
    if (!inputs.tools.is_null() && inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = true;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    { "type",       "object"                                                   },
                    { "properties",
                        {
                            { function.at("name"), function.at("parameters") }
                        }                                                                        },
                    { "required",   json::array({ function.at("name") }) },
                });
            });
            auto schema = json{
                        { "type",     "array"                                                         },
                        { "items",    schemas.size() == 1 ? schemas[0] : json{ { "anyOf", schemas } } },
                        { "minItems", 1                                                               },
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }
            builder.add_rule("root",
                                std::string(data.thinking_forced_open ? "( \"<|inner_suffix|>\" space )? " : "") +
                                    "\"<|tools_prefix|>\"" + builder.add_schema("tool_calls", schema) + "\"<|tools_suffix|>\"");
                            });
        data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
            // If thinking_forced_open, then we capture the <|inner_suffix|> tag in the grammar,
            // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
            std::string(data.thinking_forced_open ?
                            "[\\s\\S]*?(<\\|inner_suffix\\|>\\s*)" :
                            "(?:<\\|inner_prefix\\|>[\\s\\S]*?<\\|inner_suffix\\|>\\s*)?") +
                "(<\\|tools_prefix\\|>)[\\s\\S]*" });
        data.preserved_tokens = {
            "<|system_start|>",
            "<|system_end|>",
            "<|developer_start|>",
            "<|developer_end|>",
            "<|user_start|>",
            "<|user_end|>",
            "<|assistant_start|>",
            "<|assistant_end|>",
            "<|inner_prefix|>",
            "<|inner_suffix|>",
            "<|tools_prefix|>",
            "<|tools_suffix|>",
        };
    }
    return data;
}

static common_chat_params common_chat_params_init_deepseek_r1(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    auto prompt = apply(tmpl, inputs);

    // Hacks to fix the official (broken) prompt.
    // It is advisable to use --chat-template-file models/templates/llama-cpp-deepseek-r1.jinja instead,
    // until the official template is fixed.
    if (tmpl.source().find("{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}") != std::string::npos) {
        // Don't leave the chat dangling after tool results
        if (string_ends_with(prompt, "<｜tool▁outputs▁end｜>")) {
            prompt += "<｜end▁of▁sentence｜>";
            if (inputs.add_generation_prompt) {
                prompt += "<｜Assistant｜>";
            }
        }
        // Fix up tool call delta example added by Minja
        prompt = std::regex_replace(
            prompt,
            std::regex("(<｜tool▁call▁end｜>)[\\s\\r\\n]*(<｜tool▁outputs▁begin｜>|<｜User｜>)"),
            "$1<｜tool▁calls▁end｜><｜end▁of▁sentence｜>$2");
    }
    data.prompt = prompt;
    data.format = COMMON_CHAT_FORMAT_DEEPSEEK_R1;
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED && inputs.json_schema.is_null();
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_rule(name + "-call",
                    "( \"<｜tool▁call▁begin｜>\" )? \"function<｜tool▁sep｜>" + name + "\\n"
                    "```json\\n\" " + builder.add_schema(name + "-args", parameters) + " "
                    "\"```<｜tool▁call▁end｜>\""));
            });
            // Distill Qwen 7B & 32B models seem confused re/ syntax of their tool call opening tag,
            // so we accept common variants (then it's all constrained)
            builder.add_rule("root",
                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                "( \"<｜tool▁calls▁begin｜>\" | \"<｜tool_calls_begin｜>\" | \"<｜tool calls begin｜>\" | \"<｜tool\\\\_calls\\\\_begin｜>\" | \"<｜tool▁calls｜>\" ) "
                "(" + string_join(tool_rules, " | ") + ")" + (inputs.parallel_tool_calls ? "*" : "") + " "
                "\"<｜tool▁calls▁end｜>\""
                " space");
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                // If thinking_forced_open, then we capture the </think> tag in the grammar,
                // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
                std::string(data.thinking_forced_open ? "[\\s\\S]*?(</think>\\s*)" : "(?:<think>[\\s\\S]*?</think>\\s*)?") +
                    "(<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)[\\s\\S]*"
            });
            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁call▁begin｜>",
                "<｜tool▁sep｜>",
                "<｜tool▁call▁end｜>",
                "<｜tool▁calls▁end｜",
            };
        });
    }
    return data;
}

static common_chat_params common_chat_params_init_deepseek_v3_1(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Pass thinking context for DeepSeek V3.1 template
    json additional_context = {
        {"thinking", inputs.enable_thinking},
    };

    auto prompt = apply(tmpl, inputs,
                       /* messages_override= */ inputs.messages,
                       /* tools_override= */ std::nullopt,
                       additional_context);
    data.prompt = prompt;
    data.format = COMMON_CHAT_FORMAT_DEEPSEEK_V3_1;
    if (string_ends_with(data.prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED && inputs.json_schema.is_null();
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_rule(name + "-call",
                    "( \"<｜tool▁call▁begin｜>\" )? \"" + name + "<｜tool▁sep｜>"
                    "\" " + builder.add_schema(name + "-args", parameters) + " "
                    "\"<｜tool▁call▁end｜>\""));
            });
            // Distill Qwen 7B & 32B models seem confused re/ syntax of their tool call opening tag,
            // so we accept common variants (then it's all constrained)
            builder.add_rule("root",
                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                "( \"<｜tool▁calls▁begin｜>\" | \"<｜tool_calls_begin｜>\" | \"<｜tool calls begin｜>\" | \"<｜tool\\\\_calls\\\\_begin｜>\" | \"<｜tool▁calls｜>\" ) "
                "(" + string_join(tool_rules, " | ") + ")" + (inputs.parallel_tool_calls ? "*" : "") + " "
                "\"<｜tool▁calls▁end｜>\""
                " space");
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                // If thinking_forced_open, then we capture the </think> tag in the grammar,
                // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
                std::string(data.thinking_forced_open ? "[\\s\\S]*?(</think>\\s*)" : "(?:<think>[\\s\\S]*?</think>\\s*)?") +
                    "(<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)[\\s\\S]*"
            });
            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁call▁begin｜>",
                "<｜tool▁sep｜>",
                "<｜tool▁call▁end｜>",
                "<｜tool▁calls▁end｜>",
            };
        });
    }
    return data;
}

static common_chat_params common_chat_params_init_minimax_m2(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_MINIMAX_M2;

    // Handle thinking tags based on prompt ending
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!params.enable_thinking) {
            // Close the thinking tag immediately if thinking is disabled
            data.prompt += "</think>\n\n";
        } else {
            // Mark thinking as forced open (template started with <think>)
            data.thinking_forced_open = true;
        }
    }

    // Preserve MiniMax-M2 special tokens
    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<minimax:tool_call>",
        "</minimax:tool_call>",
    };

    // build grammar for tool call
    static const xml_tool_call_format form {
        /* form.scope_start = */ "<minimax:tool_call>\n",
        /* form.tool_start  = */ "<invoke name=\"",
        /* form.tool_sep    = */ "\">\n",
        /* form.key_start   = */ "<parameter name=\"",
        /* form.key_val_sep = */ "\">",
        /* form.val_end     = */ "</parameter>\n",
        /* form.tool_end    = */ "</invoke>\n",
        /* form.scope_end   = */ "</minimax:tool_call>",
    };
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static common_chat_params common_chat_params_init_qwen3_coder_xml(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_QWEN3_CODER_XML;

    data.preserved_tokens = {
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    };

    // build grammar for tool call
    static const xml_tool_call_format form {
        /* form.scope_start = */ "<tool_call>\n",
        /* form.tool_start  = */ "<function=",
        /* form.tool_sep    = */ ">\n",
        /* form.key_start   = */ "<parameter=",
        /* form.key_val_sep = */ ">\n",
        /* form.val_end     = */ "\n</parameter>\n",
        /* form.tool_end    = */ "</function>\n",
        /* form.scope_end   = */ "</tool_call>",
    };
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static common_chat_params common_chat_params_init_kimi_k2(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_KIMI_K2;

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
        "<|im_end|>",
        "<|im_system|>",
        "<|im_middle|>",
    };

    data.additional_stops.insert(data.additional_stops.end(), {
        "<|im_end|>",
        "<|im_middle|>"
    });
    // build grammar for tool call
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "<|tool_calls_section_begin|>";
        form.tool_start  = "<|tool_call_begin|>";
        form.tool_sep    = "<|tool_call_argument_begin|>{";
        form.key_start   = "\"";
        form.key_val_sep = "\": ";
        form.val_end     = ", ";
        form.tool_end    = "}<|tool_call_end|>";
        form.scope_end   = "<|tool_calls_section_end|>";
        form.raw_argval  = false;
        form.last_val_end = "";
        return form;
    })();
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static common_chat_params common_chat_params_init_apriel_1_5(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_APRIEL_1_5;

    data.preserved_tokens = {
        "<thinking>",
        "</thinking>",
        "<tool_calls>",
        "</tool_calls>",
    };

    // build grammar for tool call
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "<tool_calls>[";
        form.tool_start  = "{\"name\": \"";
        form.tool_sep    = "\", \"arguments\": {";
        form.key_start   = "\"";
        form.key_val_sep = "\": ";
        form.val_end     = ", ";
        form.tool_end    = "}, ";
        form.scope_end   = "]</tool_calls>";
        form.raw_argval  = false;
        form.last_val_end = "";
        form.last_tool_end = "}";
        return form;
    })();
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static common_chat_params common_chat_params_init_xiaomi_mimo(const common_chat_template & tmpl, const struct templates_params & params) {
    common_chat_params data;
    data.grammar_lazy = params.tools.is_array() && !params.tools.empty() && params.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_XIAOMI_MIMO;

    data.preserved_tokens = {
        "<tool_call>",
        "</tool_call>",
    };

    // build grammar for tool call
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "\n";
        form.tool_start  = "<tool_call>\n{\"name\": \"";
        form.tool_sep    = "\", \"arguments\": {";
        form.key_start   = "\"";
        form.key_val_sep = "\": ";
        form.val_end     = ", ";
        form.tool_end    = "}\n</tool_call>";
        form.scope_end   = "";
        form.raw_argval  = false;
        form.last_val_end = "";
        return form;
    })();
    build_grammar_xml_tool_call(data, params.tools, form);

    return data;
}

static common_chat_params common_chat_params_init_gpt_oss(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Copy reasoning to the "thinking" field as expected by the gpt-oss template
    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto has_reasoning_content = msg.contains("reasoning_content") && msg.at("reasoning_content").is_string();
        auto has_tool_calls = msg.contains("tool_calls") && msg.at("tool_calls").is_array();

        if (has_reasoning_content && has_tool_calls) {
            auto adjusted_message = msg;
            adjusted_message["thinking"] = msg.at("reasoning_content");
            adjusted_messages.push_back(adjusted_message);
        } else {
            adjusted_messages.push_back(msg);
        }
    }

    auto prompt = apply(tmpl, inputs, /* messages_override= */ adjusted_messages);

    // Check if we need to replace the return token with end token during
    // inference and without generation prompt. For more details see:
    // https://github.com/ggml-org/llama.cpp/issues/15417
    if (inputs.is_inference && !inputs.add_generation_prompt) {
        static constexpr std::string_view return_token = "<|return|>";
        static constexpr std::string_view end_token    = "<|end|>";
        if (size_t pos = prompt.rfind(return_token); pos != std::string::npos) {
            prompt.replace(pos, return_token.length(), end_token);
        }
    }

    data.prompt = prompt;
    data.format = COMMON_CHAT_FORMAT_GPT_OSS;

    // These special tokens are required to parse properly, so we include them
    // even if parse_tool_calls is false.
    data.preserved_tokens = {
        "<|channel|>",
        "<|constrain|>",
        "<|message|>",
        "<|start|>",
        "<|end|>",
    };

    if (!inputs.json_schema.is_null()) {
        data.grammar_lazy = false;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schema = inputs.json_schema;
            builder.resolve_refs(schema);

            auto not_end = builder.add_rule("not-end",
                "[^<] | \"<\" [^|] | \"<|\" [^e] | \"<|e\" [^n] | \"<|en\" [^d] | \"<|end\" [^|] | \"<|end|\" [^>]");
            auto analysis = builder.add_rule("analysis",
                "\"<|channel|>analysis<|message|>\" ( " + not_end + " )* \"<|end|>\"");
            auto constraint = builder.add_rule("constraint", "\"<|constrain|>\"? [a-zA-Z0-9_-]+");
            auto final = builder.add_rule("final",
                "\"<|channel|>final\" ( \" \" " + constraint + " )? \"<|message|>\" " +
                builder.add_schema("response", schema)
            );

            builder.add_rule("root", "( " + analysis + " \"<|start|>assistant\" )? " + final);
        });
    }

    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            // tool calls can appear in commentary or analysis channels
            auto channel = builder.add_rule("channel", "\"<|channel|>\" ( \"commentary\" | \"analysis\" )");

            std::vector<std::string> tool_rules_recipient_in_role;
            std::vector<std::string> tool_rules_recipient_in_channel;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);

                tool_rules_recipient_in_role.push_back(
                    builder.add_rule(name + "-call",
                        "\"" + name + "\"" + channel + " \" <|constrain|>json\"? \"<|message|>\" " +
                        builder.add_schema(name + "-args", parameters)
                    )
                );

                tool_rules_recipient_in_channel.push_back(
                    builder.add_rule(name + "-call",
                        "\"" + name + "\"" + " \" <|constrain|>json\"? \"<|message|>\" " +
                        builder.add_schema(name + "-args", parameters)
                    )
                );
            });

            auto recipient_in_channel = builder.add_rule("recipient_in_channel",
                channel + " \" to=functions.\" ( " +
                string_join(tool_rules_recipient_in_channel, " | ") + " )"
            );

            if (data.grammar_lazy) {
                auto recipient_in_role = builder.add_rule("recipient_in_role",
                    "\"<|start|>assistant\"? \" to=functions.\" ( " +
                    string_join(tool_rules_recipient_in_role, " | ") + " )"
                );

                builder.add_rule("root", recipient_in_role + " | " + recipient_in_channel);
            } else {
                auto not_end = builder.add_rule("not-end",
                    "[^<] | \"<\" [^|] | \"<|\" [^e] | \"<|e\" [^n] | \"<|en\" [^d] | \"<|end\" [^|] | \"<|end|\" [^>]");
                auto analysis = builder.add_rule("analysis",
                    "\"<|channel|>analysis<|message|>\" ( " + not_end + " )* \"<|end|>\"");
                auto commentary = builder.add_rule("commentary",
                    "\"<|channel|>commentary<|message|>\" ( " + not_end + " )* \"<|end|>\"");

                auto recipient_in_role = builder.add_rule("recipient_in_role",
                    "\" to=functions.\" ( " + string_join(tool_rules_recipient_in_role, " | ") + " )"
                );

                builder.add_rule("root",
                    "( " + analysis + " \"<|start|>assistant\" )? " +
                    "( " + commentary + " \"<|start|>assistant\" )? " +
                    "( " + recipient_in_role + " | " + recipient_in_channel + " )"
                );
            }

            // Trigger on tool calls that appear in the commentary channel
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                "<\\|channel\\|>(commentary|analysis) to"
            });

            // Trigger tool calls that appear in the role section, either at the
            // start or in the middle.
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                "^ to"
            });

            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                "<\\|start\\|>assistant to"
            });
        });
    }

    return data;
}

static common_chat_params common_chat_params_init_glm_4_5(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.grammar_lazy = inputs.tools.is_array() && !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    std::string prompt = apply(tmpl, inputs);

    // match the existing trimming behavior
    if (inputs.add_bos && string_starts_with(prompt, tmpl.bos_token())) {
        prompt.erase(0, tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(prompt, tmpl.eos_token())) {
        prompt.erase(prompt.size() - tmpl.eos_token().size());
    }
    if (string_ends_with(prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    // add GLM preserved tokens
    data.preserved_tokens = {
        "<|endoftext|>",
        "[MASK]",
        "[gMASK]",
        "[sMASK]",
        "<sop>",
        "<eop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
        "<|begin_of_audio|>",
        "<|end_of_audio|>",
        "<|begin_of_transcription|>",
        "<|end_of_transcription|>",
        "<|code_prefix|>",
        "<|code_middle|>",
        "<|code_suffix|>",
        "/nothink",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>"
    };

    // extra GLM 4.5 stop word
    data.additional_stops.insert(data.additional_stops.end(), {
        "<|user|>",
        "<|observation|>"
    });

    // build grammar for tool call
    static const xml_tool_call_format form {
        /* form.scope_start = */ "",
        /* form.tool_start  = */ "\n<tool_call>",
        /* form.tool_sep    = */ "\n",
        /* form.key_start   = */ "<arg_key>",
        /* form.key_val_sep = */ "</arg_key>\n<arg_value>",
        /* form.val_end     = */ "</arg_value>\n",
        /* form.tool_end    = */ "</tool_call>\n",
        /* form.scope_end   = */ "",
    };
    build_grammar_xml_tool_call(data, inputs.tools, form);

    data.prompt = prompt;
    data.format = COMMON_CHAT_FORMAT_GLM_4_5;
    return data;
}

static common_chat_params common_chat_params_init_firefunction_v2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    LOG_DBG("%s\n", __func__);
    common_chat_params data;
    const std::optional<json> tools_override = json();
    const std::optional<json> additional_context = json {
        {"datetime", format_time(inputs.now, "%b %d %Y %H:%M:%S GMT")},
        {"functions", json(inputs.tools.empty() ? "" : inputs.tools.dump(2))},
    };
    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, tools_override, additional_context);
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    {"type", "object"},
                    {"properties", {
                        {"name", {
                            {"type", "string"},
                            {"const", function.at("name")},
                        }},
                        {"arguments", function.at("parameters")},
                    }},
                    {"required", json::array({"name", "arguments", "id"})},
                });
            });
            auto schema = json {
                {"type", "array"},
                {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
                {"minItems", 1},
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }
            builder.add_rule("root", "\" functools\"? " + builder.add_schema("tool_calls", schema));
        });
        data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, " functools["});
        data.preserved_tokens = {
            " functools[",
        };
        data.format = COMMON_CHAT_FORMAT_FIREFUNCTION_V2;
    } else {
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    }
    return data;
}

static common_chat_params common_chat_params_init_functionary_v3_2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    // >>>all\nlet's call functions>>>fn1\n{"arg1": 1...}\n>>>fn2\n{"arg1": 1...}...
    // Using ">>>f1\n", ">>>f2\n"... as trigger words for the grammar
    // If the function is python, we also allow raw python code (if the line after `python\n` doesn't start w/ opening `{`), which the model seems to prefer for multiline code.
    common_chat_params data;
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2;
    if (inputs.tools.is_array() && !inputs.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> first_tool_rules;
            std::vector<std::string> subsequent_tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                std::string args_pattern = "[\\s\\S]*";
                auto args_rule = builder.add_schema(name + "-args", parameters);
                if (name == "python") {
                    args_rule = builder.add_rule(name + "-maybe-raw-args", args_rule + " | [^{] .*");
                } else {
                    args_pattern = "\\{" + args_pattern;
                }
                auto call_rule = builder.add_rule(name + "-call", "\"" + name + "\\n\" " + args_rule);
                first_tool_rules.push_back(call_rule);
                if (inputs.parallel_tool_calls) {
                    subsequent_tool_rules.push_back(builder.add_rule(name + "-call2", "\">>>\" " + call_rule));
                }
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                    "((?:[\\s\\S]+?>>>)?" + regex_escape(name) + "\n)" + args_pattern,
                });
            });
            data.preserved_tokens = {
                "<|end_header_id|>",
            };
            auto first_rule = first_tool_rules.empty() ? "" : builder.add_rule("first_tool_call", string_join(first_tool_rules, " | ")) + " space";
            if (inputs.parallel_tool_calls) {
                auto subsequent_rule = builder.add_rule("subsequent_tool_call", string_join(subsequent_tool_rules, " | ")) + " space";
                builder.add_rule("root", first_rule + " (" + subsequent_rule + ")*");
            } else {
                builder.add_rule("root", first_rule);
            }

        });
    }
    return data;
}

static common_chat_params common_chat_params_init_functionary_v3_1_llama_3_1(const common_chat_template & tmpl, const struct templates_params & inputs) {
    // https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt
    common_chat_params data;

    if (!inputs.tools.is_null()) {
        std::string python_code_argument_name;
        auto has_raw_python = false;

        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                const auto & parameters = function.at("parameters");
                std::string name = function.at("name");
                if (name == "python" || name == "ipython") {
                    if (!parameters.contains("type")) {
                        throw std::runtime_error("Missing type in python tool");
                    }
                    has_raw_python = true;
                    const auto & type = parameters.at("type");
                    if (type == "object") {
                        auto properties = parameters.at("properties");
                        for (auto it = properties.begin(); it != properties.end(); ++it) {
                            if (it.value().at("type") == "string") {
                                if (!python_code_argument_name.empty()) {
                                    throw std::runtime_error("Multiple string arguments found in python tool");
                                }
                                python_code_argument_name = it.key();
                            }
                        }
                        if (python_code_argument_name.empty()) {
                            throw std::runtime_error("No string argument found in python tool");
                        }
                    } else if (type != "string") {
                        throw std::runtime_error("Invalid type in python tool: " + type.dump());
                    }
                }
                tool_rules.push_back(builder.add_rule(name + "-call", "\"<function=" + name + ">\" " + builder.add_schema(name + "-args", parameters) + " \"</function>\" space"));
            });
            if (has_raw_python) {
                tool_rules.push_back(builder.add_rule("python-call", "\"<|python_tag|>\" .*"));
                data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|python_tag|>"});
                data.preserved_tokens.push_back("<|python_tag|>");
            }
            auto tool_call = builder.add_rule("tool_call", string_join(tool_rules, " | ")) + " space";
            builder.add_rule("root", inputs.parallel_tool_calls ? "(" + tool_call + ")+" : tool_call);
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<function="});
        });
        data.format = COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1;
    } else {
        data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    }

    data.prompt = apply(tmpl, inputs);
    // TODO: if (has_raw_python)
    return data;
}

static common_chat_params common_chat_params_init_hermes_2_pro(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    json extra_context = json {
        {"enable_thinking", inputs.enable_thinking},
    };
    extra_context.update(inputs.extra_context);

    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, /* tools_override= */ std::nullopt, extra_context);
    data.format = COMMON_CHAT_FORMAT_HERMES_2_PRO;
    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!extra_context["enable_thinking"]) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    if (!inputs.tools.is_null()) {
        // (content)?(<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>)*
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            std::vector<std::string> tool_call_alts;
            std::vector<std::string> escaped_names;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_schema(name + "-call", {
                    {"type", "object"},
                    {"properties", json {
                        {"name", json {{"const", name}}},
                        {"arguments", parameters},
                    }},
                    {"required", json::array({"name", "arguments"})},
                }));
                tool_call_alts.push_back(builder.add_rule(
                    name + "-function-tag",
                    "\"<function\" ( \"=" + name + "\" | \" name=\\\"" + name + "\\\"\" ) \">\" space " +
                    builder.add_schema(name + "-args", parameters) + " "
                    "\"</function>\" space"));

                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
                    "<function=" + name + ">",
                });
                auto escaped_name = regex_escape(name);
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                    "<function\\s+name\\s*=\\s*\"" + escaped_name + "\"",
                });
                escaped_names.push_back(escaped_name);
            });
            auto any_tool_call = builder.add_rule("any_tool_call", "( " + string_join(tool_rules, " | ") + " ) space");
            std::vector<std::string> alt_tags {
                any_tool_call,
                "\"<tool_call>\" space "     + any_tool_call + " \"</tool_call>\"",
                // The rest is just to accommodate common "good bad" outputs.
                "\"<function_call>\" space " + any_tool_call + " \"</function_call>\"",
                "\"<response>\"  space "     + any_tool_call + " \"</response>\"",
                "\"<tools>\"     space "     + any_tool_call + " \"</tools>\"",
                "\"<json>\"      space "     + any_tool_call + " \"</json>\"",
                "\"<xml>\"      space "     + any_tool_call + " \"</xml>\"",
                "\"<JSON>\"      space "     + any_tool_call + " \"</JSON>\"",
            };
            auto wrappable_tool_call = builder.add_rule("wrappable_tool_call", "( " + string_join(alt_tags, " | ") + " ) space");
            tool_call_alts.push_back(wrappable_tool_call);
            tool_call_alts.push_back(
                "( \"```\\n\" | \"```json\\n\" | \"```xml\\n\" ) space " + wrappable_tool_call + " space \"```\" space ");
            auto tool_call = builder.add_rule("tool_call", string_join(tool_call_alts, " | "));
            builder.add_rule("root",
                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                (inputs.parallel_tool_calls ? "(" + tool_call + ")+" : tool_call));
            // Trigger on some common known "good bad" outputs (only from the start and with a json that's about a specific argument name to avoid false positives)
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                // If thinking_forced_open, then we capture the </think> tag in the grammar,
                // (important for required tool choice) and in the trigger's first capture (decides what is sent to the grammar)
                std::string(data.thinking_forced_open ? "[\\s\\S]*?(</think>\\s*)" : "(?:<think>[\\s\\S]*?</think>\\s*)?") + (
                    "\\s*("
                    "(?:<tool_call>"
                    "|<function"
                    "|(?:```(?:json|xml)?\n\\s*)?(?:<function_call>|<tools>|<xml><json>|<response>)?"
                    "\\s*\\{\\s*\"name\"\\s*:\\s*\"(?:" + string_join(escaped_names, "|") + ")\""
                    ")"
                    ")[\\s\\S]*"
                ),
            });
            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<function",
                "<tools>",
                "</tools>",
                "<response>",
                "</response>",
                "<function_call>",
                "</function_call>",
                "<json>",
                "</json>",
                "<JSON>",
                "</JSON>",
                "```",
                "```json",
                "```xml",
            };
        });
    }

    return data;
}

static common_chat_params common_chat_params_init_granite(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    // Pass thinking context for Granite template
    json additional_context = {
        {"thinking", inputs.enable_thinking},
    };

    data.prompt = apply(tmpl, inputs, /* messages_override= */ std::nullopt, /* tools_override= */ std::nullopt, additional_context);
    data.format = COMMON_CHAT_FORMAT_GRANITE;

    if (string_ends_with(data.prompt, "<think>\n") || string_ends_with(data.prompt, "<think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    if (!inputs.tools.is_null()) {
        // Granite uses <|tool_call|> followed by JSON list
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_rule(name + "-call", builder.add_schema(name +
"-args", {
                    {"type", "object"},
                    {"properties", {
                        {"name", {{"const", name}}},
                        {"arguments", parameters},
                    }},
                    {"required", json::array({"name", "arguments"})},
                })));
            });

            auto tool_call = builder.add_rule("tool_call", string_join(tool_rules, " | "));
            auto tool_list = builder.add_rule("tool_list", "\"[\" space " + tool_call + " (\",\" space " + tool_call + ")* space \"]\"");

            if (data.thinking_forced_open) {
                builder.add_rule("root", "\"</think>\" space \"<response>\" space [^<]* \"</response>\" space \"<|tool_call|>\" space " + tool_list);
            } else {
                builder.add_rule("root", "\"<|tool_call|>\" space " + tool_list);
            }

            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
                "<|tool_call|>"
            });

            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<response>",
                "</response>",
                "<|tool_call|>",
            };
        });
    } else {
        // Handle thinking tags for non-tool responses
        if (data.thinking_forced_open && inputs.enable_thinking) {
            data.grammar_lazy = false;
            data.grammar = build_grammar([&](const common_grammar_builder & builder) {
                builder.add_rule("root", "\"</think>\" space \"<response>\" space .* \"</response>\" space");
            });
            data.preserved_tokens = {
                "<think>",
                "</think>",
                "<response>",
                "</response>",
            };
        }
    }

    return data;
}

static common_chat_params common_chat_params_init_without_tools(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    data.grammar_lazy = false;
    if (!inputs.json_schema.is_null()) {
        if (!inputs.grammar.empty()) {
            throw std::runtime_error("Either \"json_schema\" or \"grammar\" can be specified, but not both");
        }
        data.grammar = json_schema_to_grammar(inputs.json_schema);
    } else {
        data.grammar = inputs.grammar;
    }
    return data;
}

static common_chat_params common_chat_params_init_seed_oss(
    const common_chat_template         & tmpl,
    templates_params                   & params,
    const common_chat_templates_inputs & inputs)
{
    common_chat_params data;
    data.prompt = apply(tmpl, params);
    data.format = COMMON_CHAT_FORMAT_SEED_OSS;
    if (string_ends_with(data.prompt, "<seed:think>")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</seed:think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    if (params.tools.is_array() && !params.tools.empty()) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(params.tools, [&](const json & tool) {
                const auto & function   = tool.at("function");
                std::string  name       = function.at("name");
                auto         parameters = function.at("parameters");
                builder.resolve_refs(parameters);

                // Create rule for Seed-OSS function call format
                std::string param_rules;
                if (parameters.contains("properties")) {
                    for (const auto & [key, value] : parameters.at("properties").items()) {
                        param_rules += "\"<parameter=" + key + ">\"" + builder.add_schema(name + "-arg-" + key, value) +
                                       "\"</parameter>\"";
                    }
                }

                tool_rules.push_back(builder.add_rule(name + "-call",
                                                      "\"<seed:tool_call>\" space \"<function=" + name + ">\" space " +
                                                          param_rules +
                                                          " \"</function>\" space \"</seed:tool_call>\""));
            });

            data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<seed:tool_call>" });

            data.preserved_tokens = {
                "<seed:think>", "</seed:think>", "<seed:tool_call>", "</seed:tool_call>",
                "<function=",   "</function>",   "<parameter=",      "</parameter>",
            };

            builder.add_rule("root", string_join(tool_rules, " | "));
        });
    }
    return data;
}

static common_chat_params common_chat_templates_apply_jinja(
    const struct common_chat_templates        * tmpls,
    const struct common_chat_templates_inputs & inputs)
{
    templates_params params;
    params.tools = common_chat_tools_to_json_oaicompat<json>(inputs.tools);
    const auto & tmpl = params.tools.is_array() && tmpls->template_tool_use
        ? *tmpls->template_tool_use
        : *tmpls->template_default;
    const auto & src = tmpl.source();
    const auto & caps = tmpl.original_caps();
    params.messages = common_chat_msgs_to_json_oaicompat<json>(inputs.messages, /* concat_text= */ !tmpl.original_caps().requires_typed_content);
    params.add_generation_prompt = inputs.add_generation_prompt;
    params.tool_choice = inputs.tool_choice;
    params.reasoning_format = inputs.reasoning_format;
    params.enable_thinking = inputs.enable_thinking;
    params.grammar = inputs.grammar;
    params.now = inputs.now;
    params.add_bos = tmpls->add_bos;
    params.add_eos = tmpls->add_eos;

    params.extra_context = json::object();
    for (auto el : inputs.chat_template_kwargs) {
        params.extra_context[el.first] = json::parse(el.second);
    }

    if (!inputs.json_schema.empty()) {
        params.json_schema = json::parse(inputs.json_schema);
    }

    if (inputs.parallel_tool_calls && !tmpl.original_caps().supports_parallel_tool_calls) {
        LOG_DBG("Disabling parallel_tool_calls because the template does not support it\n");
        params.parallel_tool_calls = false;
    } else {
        params.parallel_tool_calls = inputs.parallel_tool_calls;
    }

    if (params.tools.is_array()) {
        if (params.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !params.grammar.empty()) {
            throw std::runtime_error("Cannot specify grammar with tools");
        }
        if (caps.supports_tool_calls && !caps.supports_tools) {
            LOG_WRN("Template supports tool calls but does not natively describe tools. The fallback behaviour used may produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
        }
    }

    // DeepSeek V3.1: detect based on specific patterns in the template
    if (src.find("message['prefix'] is defined and message['prefix'] and thinking") != std::string::npos &&
        params.json_schema.is_null()) {
        return common_chat_params_init_deepseek_v3_1(tmpl, params);
    }

    // DeepSeek R1: use handler in all cases except json schema (thinking / tools).
    if (src.find("<｜tool▁calls▁begin｜>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_deepseek_r1(tmpl, params);
    }

    // Command R7B: : use handler in all cases except json schema (thinking / tools).
    if (src.find("<|END_THINKING|><|START_ACTION|>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_command_r7b(tmpl, params);
    }

    // Granite (IBM) - detects thinking / tools support
    if (src.find("elif thinking") != std::string::npos && src.find("<|tool_call|>") != std::string::npos) {
        return common_chat_params_init_granite(tmpl, params);
    }

    // GLM 4.5: detect by <arg_key> and <arg_value> tags (check before Hermes since both use <tool_call>)
    if (src.find("[gMASK]<sop>") != std::string::npos &&
        src.find("<arg_key>") != std::string::npos &&
        src.find("<arg_value>") != std::string::npos &&
        params.json_schema.is_null()) {
        return common_chat_params_init_glm_4_5(tmpl, params);
    }

    // Qwen3-Coder XML format detection (must come before Hermes 2 Pro)
    // Detect via explicit XML markers unique to Qwen3-Coder to avoid false positives in other templates.
    // Require presence of <tool_call>, <function=...>, and <parameter=...> blocks.
    if (src.find("<tool_call>") != std::string::npos &&
        src.find("<function>") != std::string::npos &&
        src.find("<function=") != std::string::npos &&
        src.find("<parameters>") != std::string::npos &&
        src.find("<parameter=") != std::string::npos) {
        // Nemotron 3 Nano 30B A3B
        if (src.find("<think>") != std::string::npos) {
            return common_chat_params_init_nemotron_v3(tmpl, params);
        }
        return common_chat_params_init_qwen3_coder_xml(tmpl, params);
    }

    // Xiaomi MiMo format detection (must come before Hermes 2 Pro)
    if (src.find("<tools>") != std::string::npos &&
        src.find("# Tools") != std::string::npos &&
        src.find("</tools>") != std::string::npos &&
        src.find("<tool_calls>") != std::string::npos &&
        src.find("</tool_calls>") != std::string::npos &&
        src.find("<tool_response>") != std::string::npos) {
        return common_chat_params_init_xiaomi_mimo(tmpl, params);
    }

    // Hermes 2/3 Pro, Qwen 2.5 Instruct (w/ tools)
    if (src.find("<tool_call>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_hermes_2_pro(tmpl, params);
    }

    // GPT-OSS
    if (src.find("<|channel|>") != std::string::npos) {
        return common_chat_params_init_gpt_oss(tmpl, params);
    }

    // Seed-OSS
    if (src.find("<seed:think>") != std::string::npos) {
        return common_chat_params_init_seed_oss(tmpl, params, inputs);
    }

    // Nemotron v2
    if (src.find("<SPECIAL_10>") != std::string::npos) {
        return common_chat_params_init_nemotron_v2(tmpl, params);
    }

    // Apertus format detection
    if (src.find("<|system_start|>") != std::string::npos && src.find("<|tools_prefix|>") != std::string::npos) {
        return common_chat_params_init_apertus(tmpl, params);
    }

    // LFM2 (w/ tools)
    if (src.find("List of tools: <|tool_list_start|>[") != std::string::npos &&
        src.find("]<|tool_list_end|>") != std::string::npos) {
        return common_chat_params_init_lfm2(tmpl, params);
    }

    // MiniMax-M2 format detection
    if (src.find("]~!b[") != std::string::npos && src.find("]~b]") != std::string::npos) {
        return common_chat_params_init_minimax_m2(tmpl, params);
    }

    // Kimi K2 format detection
    if (src.find("<|im_system|>tool_declare<|im_middle|>") != std::string::npos &&
        src.find("<|tool_calls_section_begin|>") != std::string::npos &&
        src.find("## Return of") != std::string::npos) {
        return common_chat_params_init_kimi_k2(tmpl, params);
    }

    // Apriel 1.5 format detection
    if (src.find("<thinking>") != std::string::npos &&
        src.find("</thinking>") != std::string::npos &&
        src.find("<available_tools>") != std::string::npos &&
        src.find("<|assistant|>") != std::string::npos &&
        src.find("<|tool_result|>") != std::string::npos &&
        src.find("<tool_calls>[") != std::string::npos &&
        src.find("]</tool_calls>") != std::string::npos) {
        return common_chat_params_init_apriel_1_5(tmpl, params);
    }

    // Use generic handler when mixing tools + JSON schema.
    // TODO: support that mix in handlers below.
    if ((params.tools.is_array() && params.json_schema.is_object())) {
        return common_chat_params_init_generic(tmpl, params);
    }

    // Functionary prepends "all\n" to plain content outputs, so we use its handler in all cases.
    if (src.find(">>>all") != std::string::npos) {
        return common_chat_params_init_functionary_v3_2(tmpl, params);
    }

    // Firefunction v2 requires datetime and functions in the context even w/o tools, so we also use its handler in all cases.
    if (src.find(" functools[") != std::string::npos) {
        return common_chat_params_init_firefunction_v2(tmpl, params);
    }

    // Functionary v3.1 (w/ tools)
    if (src.find("<|start_header_id|>") != std::string::npos
        && src.find("<function=") != std::string::npos) {
        return common_chat_params_init_functionary_v3_1_llama_3_1(tmpl, params);
    }

    // Llama 3.1, 3.2, 3.3 (also requires date_string so using it even w/o tools)
    if (src.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos) {
        auto allow_python_tag_builtin_tools = src.find("<|python_tag|>") != std::string::npos;
        return common_chat_params_init_llama_3_x(tmpl, params, allow_python_tag_builtin_tools);
    }

    // Ministral/Mistral Large 3
    if (src.find("[SYSTEM_PROMPT]") != std::string::npos &&
        src.find("[TOOL_CALLS]") != std::string::npos &&
        src.find("[ARGS]") != std::string::npos) {
        return common_chat_params_init_ministral_3(tmpl, params);
    }

    if (src.find("[THINK]") != std::string::npos && src.find("[/THINK]") != std::string::npos) {
        return common_chat_params_init_magistral(tmpl, params);
    }

    // Plain handler (no tools)
    if (params.tools.is_null() || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
        return common_chat_params_init_without_tools(tmpl, params);
    }

    // Mistral Nemo (w/ tools)
    if (src.find("[TOOL_CALLS]") != std::string::npos) {
        return common_chat_params_init_mistral_nemo(tmpl, params);
    }

    // Generic fallback
    return common_chat_params_init_generic(tmpl, params);
}

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_params common_chat_templates_apply_legacy(
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs)
{
    size_t alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string> contents;

    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text") {
                LOG_WRN("Ignoring non-text content part: %s\n", part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";;
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const auto & msg = inputs.messages[i];
        const auto & content = contents[i];
        chat.push_back({msg.role.c_str(), content.c_str()});
        size_t msg_size = msg.role.size() + content.size();
        alloc_size += msg_size + (msg_size / 4); // == msg_size * 1.25 but avoiding float ops
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default->source();
    int32_t res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("this custom template is not supported, try using --jinja");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(), buf.size());
    }

    // for safety, we check the result again
    if (res < 0 || (size_t) res > buf.size()) {
        throw std::runtime_error("failed to apply chat template, try using --jinja");
    }

    common_chat_params params;
    params.prompt = std::string(buf.data(), res);
    if (!inputs.json_schema.empty()) {
        params.grammar = json_schema_to_grammar(json::parse(inputs.json_schema));
    } else {
        params.grammar = inputs.grammar;
    }
    return params;
}

common_chat_params common_chat_templates_apply(
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs)
{
    GGML_ASSERT(tmpls != nullptr);
    return inputs.use_jinja
        ? common_chat_templates_apply_jinja(tmpls, inputs)
        : common_chat_templates_apply_legacy(tmpls, inputs);
}
