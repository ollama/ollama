#include "chat.h"
#include "chat-parser.h"
#include "common.h"
#include "json-partial.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "regex-partial.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <cstdio>
#include <exception>
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

std::vector<common_chat_msg_diff> common_chat_msg_diff::compute_diffs(const common_chat_msg & previous_msg, const common_chat_msg & new_msg) {
    std::vector<common_chat_msg_diff> diffs;
    if (previous_msg.reasoning_content != new_msg.reasoning_content) {
        auto & diff = diffs.emplace_back();
        diff.reasoning_content_delta = string_diff(previous_msg.reasoning_content, new_msg.reasoning_content);
    }
    if (previous_msg.content != new_msg.content) {
        auto & diff = diffs.emplace_back();
        diff.content_delta = string_diff(previous_msg.content, new_msg.content);
    }

    if (new_msg.tool_calls.size() < previous_msg.tool_calls.size()) {
        throw std::runtime_error("Invalid diff: now finding less tool calls!");
    }

    if (!previous_msg.tool_calls.empty()) {
        auto idx = previous_msg.tool_calls.size() - 1;
        const auto & pref = previous_msg.tool_calls[idx];
        const auto & newf = new_msg.tool_calls[idx];
        if (pref.name != newf.name) {
            throw std::runtime_error("Invalid diff: tool call mismatch!");
        }
        auto args_diff = string_diff(pref.arguments, newf.arguments);
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
    for (size_t idx = previous_msg.tool_calls.size(); idx < new_msg.tool_calls.size(); ++idx) {
        auto & diff = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = new_msg.tool_calls[idx];
    }
    return diffs;
}

typedef minja::chat_template common_chat_template;

struct common_chat_templates {
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
    bool stream;
    std::string grammar;
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    json extra_context;
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
    throw std::runtime_error("Invalid tool_choice: " + tool_choice);
}

template <>
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {

        if (!messages.is_array()) {
            throw std::runtime_error("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::runtime_error("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::runtime_error("Missing 'role' in message: " + message.dump());
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
                            throw std::runtime_error("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text") {
                            throw std::runtime_error("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::runtime_error("Invalid 'content' type: expected string or array, got " + content.dump() + " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    common_chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::runtime_error("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::runtime_error("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::runtime_error("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::runtime_error("Missing tool call name: " + tool_call.dump());
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
                throw std::runtime_error("Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & https://github.com/ggml-org/llama.cpp/issues/12279)");
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
                throw std::runtime_error("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::runtime_error("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::runtime_error("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::runtime_error("Missing tool function: " + tool.dump());
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

std::string common_chat_format_example(const struct common_chat_templates * tmpls, bool use_jinja) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
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
    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
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
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
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
        case COMMON_CHAT_FORMAT_LLAMA_3_X: return "Llama 3.x";
        case COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS: return "Llama 3.x with builtin tools";
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1: return "DeepSeek R1";
        case COMMON_CHAT_FORMAT_FIREFUNCTION_V2: return "FireFunction v2";
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2: return "Functionary v3.2";
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1: return "Functionary v3.1 Llama 3.1";
        case COMMON_CHAT_FORMAT_HERMES_2_PRO: return "Hermes 2 Pro";
        case COMMON_CHAT_FORMAT_COMMAND_R7B: return "Command R7B";
        default:
            throw std::runtime_error("Unknown chat format");
    }
}

const char * common_reasoning_format_name(common_reasoning_format format) {
    switch (format) {
        case COMMON_REASONING_FORMAT_NONE:     return "none";
        case COMMON_REASONING_FORMAT_DEEPSEEK: return "deepseek";
        case COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY: return "deepseek-legacy";
        default:
            throw std::runtime_error("Unknown reasoning format");
    }
}

static std::string wrap_code_as_arguments(common_chat_msg_parser & builder, const std::string & code) {
    std::string arguments;
    if (builder.is_partial()) {
        arguments = (json {{"code", code + builder.healing_marker()}}).dump();
        auto idx = arguments.find(builder.healing_marker());
        if (idx != std::string::npos) {
            arguments.resize(idx);
        }
    } else {
        arguments = (json {{"code", code}}).dump();
    }
    return arguments;
}

/**
 * Takes a prefix regex that must have 1 group to capture the function name, a closing suffix, and expects json parameters in between.
 * Aggregates the prefix, suffix and in-between text into the content.
 */
static void parse_json_tool_calls(
    common_chat_msg_parser & builder,
    const std::optional<common_regex> & block_open,
    const std::optional<common_regex> & function_regex_start_only,
    const std::optional<common_regex> & function_regex,
    const common_regex & close_regex,
    const std::optional<common_regex> & block_close,
    bool allow_raw_python = false,
    const std::function<std::string(const common_chat_msg_parser::find_regex_result & fres)> & get_function_name = nullptr) {

    auto parse_tool_calls = [&]() {
        size_t from = std::string::npos;
        auto first = true;
        while (true) {
            auto res = function_regex_start_only && first
                ? builder.try_consume_regex(*function_regex_start_only)
                : function_regex
                    ? builder.try_find_regex(*function_regex, from)
                    : std::nullopt;
            if (res) {
                std::string name;
                if (get_function_name) {
                    name = get_function_name(*res);
                } else {
                    GGML_ASSERT(res->groups.size() == 2);
                    name = builder.str(res->groups[1]);
                }
                first = false;
                if (name.empty()) {
                    // get_function_name signalled us that we should skip this match and treat it as content.
                    from = res->groups[0].begin + 1;
                    continue;
                }
                from = std::string::npos;

                auto maybe_raw_python = name == "python" && allow_raw_python;
                if (builder.input()[builder.pos()] == '{' || !maybe_raw_python) {
                    if (auto arguments = builder.try_consume_json_with_dumped_args({{}})) {
                        if (!builder.add_tool_call(name, "", arguments->value) || arguments->is_partial) {
                            throw common_chat_msg_partial_exception("incomplete tool call");
                        }
                        builder.consume_regex(close_regex);
                    }
                    continue;
                }
                if (maybe_raw_python) {
                    auto arguments = wrap_code_as_arguments(builder, builder.consume_rest());
                    if (!builder.add_tool_call(name, "", arguments)) {
                        throw common_chat_msg_partial_exception("incomplete tool call");
                    }
                    return;
                }
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
            break;
        }
        if (block_close) {
            builder.consume_regex(*block_close);
        }
        builder.consume_spaces();
        builder.add_content(builder.consume_rest());
    };
    if (block_open) {
        if (auto res = builder.try_find_regex(*block_open)) {
            parse_tool_calls();
        } else {
            builder.add_content(builder.consume_rest());
        }
    } else {
        parse_tool_calls();
    }
}

static void parse_prefixed_json_tool_call_array(common_chat_msg_parser & builder, const common_regex & prefix, size_t rstrip_prefix = 0) {
    static const std::vector<std::vector<std::string>> args_paths = {{"arguments"}};
    if (auto res = builder.try_find_regex(prefix)) {
        builder.move_back(rstrip_prefix);
        auto tool_calls = builder.consume_json_with_dumped_args(args_paths);
        if (!builder.add_tool_calls(tool_calls.value) || tool_calls.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool call array");
        }
    } else {
        builder.add_content(builder.consume_rest());
    }
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
    if (string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (string_ends_with(result, tmpl.eos_token())) {
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
static void common_chat_parse_generic(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
    static const std::vector<std::vector<std::string>> content_paths = {
        {"response"},
    };
    static const std::vector<std::vector<std::string>> args_paths = {
        {"tool_call", "arguments"},
        {"tool_calls", "arguments"},
    };
    auto data = builder.consume_json_with_dumped_args(args_paths, content_paths);
    if (data.value.contains("tool_calls")) {
        if (!builder.add_tool_calls(data.value.at("tool_calls")) || data.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool calls");
        }
    } else if (data.value.contains("tool_call")) {
        if (!builder.add_tool_call(data.value.at("tool_call")) || data.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool call");
        }
    } else if (data.value.contains("response")) {
        const auto & response = data.value.at("response");
        builder.add_content(response.is_string() ? response.template get<std::string>() : response.dump(2));
        if (data.is_partial) {
            throw common_chat_msg_partial_exception("incomplete response");
        }
    } else {
        throw common_chat_msg_partial_exception("Expected 'tool_call', 'tool_calls' or 'response' in JSON");
    }
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
static void common_chat_parse_mistral_nemo(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex prefix(regex_escape("[TOOL_CALLS]"));
    parse_prefixed_json_tool_call_array(builder, prefix);
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

static void common_chat_parse_command_r7b(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<|START_THINKING|>", "<|END_THINKING|>");

    static const common_regex start_action_regex("<\\|START_ACTION\\|>");
    static const common_regex end_action_regex("<\\|END_ACTION\\|>");
    static const common_regex start_response_regex("<\\|START_RESPONSE\\|>");
    static const common_regex end_response_regex("<\\|END_RESPONSE\\|>");

    if (auto res = builder.try_find_regex(start_action_regex)) {
        // If we didn't extract thoughts, prelude includes them.
        auto tool_calls = builder.consume_json_with_dumped_args({{"parameters"}});
        for (const auto & tool_call : tool_calls.value) {
            std::string name = tool_call.contains("tool_name") ? tool_call.at("tool_name") : "";
            std::string id = tool_call.contains("tool_call_id") ? tool_call.at("tool_call_id") : "";
            std::string arguments = tool_call.contains("parameters") ? tool_call.at("parameters") : "";
            if (!builder.add_tool_call(name, id, arguments) || tool_calls.is_partial) {
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
        }
        if (tool_calls.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool call");
        }
        builder.consume_regex(end_action_regex);
    } else if (auto res = builder.try_find_regex(start_response_regex)) {
        if (!builder.try_find_regex(end_response_regex)) {
            builder.add_content(builder.consume_rest());
            throw common_chat_msg_partial_exception(end_response_regex.str());
        }
    } else {
        builder.add_content(builder.consume_rest());
    }
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
static void common_chat_parse_llama_3_1(common_chat_msg_parser & builder, bool with_builtin_tools = false) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex function_regex(
        "\\s*\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"parameters\"\\s*: ");
    static const common_regex close_regex("\\}\\s*");

    static const common_regex function_name_regex("\\s*(\\w+)\\s*\\.\\s*call\\(");
    static const common_regex arg_name_regex("\\s*(\\w+)\\s*=\\s*");

    if (with_builtin_tools) {
        static const common_regex builtin_call_regex("<\\|python_tag\\|>");
        if (auto res = builder.try_find_regex(builtin_call_regex)) {
            auto fun_res = builder.consume_regex(function_name_regex);
            auto function_name = builder.str(fun_res.groups[1]);

            common_healing_marker healing_marker;
            json args = json::object();
            while (true) {
                if (auto arg_res = builder.try_consume_regex(arg_name_regex)) {
                    auto arg_name = builder.str(arg_res->groups[1]);
                    auto partial = builder.consume_json();
                    args[arg_name] = partial.json;
                    healing_marker.marker = partial.healing_marker.marker;
                    healing_marker.json_dump_marker = partial.healing_marker.json_dump_marker;
                    builder.consume_spaces();
                    if (!builder.try_consume_literal(",")) {
                        break;
                    }
                } else {
                    break;
                }
            }
            builder.consume_literal(")");
            builder.consume_spaces();

            auto arguments = args.dump();
            if (!builder.add_tool_call(function_name, "", arguments)) {
                throw common_chat_msg_partial_exception("Incomplete tool call");
            }
            return;
        }
    }
    parse_json_tool_calls(
        builder,
        /* block_open= */ std::nullopt,
        /* function_regex_start_only= */ function_regex,
        /* function_regex= */ std::nullopt,
        close_regex,
        std::nullopt);

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
static void common_chat_parse_deepseek_r1(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<think>", "</think>");
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex tool_calls_begin("(?:<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)");
    static const common_regex tool_calls_end("<｜tool▁calls▁end｜>");
    static const common_regex function_regex("(?:<｜tool▁call▁begin｜>)?function<｜tool▁sep｜>([^\n]+)\n```json\n");
    static const common_regex close_regex("```[\\s\\r\\n]*<｜tool▁call▁end｜>");

    parse_json_tool_calls(
        builder,
        /* block_open= */ tool_calls_begin,
        /* function_regex_start_only= */ std::nullopt,
        function_regex,
        close_regex,
        tool_calls_end);
}

static common_chat_params common_chat_params_init_firefunction_v2(const common_chat_template & tmpl, const struct templates_params & inputs) {
    LOG_DBG("%s\n", __func__);
    common_chat_params data;
    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, /* tools_override= */ json(), json {
        {"datetime", format_time(inputs.now, "%b %d %Y %H:%M:%S GMT")},
        {"functions", json(inputs.tools.empty() ? "" : inputs.tools.dump(2))},
    });
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
static void common_chat_parse_firefunction_v2(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
    static const common_regex prefix(regex_escape(" functools["));
    parse_prefixed_json_tool_call_array(builder, prefix, /* rstrip_prefix= */ 1);
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
static void common_chat_parse_functionary_v3_2(common_chat_msg_parser & builder) {
    static const common_regex function_regex_start_only(R"((\w+\n\{|python\n|all\n))");
    static const common_regex function_regex(R"(>>>(\w+\n\{|python\n|all\n))");
    static const common_regex close_regex(R"(\s*)");

    parse_json_tool_calls(
        builder,
        std::nullopt,
        function_regex_start_only,
        function_regex,
        close_regex,
        std::nullopt,
        /* allow_raw_python= */ true,
        /* get_function_name= */ [&](const auto & res) -> std::string {
            auto at_start = res.groups[0].begin == 0;
            auto name = builder.str(res.groups[1]);
            if (!name.empty() && name.back() == '{') {
                // Unconsume the opening brace '{' to ensure the JSON parsing goes well.
                builder.move_back(1);
            }
            auto idx = name.find_last_not_of("\n{");
            name = name.substr(0, idx + 1);
            if (at_start && name == "all") {
                return "";
            }
            return name;
        });
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
static void common_chat_parse_functionary_v3_1_llama_3_1(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
    // This version of Functionary still supports the llama 3.1 tool call format for the python tool.
    static const common_regex python_tag_regex(regex_escape("<|python_tag|>"));

    static const common_regex function_regex(R"(<function=(\w+)>)");
    static const common_regex close_regex(R"(</function>)");

    parse_json_tool_calls(
        builder,
        /* block_open= */ std::nullopt,
        /* function_regex_start_only= */ std::nullopt,
        function_regex,
        close_regex,
        std::nullopt);

    if (auto res = builder.try_find_regex(python_tag_regex)) {
        auto arguments = wrap_code_as_arguments(builder, builder.consume_rest());
        builder.add_tool_call("python", "", arguments);
        return;
    }
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
                    "(\\s*"
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
static void common_chat_parse_hermes_2_pro(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<think>", "</think>");
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex open_regex(
        "(?:"
            "(```(?:xml|json)?\\n\\s*)?" // match 1 (block_start)
            "("                          // match 2 (open_tag)
                "<tool_call>"
                "|<function_call>"
                "|<tool>"
                "|<tools>"
                "|<response>"
                "|<json>"
                "|<xml>"
                "|<JSON>"
            ")?"
            "(\\s*\\{\\s*\"name\")" // match 3 (named tool call)
        ")"
        "|<function=([^>]+)>"            // match 4 (function name)
        "|<function name=\"([^\"]+)\">"  // match 5 (function name again)
    );

    if (auto res = builder.try_find_regex(open_regex)) {
        const auto & block_start = res->groups[1];
        std::string block_end = block_start.empty() ? "" : "```";

        const auto & open_tag = res->groups[2];
        std::string close_tag;

        if (!res->groups[3].empty()) {
            builder.move_to(res->groups[3].begin);
            close_tag = open_tag.empty() ? "" : "</" + builder.str(open_tag).substr(1);

            if (auto tool_call = builder.try_consume_json_with_dumped_args({{"arguments"}})) {
                if (!builder.add_tool_call(tool_call->value) || tool_call->is_partial) {
                    throw common_chat_msg_partial_exception("incomplete tool call");
                }
                builder.consume_spaces();
                builder.consume_literal(close_tag);
                builder.consume_spaces();
                if (!block_end.empty()) {
                    builder.consume_literal(block_end);
                    builder.consume_spaces();
                }
                builder.add_content(builder.consume_rest());
            } else {
                throw common_chat_msg_partial_exception("failed to parse tool call");
            }
        } else {
            auto function_name = builder.str(res->groups[4]);
            if (function_name.empty()) {
                function_name = builder.str(res->groups[5]);
            }
            GGML_ASSERT(!function_name.empty());

            close_tag = "</function>";

            if (auto arguments = builder.try_consume_json_with_dumped_args({{}})) {
                if (!builder.add_tool_call(function_name, "", arguments->value) || arguments->is_partial) {
                    throw common_chat_msg_partial_exception("incomplete tool call");
                }
                builder.consume_spaces();
                builder.consume_literal(close_tag);
                builder.consume_spaces();
                if (!block_end.empty()) {
                    builder.consume_literal(block_end);
                    builder.consume_spaces();
                }
            }
            builder.add_content(builder.consume_rest());
        }
    } else {
        builder.add_content(builder.consume_rest());
    }
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

static common_chat_params common_chat_templates_apply_jinja(
    const struct common_chat_templates * tmpls,
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
    params.enable_thinking = inputs.enable_thinking;
    params.grammar = inputs.grammar;
    params.now = inputs.now;

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

    // DeepSeek R1: use handler in all cases except json schema (thinking / tools).
    if (src.find("<｜tool▁calls▁begin｜>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_deepseek_r1(tmpl, params);
    }

    // Command R7B: : use handler in all cases except json schema (thinking / tools).
    if (src.find("<|END_THINKING|><|START_ACTION|>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_command_r7b(tmpl, params);
    }

    // Hermes 2/3 Pro, Qwen 2.5 Instruct (w/ tools)
    if (src.find("<tool_call>") != std::string::npos && params.json_schema.is_null()) {
        return common_chat_params_init_hermes_2_pro(tmpl, params);
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
    int alloc_size = 0;
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
        alloc_size += (msg.role.size() + content.size()) * 1.25;
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

static void common_chat_parse_content_only(common_chat_msg_parser & builder) {
    builder.add_content(builder.consume_rest());
}

static void common_chat_parse(common_chat_msg_parser & builder) {
    LOG_DBG("Parsing input with format %s: %s\n", common_chat_format_name(builder.syntax().format), builder.input().c_str());

    switch (builder.syntax().format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            common_chat_parse_content_only(builder);
            break;
        case COMMON_CHAT_FORMAT_GENERIC:
            common_chat_parse_generic(builder);
            break;
        case COMMON_CHAT_FORMAT_MISTRAL_NEMO:
            common_chat_parse_mistral_nemo(builder);
            break;
        case COMMON_CHAT_FORMAT_LLAMA_3_X:
            common_chat_parse_llama_3_1(builder);
            break;
        case COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS:
            common_chat_parse_llama_3_1(builder, /* with_builtin_tools= */ true);
            break;
        case COMMON_CHAT_FORMAT_DEEPSEEK_R1:
            common_chat_parse_deepseek_r1(builder);
            break;
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2:
            common_chat_parse_functionary_v3_2(builder);
            break;
        case COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1:
            common_chat_parse_functionary_v3_1_llama_3_1(builder);
            break;
        case COMMON_CHAT_FORMAT_HERMES_2_PRO:
            common_chat_parse_hermes_2_pro(builder);
            break;
        case COMMON_CHAT_FORMAT_FIREFUNCTION_V2:
            common_chat_parse_firefunction_v2(builder);
            break;
        case COMMON_CHAT_FORMAT_COMMAND_R7B:
            common_chat_parse_command_r7b(builder);
            break;
        default:
            throw std::runtime_error(std::string("Unsupported format: ") + common_chat_format_name(builder.syntax().format));
    }
    builder.finish();
}

common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_syntax & syntax) {
    common_chat_msg_parser builder(input, is_partial, syntax);
    try {
        common_chat_parse(builder);
    } catch (const common_chat_msg_partial_exception & ex) {
        LOG_DBG("Partial parse: %s\n", ex.what());
        if (!is_partial) {
            builder.clear_tools();
            builder.move_to(0);
            common_chat_parse_content_only(builder);
        }
    }
    auto msg = builder.result();
    LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat<json>({msg}).at(0).dump().c_str());
    return msg;
}
