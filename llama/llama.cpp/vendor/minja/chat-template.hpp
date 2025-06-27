/*
    Copyright 2024 Google LLC

    Use of this source code is governed by an MIT-style
    license that can be found in the LICENSE file or at
    https://opensource.org/licenses/MIT.
*/
// SPDX-License-Identifier: MIT
#pragma once

#include "minja.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <exception>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace minja {

struct chat_template_caps {
    bool supports_tools = false;
    bool supports_tool_calls = false;
    bool supports_tool_responses = false;
    bool supports_system_role = false;
    bool supports_parallel_tool_calls = false;
    bool supports_tool_call_id = false;
    // meta-llama/Llama-3.1-8B-Instruct expects arguments to be an object.
    // Most other templates (and OpenAI's API) expect the arguments object to be stringified.
    bool requires_object_arguments = false;
    // CohereForAI/c4ai-command-r-plus simple variant
    bool requires_non_null_content = false;
    // MiniMaxAI/MiniMax-Text-01 special
    bool requires_typed_content = false;
};

struct chat_template_inputs {
    nlohmann::ordered_json messages;
    nlohmann::ordered_json tools;
    bool add_generation_prompt = true;
    nlohmann::ordered_json extra_context;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
};

struct chat_template_options {
    bool apply_polyfills = true;
    bool use_bos_token = true;
    bool use_eos_token = true;
    bool define_strftime_now = true;

    bool polyfill_tools = true;
    bool polyfill_tool_call_examples = true;
    bool polyfill_tool_calls = true;
    bool polyfill_tool_responses = true;
    bool polyfill_system_role = true;
    bool polyfill_object_arguments = true;
    bool polyfill_typed_content = true;
};

class chat_template {

  private:
    chat_template_caps caps_;
    std::string source_;
    std::string bos_token_;
    std::string eos_token_;
    std::shared_ptr<minja::TemplateNode> template_root_;
    std::string tool_call_example_;

    std::string try_raw_render(
        const nlohmann::ordered_json & messages,
        const nlohmann::ordered_json & tools,
        bool add_generation_prompt,
        const nlohmann::ordered_json & extra_context = nlohmann::ordered_json()) const
    {
        try {
            chat_template_inputs inputs;
            inputs.messages = messages;
            inputs.tools = tools;
            inputs.add_generation_prompt = add_generation_prompt;
            inputs.extra_context = extra_context;
            // Use fixed date for tests
            inputs.now = std::chrono::system_clock::from_time_t(0);

            chat_template_options opts;
            opts.apply_polyfills = false;

            auto prompt = apply(inputs, opts);
            // fprintf(stderr, "try_raw_render: %s\n", prompt.c_str());
            return prompt;
        } catch (const std::exception & e) {
            // fprintf(stderr, "try_raw_render error: %s\n", e.what());
            return "";
        }
    }

  public:

    chat_template(const std::string & source, const std::string & bos_token, const std::string & eos_token)
        : source_(source), bos_token_(bos_token), eos_token_(eos_token)
    {
        template_root_ = minja::Parser::parse(source_, {
            /* .trim_blocks = */ true,
            /* .lstrip_blocks = */ true,
            /* .keep_trailing_newline = */ false,
        });

        auto contains = [](const std::string & haystack, const std::string & needle) {
            return haystack.find(needle) != std::string::npos;
        };

        const std::string user_needle = "<User Needle>";
        const std::string sys_needle = "<System Needle>";
        const json dummy_str_user_msg = {{"role", "user"}, {"content", user_needle}};
        const json dummy_typed_user_msg = {{"role", "user"}, {"content", json::array({{{"type", "text"}, {"text", user_needle}}})}};

        caps_.requires_typed_content =
            !contains(try_raw_render(json::array({dummy_str_user_msg}), {}, false), user_needle)
            && contains(try_raw_render(json::array({dummy_typed_user_msg}), {}, false), user_needle);

        const auto dummy_user_msg = caps_.requires_typed_content
            ? dummy_typed_user_msg
            : dummy_str_user_msg;
        const json needle_system_msg = {
            {"role", "system"},
            {"content", caps_.requires_typed_content ? json::array({{{"type", "text"}, {"text", sys_needle}}}) : json(sys_needle)},
        };

        caps_.supports_system_role = contains(try_raw_render({needle_system_msg, dummy_user_msg,}, {}, false), sys_needle);

        auto out = try_raw_render(json::array({
            dummy_user_msg
        }), json::array({
            {
                {"name", "some_tool"},
                {"type", "function"},
                {"function", {
                    {"name", "some_tool"},
                    {"description", "Some tool."},
                    {"parameters", {
                        {"type", "object"},
                        {"properties", {
                            {"arg", {
                                {"type", "string"},
                                {"description", "Some argument."},
                            }},
                        }},
                        {"required", json::array({ "arg" })},
                    }},
                }},
            },
        }), false);
        caps_.supports_tools = contains(out, "some_tool");

        auto make_tool_calls_msg = [&](const json & tool_calls) {
            return json {
                {"role", "assistant"},
                {"content", nullptr},
                {"tool_calls", tool_calls},
            };
        };
        auto make_tool_call = [](const std::string & tool_name, const json & arguments) {
            return json {
                {"id", "call_1___"},
                {"type", "function"},
                {"function", {
                    {"arguments", arguments},
                    {"name", tool_name},
                }},
            };
        };
        const json dummy_args_obj {{"argument_needle", "print('Hello, World!')"}};

        // Note: the arguments are rendered in both cases, but may be double-escaped, which we don't want.
        out = try_raw_render(json::array({
            dummy_user_msg,
            make_tool_calls_msg(json::array({make_tool_call("ipython", dummy_args_obj.dump())})),
        }), {}, false);
        auto tool_call_renders_str_arguments = contains(out, "\"argument_needle\":") || contains(out, "'argument_needle':");
        out = try_raw_render(json::array({
            dummy_user_msg,
            make_tool_calls_msg(json::array({make_tool_call("ipython", dummy_args_obj)})),
        }), {}, false);
        auto tool_call_renders_obj_arguments = contains(out, "\"argument_needle\":") || contains(out, "'argument_needle':");

        caps_.supports_tool_calls = tool_call_renders_str_arguments || tool_call_renders_obj_arguments;
        caps_.requires_object_arguments = !tool_call_renders_str_arguments && tool_call_renders_obj_arguments;
        auto out_empty = try_raw_render(json::array({dummy_user_msg, {{"role", "assistant"}, {"content", ""}}}), {}, false);
        auto out_null = try_raw_render(json::array({dummy_user_msg, {{"role", "assistant"}, {"content", nullptr}}}), {}, false);
        caps_.requires_non_null_content = contains(out_empty, user_needle) && !contains(out_null, user_needle);

        if (caps_.supports_tool_calls) {
            auto dummy_args = caps_.requires_object_arguments ? dummy_args_obj : json(dummy_args_obj.dump());
            auto tc1 = make_tool_call("test_tool1", dummy_args);
            auto tc2 = make_tool_call("test_tool2", dummy_args);
            auto out = try_raw_render(json::array({
                dummy_user_msg,
                make_tool_calls_msg(json::array({tc1, tc2})),
            }), {}, false);
            caps_.supports_parallel_tool_calls = contains(out, "test_tool1") && contains(out, "test_tool2");

            out = try_raw_render(json::array({
                dummy_user_msg,
                make_tool_calls_msg(json::array({tc1})),
                {
                    {"role", "tool"},
                    {"name", "test_tool1"},
                    {"content", "Some response!"},
                    {"tool_call_id", "call_911_"},
                }
            }), {}, false);
            caps_.supports_tool_responses = contains(out, "Some response!");
            caps_.supports_tool_call_id = contains(out, "call_911_");
        }

        try {
            if (!caps_.supports_tools) {
                const json user_msg {
                    {"role", "user"},
                    {"content", "Hey"},
                };
                const json args {
                    {"arg1", "some_value"},
                };
                const json tool_call_msg {
                    {"role", "assistant"},
                    {"content", nullptr},
                    {"tool_calls", json::array({
                        {
                            // TODO: detect if requires numerical id or fixed length == 6 like Nemo
                            {"id", "call_1___"},
                            {"type", "function"},
                            {"function", {
                                {"name", "tool_name"},
                                {"arguments", (caps_.requires_object_arguments ? args : json(minja::Value(args).dump(-1, /* to_json= */ true)))},
                            }},
                        },
                    })},
                };
                std::string prefix, full;
                {
                    chat_template_inputs inputs;
                    inputs.messages = json::array({user_msg});
                    inputs.add_generation_prompt = true;
                    prefix = apply(inputs);
                }
                {
                    chat_template_inputs inputs;
                    inputs.messages = json::array({user_msg, tool_call_msg});
                    inputs.add_generation_prompt = false;
                    full = apply(inputs);
                }
                auto eos_pos_last = full.rfind(eos_token_);
                if (eos_pos_last == prefix.size() - eos_token_.size() ||
                      (full[full.size() - 1] == '\n' && (eos_pos_last == full.size() - eos_token_.size() - 1))) {
                    full = full.substr(0, eos_pos_last);
                }
                size_t common_prefix_length = 0;
                for (size_t i = 0; i < prefix.size() && i < full.size(); ++i) {
                    if (prefix[i] != full[i]) {
                        break;
                    }
                    if (prefix[i] == '<') {
                        // DeepSeek R1's template (as of 20250209) adds a trailing <think> if add_generation_prompt,
                        // but it removes thinking tags for past messages.
                        // The prefix and full strings diverge at <think> vs. <｜tool▁calls▁begin｜>, we avoid consuming the leading <.
                        continue;
                    }
                    common_prefix_length = i + 1;
                }
                auto example = full.substr(common_prefix_length);
                if (example.find("tool_name") == std::string::npos && example.find("some_value") == std::string::npos) {
                    fprintf(stderr, "Failed to infer a tool call example (possible template bug)\n");
                } else {
                    tool_call_example_ = example;
                }
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "Failed to generate tool call example: %s\n", e.what());
        }
    }

    const std::string & source() const { return source_; }
    const std::string & bos_token() const { return bos_token_; }
    const std::string & eos_token() const { return eos_token_; }
    const chat_template_caps & original_caps() const { return caps_; }

    // Deprecated, please use the form with chat_template_inputs and chat_template_options
    std::string apply(
        const nlohmann::ordered_json & messages,
        const nlohmann::ordered_json & tools,
        bool add_generation_prompt,
        const nlohmann::ordered_json & extra_context = nlohmann::ordered_json(),
        bool apply_polyfills = true)
    {
        fprintf(stderr, "[%s] Deprecated!\n", __func__);
        chat_template_inputs inputs;
        inputs.messages = messages;
        inputs.tools = tools;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.extra_context = extra_context;
        inputs.now = std::chrono::system_clock::now();

        chat_template_options opts;
        opts.apply_polyfills = apply_polyfills;

        return apply(inputs, opts);
    }

    std::string apply(
        const chat_template_inputs & inputs,
        const chat_template_options & opts = chat_template_options()) const
    {
        json actual_messages;

        auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
        auto has_tool_calls = false;
        auto has_tool_responses = false;
        auto has_string_content = false;
        for (const auto & message : inputs.messages) {
            if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
                has_tool_calls = true;
            }
            if (message.contains("role") && message["role"] == "tool") {
                has_tool_responses = true;
            }
            if (message.contains("content") && message["content"].is_string()) {
                has_string_content = true;
            }
        }

        auto polyfill_system_role = opts.polyfill_system_role && !caps_.supports_system_role;
        auto polyfill_tools = opts.polyfill_tools && has_tools && !caps_.supports_tools;
        auto polyfill_tool_call_example = polyfill_tools && opts.polyfill_tool_call_examples;
        auto polyfill_tool_calls = opts.polyfill_tool_calls && has_tool_calls && !caps_.supports_tool_calls;
        auto polyfill_tool_responses = opts.polyfill_tool_responses && has_tool_responses && !caps_.supports_tool_responses;
        auto polyfill_object_arguments = opts.polyfill_object_arguments && has_tool_calls && caps_.requires_object_arguments;
        auto polyfill_typed_content = opts.polyfill_typed_content && has_string_content && caps_.requires_typed_content;

        auto needs_polyfills = opts.apply_polyfills && (false
            || polyfill_system_role
            || polyfill_tools
            || polyfill_tool_calls
            || polyfill_tool_responses
            || polyfill_object_arguments
            || polyfill_typed_content
        );

        if (needs_polyfills) {
            actual_messages = json::array();

            auto add_message = [&](const json & msg) {
                if (polyfill_typed_content && msg.contains("content") && !msg.at("content").is_null() && msg.at("content").is_string()) {
                    actual_messages.push_back({
                        {"role", msg.at("role")},
                        {"content", {{
                            {"type", "text"},
                            {"text", msg.at("content")},
                        }}},
                    });
                } else {
                    actual_messages.push_back(msg);
                }
            };

            std::string pending_system;
            auto flush_sys = [&]() {
                if (!pending_system.empty()) {
                    add_message({
                        {"role", "user"},
                        {"content", pending_system},
                    });
                    pending_system.clear();
                }
            };

            json adjusted_messages;
            if (polyfill_tools) {
                adjusted_messages = add_system(inputs.messages,
                    "You can call any of the following tools to satisfy the user's requests: " + minja::Value(inputs.tools).dump(2, /* to_json= */ true) +
                    (!polyfill_tool_call_example || tool_call_example_.empty() ? "" : "\n\nExample tool call syntax:\n\n" + tool_call_example_ + "\n\n"));
            } else {
                adjusted_messages = inputs.messages;
            }

            for (const auto & message_ : adjusted_messages) {
                auto message = message_;
                if (!message.contains("role") || (!message.contains("content") && !message.contains("tool_calls"))) {
                    throw std::runtime_error("message must have 'role' and one of 'content' or 'tool_calls' fields: " + message.dump());
                }
                std::string role = message.at("role");

                if (message.contains("tool_calls")) {
                    if (polyfill_object_arguments || polyfill_tool_calls) {
                        for (auto & tool_call : message.at("tool_calls")) {
                            if (tool_call["type"] == "function") {
                                auto & function = tool_call.at("function");
                                auto & arguments = function.at("arguments");
                                if (arguments.is_string()) {
                                    try {
                                        arguments = json::parse(arguments.get<std::string>());
                                    } catch (const std::exception & ecvt) {
                                        fprintf(stderr, "Failed to parse arguments: %s\n", ecvt.what());
                                    }
                                }
                            }
                        }
                    }
                    if (polyfill_tool_calls) {
                        auto tool_calls = json::array();
                        for (const auto & tool_call : message.at("tool_calls")) {
                            if (tool_call.at("type") != "function") {
                                continue;
                            }
                            const auto & function = tool_call.at("function");
                            auto tc = json {
                                {"name", function.at("name")},
                                {"arguments", function.at("arguments")},
                            };
                            if (tool_call.contains("id")) {
                                tc["id"] = tool_call["id"];
                            }
                            tool_calls.push_back(tc);
                        }
                        auto obj = json {
                            {"tool_calls", tool_calls},
                        };
                        if (message.contains("content")) {
                            auto content = message.at("content");
                            if (!content.is_null() && !content.empty()) {
                                obj["content"] = content;
                            }
                        }
                        message["content"] = obj.dump(2);
                        message.erase("tool_calls");
                    }
                }
                if (polyfill_tool_responses && role == "tool") {
                    message["role"] = "user";
                    auto obj = json {
                        {"tool_response", json::object()},
                    };
                    if (message.contains("name")) {
                        obj["tool_response"]["tool"] = message.at("name");
                    }
                    obj["tool_response"]["content"] = message.at("content");
                    if (message.contains("tool_call_id")) {
                        obj["tool_response"]["tool_call_id"] = message.at("tool_call_id");
                    }
                    message["content"] = obj.dump(2);
                    message.erase("name");
                }

                if (!message["content"].is_null() && polyfill_system_role) {
                    std::string content = message.at("content");
                    if (role == "system") {
                        if (!pending_system.empty()) pending_system += "\n";
                        pending_system += content;
                        continue;
                    } else {
                        if (role == "user") {
                            if (!pending_system.empty()) {
                                message["content"] = pending_system + (content.empty() ? "" : "\n" + content);
                                pending_system.clear();
                            }
                        } else {
                            flush_sys();
                        }
                    }
                }
                add_message(message);
            }
            flush_sys();
        } else {
            actual_messages = inputs.messages;
        }

        auto context = minja::Context::make(json({
            {"messages", actual_messages},
            {"add_generation_prompt", inputs.add_generation_prompt},
        }));
        context->set("bos_token", opts.use_bos_token ? bos_token_ : "");
        context->set("eos_token", opts.use_eos_token ? eos_token_ : "");
        if (opts.define_strftime_now) {
            auto now = inputs.now;
            context->set("strftime_now", Value::callable([now](const std::shared_ptr<minja::Context> &, minja::ArgumentsValue & args) {
                args.expectArgs("strftime_now", {1, 1}, {0, 0});
                auto format = args.args[0].get<std::string>();

                auto time = std::chrono::system_clock::to_time_t(now);
                auto local_time = *std::localtime(&time);
                std::ostringstream ss;
                ss << std::put_time(&local_time, format.c_str());
                return ss.str();
            }));
        }
        if (!inputs.tools.is_null()) {
            context->set("tools", minja::Value(inputs.tools));
        }
        if (!inputs.extra_context.is_null()) {
            for (auto & kv : inputs.extra_context.items()) {
                context->set(kv.key(), minja::Value(kv.value()));
            }
        }

        auto ret = template_root_->render(context);
        // fprintf(stderr, "actual_messages: %s\n", actual_messages.dump(2).c_str());
        // fprintf(stderr, "apply: %s\n\n", ret.c_str());
        return ret;
    }

    static nlohmann::ordered_json add_system(const nlohmann::ordered_json & messages, const std::string & system_prompt) {
        json messages_with_system = messages;

        if (!messages_with_system.empty() && messages_with_system[0].at("role") == "system") {
            std::string existing_system = messages_with_system.at(0).at("content");
            messages_with_system[0] = json {
                {"role", "system"},
                {"content", existing_system + "\n\n" + system_prompt},
            };
        } else {
            messages_with_system.insert(messages_with_system.begin(), json {
                {"role", "system"},
                {"content", system_prompt},
            });
        }
        return messages_with_system;
    }
};

}  // namespace minja
