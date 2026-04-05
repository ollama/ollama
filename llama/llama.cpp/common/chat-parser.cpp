#include "chat-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "log.h"
#include "peg-parser.h"
#include "regex-partial.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::ordered_json;

static void parse_prefixed_json_tool_call_array(common_chat_msg_parser & builder,
                                                const common_regex &     prefix,
                                                size_t                   rstrip_prefix = 0) {
    static const std::vector<std::vector<std::string>> args_paths = { { "arguments" } };
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

static std::string wrap_code_as_arguments(common_chat_msg_parser & builder, const std::string & code) {
    std::string arguments;
    if (builder.is_partial()) {
        arguments = (json{
                         { "code", code + builder.healing_marker() }
        })
                        .dump();
        auto idx = arguments.find(builder.healing_marker());
        if (idx != std::string::npos) {
            arguments.resize(idx);
        }
    } else {
        arguments = (json{
                         { "code", code }
        })
                        .dump();
    }
    return arguments;
}

/**
 * Takes a prefix regex that must have 1 group to capture the function name, a closing suffix, and expects json parameters in between.
 * Aggregates the prefix, suffix and in-between text into the content.
 */
static void parse_json_tool_calls(
    common_chat_msg_parser &            builder,
    const std::optional<common_regex> & block_open,
    const std::optional<common_regex> & function_regex_start_only,
    const std::optional<common_regex> & function_regex,
    const common_regex &                close_regex,
    const std::optional<common_regex> & block_close,
    bool                                allow_raw_python = false,
    const std::function<std::string(const common_chat_msg_parser::find_regex_result & fres)> & get_function_name =
        nullptr) {
    auto parse_tool_calls = [&]() {
        size_t from  = std::string::npos;
        auto   first = true;
        while (true) {
            auto start_pos = builder.pos();
            auto res = function_regex_start_only && first ? builder.try_consume_regex(*function_regex_start_only) :
                       function_regex                     ? builder.try_find_regex(*function_regex, from) :
                                                            std::nullopt;

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
                    if (auto arguments = builder.try_consume_json_with_dumped_args({ {} })) {
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
            } else {
                builder.move_to(start_pos);
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

common_chat_msg_parser::common_chat_msg_parser(const std::string & input, bool is_partial, const common_chat_syntax & syntax)
    : input_(input), is_partial_(is_partial), syntax_(syntax)
{
    result_.role = "assistant";

    while (true) {
        std::string id = std::to_string(std::rand());
        if (input.find(id) == std::string::npos) {
            healing_marker_ = id;
            break;
        }
    }
}

std::string common_chat_msg_parser::str(const common_string_range & rng) const {
    GGML_ASSERT(rng.begin <= rng.end);
    return input_.substr(rng.begin, rng.end - rng.begin);
}

void common_chat_msg_parser::add_content(const std::string &content) {
    result_.content += content;
}

void common_chat_msg_parser::add_reasoning_content(const std::string &reasoning_content) {
    result_.reasoning_content += reasoning_content;
}

bool common_chat_msg_parser::add_tool_call(const std::string & name, const std::string & id, const std::string & arguments) {
    if (name.empty()) {
        return false;
    }

    common_chat_tool_call tool_call;
    tool_call.name = name;
    tool_call.arguments = arguments;
    tool_call.id = id;

    // LOG_DBG("Tool call arguments:\n\traw: %s\n\tresult: %s\n", arguments.c_str(), tool_call.arguments.c_str());
    result_.tool_calls.emplace_back(tool_call);

    return true;
}
bool common_chat_msg_parser::add_tool_call(const json & tool_call) {
    std::string name = tool_call.contains("name") ? tool_call.at("name") : "";
    std::string id = tool_call.contains("id") ? tool_call.at("id") : "";
    std::string arguments = "";
    if (tool_call.contains("arguments")) {
        if (tool_call.at("arguments").is_object()) {
            arguments = tool_call.at("arguments").dump();
        } else {
            arguments = tool_call.at("arguments");
        }
    }

    return add_tool_call(name, id, arguments);
}

bool common_chat_msg_parser::add_tool_calls(const json & arr) {
    for (const auto & item : arr) {
        if (!add_tool_call(item)) {
            return false;
        }
    }
    return true;
}

bool common_chat_msg_parser::add_tool_call_short_form(const json & tool_call) {
    if (!tool_call.is_object() || tool_call.size() != 1) {
        return false;
    }

    // Get the tool name (the single key in the object)
    auto it = tool_call.begin();
    std::string name = it.key();

    if (name.empty()) {
        return false;
    }

    // Get the arguments (the nested object)
    const json & args_json = it.value();
    std::string arguments = "";

    if (args_json.is_object()) {
        arguments = args_json.dump();
    } else if (args_json.is_string()) {
        arguments = args_json;
    } else if (!args_json.is_null()) {
        // For other types, convert to string representation
        arguments = args_json.dump();
    }

    return add_tool_call(name, "", arguments);
}
void common_chat_msg_parser::finish() {
    if (!is_partial_ && pos_ != input_.size()) {
        throw std::runtime_error("Unexpected content at end of input");// + input_.substr(pos_));
    }
}

bool common_chat_msg_parser::consume_spaces() {
    const auto length = input_.size();
    auto consumed = false;
    while (pos_ < length && std::isspace(input_[pos_])) {
        ++pos_;
        consumed = true;
    }
    return consumed;
}

bool common_chat_msg_parser::try_consume_literal(const std::string & literal) {
    auto pos = pos_;
    for (auto i = 0u; i < literal.size(); ++i) {
        if (pos >= input_.size()) {
            return false;
        }
        if (input_[pos] != literal[i]) {
            return false;
        }
        ++pos;
    }
    pos_ = pos;
    return true;
}

std::optional<common_chat_msg_parser::find_regex_result>  common_chat_msg_parser::try_find_literal(const std::string & literal) {
    auto idx = input_.find(literal, pos_);
    if (idx != std::string::npos) {
        find_regex_result res;
        res.prelude = input_.substr(pos_, idx - pos_);
        auto end = idx + literal.size();
        res.groups.emplace_back(common_string_range{idx, end});
        move_to(end);
        return res;
    }
    if (is_partial_) {
        idx = string_find_partial_stop(input_, literal);
        if (idx != std::string::npos && idx >= pos_) {
            find_regex_result res;
            res.prelude = input_.substr(pos_, idx - pos_);
            auto end = input_.size();
            res.groups.emplace_back(common_string_range{idx, end});
            move_to(end);
            return res;
        }
    }
    return std::nullopt;
}

void common_chat_msg_parser::consume_literal(const std::string & literal) {
    if (!try_consume_literal(literal)) {
        throw common_chat_msg_partial_exception(literal);
    }
}

bool common_chat_msg_parser::try_parse_reasoning(const std::string & start_think, const std::string & end_think) {
    std::string pending_reasoning_prefix;

    if (syntax_.reasoning_format == COMMON_REASONING_FORMAT_NONE) {
        return false;
    }

    auto set_reasoning_prefix = [&](size_t prefix_pos) {
        if (!syntax_.thinking_forced_open || syntax_.reasoning_in_content) {
            return;
        }
        if (prefix_pos + start_think.size() > input_.size()) {
            pending_reasoning_prefix.clear();
            return;
        }
        // Capture the exact literal that opened the reasoning section so we can
        // surface it back to callers. This ensures formats that force the
        // reasoning tag open (e.g. DeepSeek R1) retain their original prefix
        // instead of dropping it during parsing.
        pending_reasoning_prefix = input_.substr(prefix_pos, start_think.size());
    };

    auto handle_reasoning = [&](const std::string & reasoning, bool closed) {
        auto stripped_reasoning = string_strip(reasoning);
        if (stripped_reasoning.empty()) {
            return;
        }
        if (syntax_.reasoning_in_content) {
            add_content(syntax_.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK ? "<think>" : start_think);
            add_content(stripped_reasoning);
            if (closed) {
                add_content(syntax_.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK ? "</think>" : end_think);
            }
        } else {
            if (!pending_reasoning_prefix.empty()) {
                add_reasoning_content(pending_reasoning_prefix);
                pending_reasoning_prefix.clear();
            }
            add_reasoning_content(stripped_reasoning);
        }
    };

    const size_t saved_pos = pos_;
    const size_t saved_content_size = result_.content.size();
    const size_t saved_reasoning_size = result_.reasoning_content.size();

    auto restore_state = [&]() {
        move_to(saved_pos);
        result_.content.resize(saved_content_size);
        result_.reasoning_content.resize(saved_reasoning_size);
    };

    // Allow leading whitespace to be preserved as content when reasoning is present at the start
    size_t cursor = pos_;
    size_t whitespace_end = cursor;
    while (whitespace_end < input_.size() && std::isspace(static_cast<unsigned char>(input_[whitespace_end]))) {
        ++whitespace_end;
    }

    if (whitespace_end >= input_.size()) {
        restore_state();
        if (syntax_.thinking_forced_open) {
            auto rest = input_.substr(saved_pos);
            if (!rest.empty()) {
                handle_reasoning(rest, /* closed */ !is_partial());
            }
            move_to(input_.size());
            return true;
        }
        return false;
    }

    cursor = whitespace_end;
    const size_t remaining = input_.size() - cursor;
    const size_t start_prefix = std::min(start_think.size(), remaining);
    const bool has_start_tag = input_.compare(cursor, start_prefix, start_think, 0, start_prefix) == 0;

    if (has_start_tag && start_prefix < start_think.size()) {
        move_to(input_.size());
        return true;
    }

    if (has_start_tag) {
        if (whitespace_end > pos_) {
            add_content(input_.substr(pos_, whitespace_end - pos_));
        }
        set_reasoning_prefix(cursor);
        cursor += start_think.size();
    } else if (syntax_.thinking_forced_open) {
        cursor = whitespace_end;
    } else {
        restore_state();
        return false;
    }
    while (true) {
        if (cursor >= input_.size()) {
            move_to(input_.size());
            return true;
        }

        size_t end_pos = input_.find(end_think, cursor);
        if (end_pos == std::string::npos) {
            std::string_view remaining_view(input_.data() + cursor, input_.size() - cursor);
            size_t partial_off = string_find_partial_stop(remaining_view, end_think);
            size_t reasoning_end = partial_off == std::string::npos ? input_.size() : cursor + partial_off;
            if (reasoning_end > cursor) {
                handle_reasoning(input_.substr(cursor, reasoning_end - cursor), /* closed */ partial_off == std::string::npos && !is_partial());
            }
            move_to(input_.size());
            return true;
        }

        if (end_pos > cursor) {
            handle_reasoning(input_.substr(cursor, end_pos - cursor), /* closed */ true);
        } else {
            handle_reasoning("", /* closed */ true);
        }

        cursor = end_pos + end_think.size();

        while (cursor < input_.size() && std::isspace(static_cast<unsigned char>(input_[cursor]))) {
            ++cursor;
        }

        const size_t next_remaining = input_.size() - cursor;
        if (next_remaining == 0) {
            move_to(cursor);
            return true;
        }

        const size_t next_prefix = std::min(start_think.size(), next_remaining);
        if (input_.compare(cursor, next_prefix, start_think, 0, next_prefix) == 0) {
            if (next_prefix < start_think.size()) {
                move_to(input_.size());
                return true;
            }
            set_reasoning_prefix(cursor);
            cursor += start_think.size();
            continue;
        }

        move_to(cursor);
        return true;
    }
}

std::string common_chat_msg_parser::consume_rest() {
    auto rest = input_.substr(pos_);
    pos_ = input_.size();
    return rest;
}

// Tries to find the regex, consumes it (pos right after it) and gives the prelude (right before it) and the groups to the callback.
std::optional<common_chat_msg_parser::find_regex_result> common_chat_msg_parser::try_find_regex(const common_regex & regex, size_t from, bool add_prelude_to_content) {
    auto m = regex.search(input_, from == std::string::npos ? pos_ : from);
    if (m.type == COMMON_REGEX_MATCH_TYPE_NONE) {
        return std::nullopt;
    }
    auto prelude = input_.substr(pos_, m.groups[0].begin - pos_);
    pos_ = m.groups[0].end;

    if (add_prelude_to_content) {
        add_content(prelude);
    }
    if (m.type == COMMON_REGEX_MATCH_TYPE_PARTIAL) {
        if (is_partial()) {
            throw common_chat_msg_partial_exception(regex.str());
        }
        return std::nullopt;
    }
    return find_regex_result{prelude, m.groups};
}

common_chat_msg_parser::find_regex_result common_chat_msg_parser::consume_regex(const common_regex & regex) {
    if (auto result = try_consume_regex(regex)) {
        return *result;
    }
    throw common_chat_msg_partial_exception(regex.str());
}

std::optional<common_chat_msg_parser::find_regex_result> common_chat_msg_parser::try_consume_regex(const common_regex & regex) {
    auto m = regex.search(input_, pos_);
    if (m.type == COMMON_REGEX_MATCH_TYPE_NONE) {
        return std::nullopt;
    }
    if (m.type == COMMON_REGEX_MATCH_TYPE_PARTIAL) {
        if (is_partial()) {
            throw common_chat_msg_partial_exception(regex.str());
        }
        return std::nullopt;
    }
    if (m.groups[0].begin != pos_) {
        // Didn't match at the current position.
        return std::nullopt;
    }
    pos_ = m.groups[0].end;

    return find_regex_result {
        /* .prelude = */ "",
        m.groups,
    };
}

std::optional<common_json> common_chat_msg_parser::try_consume_json() {
    auto it = input_.cbegin() + pos_;
    const auto end = input_.cend();
    common_json result;
    if (!common_json_parse(it, end, healing_marker_, result)) {
        return std::nullopt;
    }
    pos_ = std::distance(input_.cbegin(), it);
    if (result.healing_marker.marker.empty()) {
        // No healing marker, just return the parsed json
        return result;
    }
    if (!is_partial()) {
        throw common_chat_msg_partial_exception("JSON");
    }
    return result;
}

common_json common_chat_msg_parser::consume_json() {
    if (auto result = try_consume_json()) {
        return *result;
    }
    throw common_chat_msg_partial_exception("JSON");
}

common_chat_msg_parser::consume_json_result common_chat_msg_parser::consume_json_with_dumped_args(
    const std::vector<std::vector<std::string>> & args_paths,
    const std::vector<std::vector<std::string>> & content_paths
) {
    if (auto result = try_consume_json_with_dumped_args(args_paths, content_paths)) {
        return *result;
    }
    throw common_chat_msg_partial_exception("JSON");
}

std::optional<common_chat_msg_parser::consume_json_result> common_chat_msg_parser::try_consume_json_with_dumped_args(
    const std::vector<std::vector<std::string>> & args_paths,
    const std::vector<std::vector<std::string>> & content_paths
) {
    auto partial = try_consume_json();
    if (!partial) {
        return std::nullopt;
    }
    auto is_arguments_path = [&](const std::vector<std::string> & path) {
        return std::find(args_paths.begin(), args_paths.end(), path) != args_paths.end();
    };
    auto is_content_path = [&](const std::vector<std::string> & path) {
        return std::find(content_paths.begin(), content_paths.end(), path) != content_paths.end();
    };

    if (partial->healing_marker.marker.empty()) {
        if (args_paths.empty()) {
            // No arguments to dump, and JSON was parsed fully.
            return consume_json_result {
                partial->json,
                /* .is_partial = */ false,
            };
        }
        if (is_arguments_path({})) {
            // Entire JSON is the arguments and was parsed fully.
            return consume_json_result {
                partial->json.dump(/* indent */ -1, /* indent_char */ ' ', /* ensure_ascii */ true),
                /* .is_partial = */ false,
            };
        }
    }

    LOG_DBG("Parsed partial JSON: %s (json_healing_marker: %s)\n", partial->json.dump().c_str(), partial->healing_marker.json_dump_marker.c_str());

    auto found_healing_marker = false;
    std::vector<std::string> path;
    std::function<json(const json &)> remove_unsupported_healings_and_dump_args = [&](const json & j) -> json {
        if (is_arguments_path(path)) {
            auto arguments = j.dump(/* indent */ -1, /* indent_char */ ' ', /* ensure_ascii */ true);
            if (is_partial() && !partial->healing_marker.marker.empty()) {
                auto idx = arguments.find(partial->healing_marker.json_dump_marker);
                if (idx != std::string::npos) {
                    arguments.resize(idx);
                    found_healing_marker = true;
                }
                if (arguments == "\"") {
                    // This happens because of completing `:"$magic` after `"arguments"`
                    arguments = "";
                }
            }
            return arguments;
        }
        if (is_content_path(path)) {
            if (!j.is_string()) {
                throw std::runtime_error("Content path must be a string");
            }
            std::string str = j;
            auto idx = str.find(partial->healing_marker.marker); // not using json_dump_marker as we're inside a string
            if (idx != std::string::npos) {
                str.resize(idx);
                found_healing_marker = true;
            }
            return str;
        }
        if (j.is_object()) {
            auto obj = json::object();
            for (const auto & p : j.items()) {
                const auto & key = p.key();
                const auto & value = p.value();
                const std::string key_str = key; // NOLINT
                auto idx = key_str.find(healing_marker_);
                if (idx != std::string::npos) {
                    found_healing_marker = true;
                    break;
                }
                path.push_back(key_str);
                if (value.is_string()) {
                    const std::string value_str = value;
                    if (value_str.find(healing_marker_) != std::string::npos) {
                        found_healing_marker = true;
                        if (is_content_path(path)) {
                            if (partial->healing_marker.marker == partial->healing_marker.json_dump_marker) {
                                // The healing occurred inside the string: good. Otherwise we just ditch the entire key/value pair.
                                obj[key] = remove_unsupported_healings_and_dump_args(value);
                            }
                        }
                        break;
                    }
                    obj[key] = value;
                } else {
                    obj[key] = remove_unsupported_healings_and_dump_args(value);
                }
                path.pop_back();
            }
            return obj;
        }
        if (j.is_array()) {
            auto arr = json::array();
            for (const auto & value : j) {
                if (value.is_string()) {
                    std::string str = value;
                    auto idx = str.find(healing_marker_);
                    if (idx != std::string::npos) {
                        // Don't heal array values that aren't in the arguments.
                        found_healing_marker = true;
                        break;
                    }
                }
                arr.push_back(remove_unsupported_healings_and_dump_args(value));
            }
            return arr;
        }
        return j;
    };

    auto cleaned = remove_unsupported_healings_and_dump_args(partial->json);
    LOG_DBG("Cleaned up JSON %s to %s (json_healing_marker : '%s')\n", partial->json.dump().c_str(), cleaned.dump().c_str(), partial->healing_marker.json_dump_marker.c_str());
    return consume_json_result {
        cleaned,
        /* .is_partial = */ found_healing_marker,
    };
}

void common_chat_msg_parser::clear_tools() {
    result_.tool_calls.clear();
}

/**
 * All common_chat_parse_* moved from chat.cpp to chat-parser.cpp below
 * to reduce incremental compile time for parser changes.
 */
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

static void common_chat_parse_mistral_nemo(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex prefix(regex_escape("[TOOL_CALLS]"));
    parse_prefixed_json_tool_call_array(builder, prefix);
}

static void common_chat_parse_magistral(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("[THINK]", "[/THINK]");

    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    static const common_regex prefix(regex_escape("[TOOL_CALLS]"));
    parse_prefixed_json_tool_call_array(builder, prefix);
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

static void common_chat_parse_llama_3_1(common_chat_msg_parser & builder, bool with_builtin_tools = false) {
    builder.try_parse_reasoning("<think>", "</think>");

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

static void common_chat_parse_deepseek_v3_1_content(common_chat_msg_parser & builder) {
    static const common_regex function_regex("(?:<｜tool▁call▁begin｜>)?([^\\n<]+)(?:<｜tool▁sep｜>)");

    static const common_regex close_regex("(?:[\\s]*)?<｜tool▁call▁end｜>");
    static const common_regex tool_calls_begin("(?:<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)");
    static const common_regex tool_calls_end("<｜tool▁calls▁end｜>");

    if (!builder.syntax().parse_tool_calls) {
        LOG_DBG("%s: not parse_tool_calls\n", __func__);
        builder.add_content(builder.consume_rest());
        return;
    }

    LOG_DBG("%s: parse_tool_calls\n", __func__);

    parse_json_tool_calls(
        builder,
        /* block_open= */ tool_calls_begin,
        /* function_regex_start_only= */ std::nullopt,
        function_regex,
        close_regex,
        tool_calls_end);
}

static void common_chat_parse_deepseek_v3_1(common_chat_msg_parser & builder) {
    // DeepSeek V3.1 outputs reasoning content between "<think>" and "</think>" tags, followed by regular content
    // First try to parse using the standard reasoning parsing method
    LOG_DBG("%s: thinking_forced_open: %s\n", __func__, std::to_string(builder.syntax().thinking_forced_open).c_str());

    auto start_pos = builder.pos();
    auto found_end_think = builder.try_find_literal("</think>");
    builder.move_to(start_pos);

    if (builder.syntax().thinking_forced_open && !builder.is_partial() && !found_end_think) {
        LOG_DBG("%s: no end_think, not partial, adding content\n", __func__);
        common_chat_parse_deepseek_v3_1_content(builder);
    } else if (builder.try_parse_reasoning("<think>", "</think>")) {
        // If reasoning was parsed successfully, the remaining content is regular content
        LOG_DBG("%s: parsed reasoning, adding content\n", __func__);
        // </think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\nJSON\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
        common_chat_parse_deepseek_v3_1_content(builder);
    } else {
        if (builder.syntax().reasoning_format == COMMON_REASONING_FORMAT_NONE) {
          LOG_DBG("%s: reasoning_format none, adding content\n", __func__);
          common_chat_parse_deepseek_v3_1_content(builder);
          return;
        }
        // If no reasoning tags found, check if we should treat everything as reasoning
        if (builder.syntax().thinking_forced_open) {
            // If thinking is forced open but no tags found, treat everything as reasoning
            LOG_DBG("%s: thinking_forced_open, adding reasoning content\n", __func__);
            builder.add_reasoning_content(builder.consume_rest());
        } else {
            LOG_DBG("%s: no thinking_forced_open, adding content\n", __func__);
            // <｜tool▁call▁begin｜>NAME<｜tool▁sep｜>JSON<｜tool▁call▁end｜>
            common_chat_parse_deepseek_v3_1_content(builder);
        }
    }
}

static void common_chat_parse_minimax_m2(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form {
        /* form.scope_start = */ "<minimax:tool_call>",
        /* form.tool_start  = */ "<invoke name=\"",
        /* form.tool_sep    = */ "\">",
        /* form.key_start   = */ "<parameter name=\"",
        /* form.key_val_sep = */ "\">",
        /* form.val_end     = */ "</parameter>",
        /* form.tool_end    = */ "</invoke>",
        /* form.scope_end   = */ "</minimax:tool_call>",
    };
    builder.consume_reasoning_with_xml_tool_calls(form, "<think>", "</think>");
}

static void common_chat_parse_qwen3_coder_xml(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "<tool_call>";
        form.tool_start  = "<function=";
        form.tool_sep    = ">";
        form.key_start   = "<parameter=";
        form.key_val_sep = ">";
        form.val_end     = "</parameter>";
        form.tool_end    = "</function>";
        form.scope_end   = "</tool_call>";
        form.trim_raw_argval = true;
        return form;
    })();
    builder.consume_reasoning_with_xml_tool_calls(form);
}

static void common_chat_parse_kimi_k2(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "<|tool_calls_section_begin|>";
        form.tool_start  = "<|tool_call_begin|>";
        form.tool_sep    = "<|tool_call_argument_begin|>{";
        form.key_start   = "\"";
        form.key_val_sep = "\":";
        form.val_end     = ",";
        form.tool_end    = "}<|tool_call_end|>";
        form.scope_end   = "<|tool_calls_section_end|>";
        form.raw_argval  = false;
        form.last_val_end = "";
        form.allow_toolcall_in_think = true;
        return form;
    })();
    builder.consume_reasoning_with_xml_tool_calls(form, "<think>", "</think>");
}

static void common_chat_parse_apriel_1_5(common_chat_msg_parser & builder) {
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
    builder.consume_reasoning_with_xml_tool_calls(form, "<thinking>", "</thinking>");
}

static void common_chat_parse_xiaomi_mimo(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form = ([]() {
        xml_tool_call_format form {};
        form.scope_start = "";
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
    builder.consume_reasoning_with_xml_tool_calls(form);
}

static void common_chat_parse_gpt_oss(common_chat_msg_parser & builder) {
    static const std::string constraint = "(?: (<\\|constrain\\|>)?([a-zA-Z0-9_-]+))";
    static const std::string recipient("(?: to=functions\\.([^<\\s]+))");

    static const common_regex start_regex("<\\|start\\|>assistant");
    static const common_regex analysis_regex("<\\|channel\\|>analysis");
    static const common_regex final_regex("<\\|channel\\|>final" + constraint + "?");
    static const common_regex preamble_regex("<\\|channel\\|>commentary");
    static const common_regex tool_call1_regex(recipient + "<\\|channel\\|>(analysis|commentary)" + constraint + "?");
    static const common_regex tool_call2_regex("<\\|channel\\|>(analysis|commentary)" + recipient + constraint + "?");

    auto consume_end = [&](bool include_end = false) {
        if (auto res = builder.try_find_literal("<|end|>")) {
            return res->prelude + (include_end ? builder.str(res->groups[0]) : "");
        }
        return builder.consume_rest();
    };

    auto handle_tool_call = [&](const std::string & name) {
        if (auto args = builder.try_consume_json_with_dumped_args({{}})) {
            if (builder.syntax().parse_tool_calls) {
                if (!builder.add_tool_call(name, "", args->value) || args->is_partial) {
                    throw common_chat_msg_partial_exception("incomplete tool call");
                }
            } else if (args->is_partial) {
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
        }
    };

    auto regex_match = [](const common_regex & regex, const std::string & input) -> std::optional<common_regex_match> {
        auto match = regex.search(input, 0, true);
        if (match.type == COMMON_REGEX_MATCH_TYPE_FULL) {
            return match;
        }
        return std::nullopt;
    };

    do {
        auto header_start_pos = builder.pos();
        auto content_start = builder.try_find_literal("<|message|>");
        if (!content_start) {
            throw common_chat_msg_partial_exception("incomplete header");
        }

        auto header = content_start->prelude;

        if (auto match = regex_match(tool_call1_regex, header)) {
            auto group = match->groups[1];
            auto name = header.substr(group.begin, group.end - group.begin);
            handle_tool_call(name);
            continue;
        }

        if (auto match = regex_match(tool_call2_regex, header)) {
            auto group = match->groups[2];
            auto name = header.substr(group.begin, group.end - group.begin);
            handle_tool_call(name);
            continue;
        }

        if (regex_match(analysis_regex, header)) {
            builder.move_to(header_start_pos);
            if (builder.syntax().reasoning_format == COMMON_REASONING_FORMAT_NONE || builder.syntax().reasoning_in_content) {
                builder.add_content(consume_end(true));
            } else {
                builder.try_parse_reasoning("<|channel|>analysis<|message|>", "<|end|>");
            }
            continue;
        }

        if(regex_match(final_regex, header) || regex_match(preamble_regex, header)) {
            builder.add_content(consume_end());
            continue;
        }

        // Possibly a malformed message, attempt to recover by rolling
        // back to pick up the next <|start|>
        LOG_DBG("%s: unknown header from message: %s\n", __func__, header.c_str());
        builder.move_to(header_start_pos);
    } while (builder.try_find_regex(start_regex, std::string::npos, false));

    auto remaining = builder.consume_rest();
    if (!remaining.empty()) {
        LOG_DBG("%s: content after last message: %s\n", __func__, remaining.c_str());
    }
}

static void common_chat_parse_glm_4_5(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form {
        /* form.scope_start  = */ "",
        /* form.tool_start   = */ "<tool_call>",
        /* form.tool_sep     = */ "",
        /* form.key_start    = */ "<arg_key>",
        /* form.key_val_sep  = */ "</arg_key>",
        /* form.val_end      = */ "</arg_value>",
        /* form.tool_end     = */ "</tool_call>",
        /* form.scope_end    = */ "",
        /* form.key_val_sep2 = */ "<arg_value>",
    };
    builder.consume_reasoning_with_xml_tool_calls(form, "<think>", "</think>");
}

static void common_chat_parse_firefunction_v2(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }
    static const common_regex prefix(regex_escape(" functools["));
    parse_prefixed_json_tool_call_array(builder, prefix, /* rstrip_prefix= */ 1);
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

    while (auto res = builder.try_find_regex(open_regex)) {
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
        }
    }

    builder.add_content(builder.consume_rest());
}

static void common_chat_parse_granite(common_chat_msg_parser & builder) {
    // Parse thinking tags
    static const common_regex start_think_regex(regex_escape("<think>"));
    static const common_regex end_think_regex(regex_escape("</think>"));
    // Granite models output partial tokens such as "<" and "<think".
    // By leveraging try_consume_regex()/try_find_regex() throwing
    // common_chat_msg_partial_exception for these partial tokens,
    // processing is interrupted and the tokens are not passed to add_content().
    if (auto res = builder.try_consume_regex(start_think_regex)) {
        // Restore position for try_parse_reasoning()
        builder.move_to(res->groups[0].begin);
        builder.try_find_regex(end_think_regex, std::string::npos, false);
        // Restore position for try_parse_reasoning()
        builder.move_to(res->groups[0].begin);
    }
    builder.try_parse_reasoning("<think>", "</think>");

    // Parse response tags
    static const common_regex start_response_regex(regex_escape("<response>"));
    static const common_regex end_response_regex(regex_escape("</response>"));
    // Granite models output partial tokens such as "<" and "<response".
    // Same hack as reasoning parsing.
    if (builder.try_consume_regex(start_response_regex)) {
        builder.try_find_regex(end_response_regex);
    }

    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    // Look for tool calls
    static const common_regex tool_call_regex(regex_escape("<|tool_call|>"));
    if (auto res = builder.try_find_regex(tool_call_regex)) {
        builder.move_to(res->groups[0].end);

        // Expect JSON array of tool calls
        if (auto tool_call = builder.try_consume_json_with_dumped_args({{{"arguments"}}})) {
            if (!builder.add_tool_calls(tool_call->value) || tool_call->is_partial) {
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
        }
    } else {
        builder.add_content(builder.consume_rest());
    }
}

static void common_chat_parse_nemotron_v2(common_chat_msg_parser & builder) {
    // Parse thinking tags
    builder.try_parse_reasoning("<think>", "</think>");
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    // Look for tool calls
    static const common_regex tool_call_regex(regex_escape("<TOOLCALL>"));
    if (auto res = builder.try_find_regex(tool_call_regex)) {
        builder.move_to(res->groups[0].end);

        // Expect JSON array of tool calls
        auto tool_calls_data = builder.consume_json();
        if (tool_calls_data.json.is_array()) {
            if (!builder.try_consume_literal("</TOOLCALL>")) {
                throw common_chat_msg_partial_exception("Incomplete tool call");
            }
            builder.add_tool_calls(tool_calls_data.json);
        } else {
            throw common_chat_msg_partial_exception("Incomplete tool call");
        }
    }
    builder.add_content(builder.consume_rest());
}

static void common_chat_parse_apertus(common_chat_msg_parser & builder) {
    // Parse thinking tags
    builder.try_parse_reasoning("<|inner_prefix|>", "<|inner_suffix|>");
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    // Look for tool calls
    static const common_regex tool_call_regex(regex_escape("<|tools_prefix|>"));
    if (auto res = builder.try_find_regex(tool_call_regex)) {
        builder.move_to(res->groups[0].end);

        auto tool_calls_data = builder.consume_json();
        if (tool_calls_data.json.is_array()) {
            builder.consume_spaces();
            if (!builder.try_consume_literal("<|tools_suffix|>")) {
                throw common_chat_msg_partial_exception("Incomplete tool call");
            }
            for (const auto & value : tool_calls_data.json) {
                if (value.is_object()) {
                    builder.add_tool_call_short_form(value);
                }
            }
        } else {
            throw common_chat_msg_partial_exception("Incomplete tool call");
        }
    }
    builder.add_content(builder.consume_rest());
}


static void common_chat_parse_lfm2(common_chat_msg_parser & builder) {
    if (!builder.syntax().parse_tool_calls) {
        builder.add_content(builder.consume_rest());
        return;
    }

    // LFM2 format: <|tool_call_start|>[{"name": "get_current_time", "arguments": {"location": "Paris"}}]<|tool_call_end|>
    static const common_regex tool_call_start_regex(regex_escape("<|tool_call_start|>"));
    static const common_regex tool_call_end_regex(regex_escape("<|tool_call_end|>"));

    // Loop through all tool calls
    while (auto res = builder.try_find_regex(tool_call_start_regex, std::string::npos, /* add_prelude_to_content= */ true)) {
        builder.move_to(res->groups[0].end);

        // Parse JSON array format: [{"name": "...", "arguments": {...}}]
        auto tool_calls_data = builder.consume_json();

        // Consume end marker
        builder.consume_spaces();
        if (!builder.try_consume_regex(tool_call_end_regex)) {
            throw common_chat_msg_partial_exception("Expected <|tool_call_end|>");
        }

        // Process each tool call in the array
        if (tool_calls_data.json.is_array()) {
            for (const auto & tool_call : tool_calls_data.json) {
                if (!tool_call.is_object()) {
                    throw common_chat_msg_partial_exception("Tool call must be an object");
                }

                if (!tool_call.contains("name")) {
                    throw common_chat_msg_partial_exception("Tool call missing 'name' field");
                }

                std::string function_name = tool_call.at("name");
                std::string arguments = "{}";

                if (tool_call.contains("arguments")) {
                    if (tool_call.at("arguments").is_object()) {
                        arguments = tool_call.at("arguments").dump();
                    } else if (tool_call.at("arguments").is_string()) {
                        arguments = tool_call.at("arguments");
                    }
                }

                if (!builder.add_tool_call(function_name, "", arguments)) {
                    throw common_chat_msg_partial_exception("Incomplete tool call");
                }
            }
        } else {
            throw common_chat_msg_partial_exception("Expected JSON array for tool calls");
        }

        // Consume any trailing whitespace after this tool call
        builder.consume_spaces();
    }

    // Consume any remaining content after all tool calls
    auto remaining = builder.consume_rest();
    if (!string_strip(remaining).empty()) {
        builder.add_content(remaining);
    }
}

static void common_chat_parse_seed_oss(common_chat_msg_parser & builder) {
    static const xml_tool_call_format form {
        /* form.scope_start = */ "<seed:tool_call>",
        /* form.tool_start  = */ "<function=",
        /* form.tool_sep    = */ ">",
        /* form.key_start   = */ "<parameter=",
        /* form.key_val_sep = */ ">",
        /* form.val_end     = */ "</parameter>",
        /* form.tool_end    = */ "</function>",
        /* form.scope_end   = */ "</seed:tool_call>",
    };
    builder.consume_reasoning_with_xml_tool_calls(form, "<seed:think>", "</seed:think>");
}

static void common_chat_parse_content_only(common_chat_msg_parser & builder) {
    builder.try_parse_reasoning("<think>", "</think>");
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
        case COMMON_CHAT_FORMAT_MAGISTRAL:
            common_chat_parse_magistral(builder);
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
        case COMMON_CHAT_FORMAT_DEEPSEEK_V3_1:
            common_chat_parse_deepseek_v3_1(builder);
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
        case COMMON_CHAT_FORMAT_GRANITE:
            common_chat_parse_granite(builder);
            break;
        case COMMON_CHAT_FORMAT_GPT_OSS:
            common_chat_parse_gpt_oss(builder);
            break;
        case COMMON_CHAT_FORMAT_SEED_OSS:
            common_chat_parse_seed_oss(builder);
            break;
        case COMMON_CHAT_FORMAT_NEMOTRON_V2:
            common_chat_parse_nemotron_v2(builder);
            break;
        case COMMON_CHAT_FORMAT_APERTUS:
            common_chat_parse_apertus(builder);
            break;
        case COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS:
            common_chat_parse_lfm2(builder);
            break;
        case COMMON_CHAT_FORMAT_MINIMAX_M2:
            common_chat_parse_minimax_m2(builder);
            break;
        case COMMON_CHAT_FORMAT_GLM_4_5:
            common_chat_parse_glm_4_5(builder);
            break;
        case COMMON_CHAT_FORMAT_KIMI_K2:
            common_chat_parse_kimi_k2(builder);
            break;
        case COMMON_CHAT_FORMAT_QWEN3_CODER_XML:
            common_chat_parse_qwen3_coder_xml(builder);
            break;
        case COMMON_CHAT_FORMAT_APRIEL_1_5:
            common_chat_parse_apriel_1_5(builder);
            break;
        case COMMON_CHAT_FORMAT_XIAOMI_MIMO:
            common_chat_parse_xiaomi_mimo(builder);
            break;
        default:
            throw std::runtime_error(std::string("Unsupported format: ") + common_chat_format_name(builder.syntax().format));
    }
    builder.finish();
}

common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_syntax & syntax) {
    if (syntax.format == COMMON_CHAT_FORMAT_PEG_SIMPLE ||
        syntax.format == COMMON_CHAT_FORMAT_PEG_NATIVE ||
        syntax.format == COMMON_CHAT_FORMAT_PEG_CONSTRUCTED) {
        return common_chat_peg_parse(syntax.parser, input, is_partial, syntax);
    }
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
    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat<json>({msg}).at(0).dump().c_str());
    }
    return msg;
}

common_chat_msg common_chat_peg_parse(const common_peg_arena & parser, const std::string & input, bool is_partial, const common_chat_syntax & syntax) {
    if (parser.empty()) {
        throw std::runtime_error("Failed to parse due to missing parser definition.");
    }

    LOG_DBG("Parsing input with format %s: %s\n", common_chat_format_name(syntax.format), input.c_str());

    common_peg_parse_context ctx(input, is_partial);
    auto result = parser.parse(ctx);
    if (result.fail()) {
        throw std::runtime_error(std::string("Failed to parse input at pos ") + std::to_string(result.end));
    }

    common_chat_msg msg;
    msg.role = "assistant";

    if (syntax.format == COMMON_CHAT_FORMAT_PEG_NATIVE) {
        auto mapper = common_chat_peg_native_mapper(msg);
        mapper.from_ast(ctx.ast, result);
    } else if (syntax.format == COMMON_CHAT_FORMAT_PEG_CONSTRUCTED) {
        auto mapper = common_chat_peg_constructed_mapper(msg);
        mapper.from_ast(ctx.ast, result);
    } else {
        // Generic mapper
        auto mapper = common_chat_peg_mapper(msg);
        mapper.from_ast(ctx.ast, result);
    }
    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat<json>({msg}).at(0).dump().c_str());
    }
    return msg;
}
