#pragma once

#include "chat.h"
#include "json-partial.h"
#include "regex-partial.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

class common_chat_msg_partial_exception : public std::runtime_error {
  public:
    common_chat_msg_partial_exception(const std::string & message) : std::runtime_error(message) {}
};

class common_chat_msg_parser {
    std::string input_;
    bool is_partial_;
    common_chat_syntax syntax_;
    std::string healing_marker_;

    size_t pos_ = 0;
    common_chat_msg result_;

  public:
    common_chat_msg_parser(const std::string & input, bool is_partial, const common_chat_syntax & syntax);
    const std::string & input() const { return input_; }
    size_t pos() const { return pos_; }
    const std::string & healing_marker() const { return healing_marker_; }
    const bool & is_partial() const { return is_partial_; }
    const common_chat_msg & result() const { return result_; }
    const common_chat_syntax & syntax() const { return syntax_; }

    void move_to(size_t pos) {
        if (pos > input_.size()) {
            throw std::runtime_error("Invalid position!");
        }
        pos_ = pos;
    }
    void move_back(size_t n) {
        if (pos_ < n) {
            throw std::runtime_error("Can't move back that far!");
        }
        pos_ -= n;
    }

    // Get the substring of the input at the given range
    std::string str(const common_string_range & rng) const;

    // Appends to the result.content field
    void add_content(const std::string & content);

    // Appends to the result.reasoning_content field
    void add_reasoning_content(const std::string & reasoning_content);

    // Adds a tool call to the result. If the tool call is too incomplete (e.g. name empty), it won't add anything.
    bool add_tool_call(const std::string & name, const std::string & id, const std::string & arguments);

    // Adds a tool call using the "name", "id" and "arguments" fields of the json object
    bool add_tool_call(const nlohmann::ordered_json & tool_call);

    // Adds an array of tool calls using their "name", "id" and "arguments" fields.
    bool add_tool_calls(const nlohmann::ordered_json & arr);

    void finish();

    bool consume_spaces();

    void consume_literal(const std::string & literal);

    bool try_parse_reasoning(const std::string & start_think, const std::string & end_think);

    std::string consume_rest();

    struct find_regex_result {
        std::string prelude;
        std::vector<common_string_range> groups;
    };

    std::optional<find_regex_result> try_find_regex(const common_regex & regex, size_t from = std::string::npos, bool add_prelude_to_content = true);

    bool try_consume_literal(const std::string & literal);

    std::optional<find_regex_result> try_find_literal(const std::string & literal);

    find_regex_result consume_regex(const common_regex & regex);

    std::optional<find_regex_result> try_consume_regex(const common_regex & regex);

    std::optional<common_json> try_consume_json();
    common_json consume_json();

    struct consume_json_result {
        nlohmann::ordered_json value;
        bool is_partial;
    };

    /*
        Consume (possibly partial) json and converts specific subtrees to (possibly truncated) JSON strings.

        By default, object keys can't be truncated, nor can string values (their corresponding key is removed,
        e.g. `{"foo": "bar", "baz": "b` -> `{"foo": "bar"}`

        But one can allow subpaths to be kept truncated, and possibly json-dumped to truncated json strings
        - with `content_paths={{"foo"}}` -> `{"foo": "b` -> {"foo": "b"}`
        - with `args_paths={{"foo"}}` -> `{"foo": {"b` -> `{"foo": "{b"}`
    */
    consume_json_result consume_json_with_dumped_args(
        const std::vector<std::vector<std::string>> & args_paths = {},
        const std::vector<std::vector<std::string>> & content_paths = {}
    );
    std::optional<consume_json_result> try_consume_json_with_dumped_args(
        const std::vector<std::vector<std::string>> & args_paths = {},
        const std::vector<std::vector<std::string>> & content_paths = {}
    );

    void clear_tools();
};
