#pragma once

#include "chat.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>


// Sample config:
// MiniMax-M2 (left): <minimax:tool_call>\n<invoke name="tool-name">\n<parameter name="key">value</parameter>\n...</invoke>\n...</minimax:tool_call>
// GLM 4.5   (right): <tool_call>function_name\n<arg_key>key</arg_key>\n<arg_value>value</arg_value>\n</tool_call>
struct xml_tool_call_format {
    std::string scope_start; // <minimax:tool_call>\n  // \n                      // can be empty
    std::string tool_start;  // <invoke name=\"        // <tool_call>
    std::string tool_sep;    // \">\n                  // \n                      // can be empty only for parse_xml_tool_calls
    std::string key_start;   // <parameter name=\"     // <arg_key>
    std::string key_val_sep; // \">                    // </arg_key>\n<arg_value>
    std::string val_end;     // </parameter>\n         // </arg_value>\n
    std::string tool_end;    // </invoke>\n            // </tool_call>\n
    std::string scope_end;   // </minimax:tool_call>   //                         // can be empty
    // Set this if there can be dynamic spaces inside key_val_sep.
    // e.g. key_val_sep=</arg_key> key_val_sep2=<arg_value> for GLM4.5
    std::optional<std::string> key_val_sep2 = std::nullopt;
    // Set true if argval should only be raw string. e.g. Hello "world" hi
    // Set false if argval should only be json string. e.g. "Hello \"world\" hi"
    // Defaults to std::nullopt, both will be allowed.
    std::optional<bool> raw_argval = std::nullopt;
    std::optional<std::string> last_val_end = std::nullopt;
    std::optional<std::string> last_tool_end = std::nullopt;
    bool trim_raw_argval = false;
    bool allow_toolcall_in_think = false;
};

// make a GBNF that accept any strings except those containing any of the forbidden strings.
std::string make_gbnf_excluding(std::vector<std::string> forbids);

/**
 * Build grammar for xml-style tool call
 * form.scope_start and form.scope_end can be empty.
 * Requires data.format for model-specific hacks.
 */
void build_grammar_xml_tool_call(common_chat_params & data, const nlohmann::ordered_json & tools, const struct xml_tool_call_format & form);
