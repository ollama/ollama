#include "chat.h"
#include "chat-parser.h"
#include "common.h"
#include "json-partial.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "regex-partial.h"

using json = nlohmann::ordered_json;

class xml_toolcall_syntax_exception : public std::runtime_error {
  public:
    xml_toolcall_syntax_exception(const std::string & message) : std::runtime_error(message) {}
};

template<typename T>
inline void sort_uniq(std::vector<T> &vec) {
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}

template<typename T>
inline bool all_space(const T &str) {
    return std::all_of(str.begin(), str.end(), [](unsigned char ch) { return std::isspace(ch); });
}

static size_t utf8_truncate_safe(const std::string_view s) {
    size_t len = s.size();
    if (len == 0) return 0;
    size_t i = len;
    for (size_t back = 0; back < 4 && i > 0; ++back) {
        --i;
        unsigned char c = s[i];
        if ((c & 0x80) == 0) {
            return len;
        } else if ((c & 0xC0) == 0xC0) {
            size_t expected_len = 0;
            if ((c & 0xE0) == 0xC0) expected_len = 2;
            else if ((c & 0xF0) == 0xE0) expected_len = 3;
            else if ((c & 0xF8) == 0xF0) expected_len = 4;
            else return i;
            if (len - i >= expected_len) {
                return len;
            } else {
                return i;
            }
        }
    }
    return len - std::min(len, size_t(3));
}

inline void utf8_truncate_safe_resize(std::string &s) {
    s.resize(utf8_truncate_safe(s));
}

inline std::string_view utf8_truncate_safe_view(const std::string_view s) {
    return s.substr(0, utf8_truncate_safe(s));
}

static std::optional<common_chat_msg_parser::find_regex_result> try_find_2_literal_splited_by_spaces(common_chat_msg_parser & builder, const std::string & literal1, const std::string & literal2) {
    if (literal1.size() == 0) return builder.try_find_literal(literal2);
    const auto saved_pos = builder.pos();
    while (auto res = builder.try_find_literal(literal1)) {
        builder.consume_spaces();
        const auto match_len = std::min(literal2.size(), builder.input().size() - builder.pos());
        if (builder.input().compare(builder.pos(), match_len, literal2, 0, match_len) == 0) {
            if (res->prelude.size() != res->groups[0].begin - saved_pos) {
                res->prelude = builder.str({saved_pos, res->groups[0].begin});
            }
            builder.move_to(builder.pos() + match_len);
            res->groups[0].end = builder.pos();
            GGML_ASSERT(res->groups[0].begin != res->groups[0].end);
            return res;
        }
        builder.move_to(res->groups[0].begin + 1);
    }
    builder.move_to(saved_pos);
    return std::nullopt;
}

/**
 * make a GBNF that accept any strings except those containing any of the forbidden strings.
 */
std::string make_gbnf_excluding(std::vector<std::string> forbids) {
    constexpr auto charclass_escape = [](unsigned char c) -> std::string {
        if (c == '\\' || c == ']' || c == '^' || c == '-') {
            std::string s = "\\";
            s.push_back((char)c);
            return s;
        }
        if (isprint(c)) {
            return std::string(1, (char)c);
        }
        char buf[16];
        snprintf(buf, 15, "\\x%02X", c);
        return std::string(buf);
    };
    constexpr auto build_expr = [charclass_escape](auto self, const std::vector<std::string>& forbids, int l, int r, int depth) -> std::string {
        std::vector<std::pair<unsigned char, std::pair<int,int>>> children;
        int i = l;
        while (i < r) {
            const std::string &s = forbids[i];
            if ((int)s.size() == depth) {
                ++i;
                continue;
            }
            unsigned char c = (unsigned char)s[depth];
            int j = i;
            while (j < r && (int)forbids[j].size() > depth &&
                   (unsigned char)forbids[j][depth] == c) {
                ++j;
            }
            children.push_back({c, {i, j}});
            i = j;
        }
        std::vector<std::string> alts;
        if (!children.empty()) {
            std::string cls;
            for (auto &ch : children) cls += charclass_escape(ch.first);
            alts.push_back(std::string("[^") + cls + "]");
        }
        for (auto &ch : children) {
            std::string childExpr = self(self, forbids, ch.second.first, ch.second.second, depth+1);
            if (!childExpr.empty()) {
                std::string quoted_ch = "\"";
                if (ch.first == '\\') quoted_ch += "\\\\";
                else if (ch.first == '"') quoted_ch += "\\\"";
                else if (isprint(ch.first)) quoted_ch.push_back(ch.first);
                else {
                    char buf[16];
                    snprintf(buf, 15, "\\x%02X", ch.first);
                    quoted_ch += buf;
                }
                quoted_ch += "\"";
                std::string branch = quoted_ch + std::string(" ") + childExpr;
                alts.push_back(branch);
            }
        }
        if (alts.empty()) return "";
        std::ostringstream oss;
        oss << "( ";
        for (size_t k = 0; k < alts.size(); ++k) {
            if (k) oss << " | ";
            oss << alts[k];
        }
        oss << " )";
        return oss.str();
    };
    if (forbids.empty()) return "( . )*";
    sort(forbids.begin(), forbids.end());
    std::string expr = build_expr(build_expr, forbids, 0, forbids.size(), 0);
    if (expr.empty()) {
        std::string cls;
        for (auto &s : forbids) if (!s.empty()) cls += charclass_escape((unsigned char)s[0]);
        expr = std::string("( [^") + cls + "] )";
    }
    if (forbids.size() == 1)
        return expr + "*";
    else
        return std::string("( ") + expr + " )*";
}

/**
 * Build grammar for xml-style tool call
 * form.scope_start and form.scope_end can be empty.
 * Requires data.format for model-specific hacks.
 */
void build_grammar_xml_tool_call(common_chat_params & data, const json & tools, const struct xml_tool_call_format & form) {
    GGML_ASSERT(!form.tool_start.empty());
    GGML_ASSERT(!form.tool_sep.empty());
    GGML_ASSERT(!form.key_start.empty());
    GGML_ASSERT(!form.val_end.empty());
    GGML_ASSERT(!form.tool_end.empty());

    std::string key_val_sep = form.key_val_sep;
    if (form.key_val_sep2) {
        key_val_sep += "\n";
        key_val_sep += *form.key_val_sep2;
    }
    GGML_ASSERT(!key_val_sep.empty());

    if (tools.is_array() && !tools.empty()) {
        data.grammar = build_grammar([&](const common_grammar_builder &builder) {
            auto string_arg_val = form.last_val_end ?
                    builder.add_rule("string-arg-val", make_gbnf_excluding({form.val_end, *form.last_val_end})) :
                    builder.add_rule("string-arg-val", make_gbnf_excluding({form.val_end}));

            std::vector<std::string> tool_rules;
            for (const auto & tool : tools) {
                if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
                    LOG_WRN("Skipping tool without function: %s", tool.dump(2).c_str());
                    continue;
                }
                const auto & function = tool.at("function");
                if (!function.contains("name") || !function.at("name").is_string()) {
                    LOG_WRN("Skipping invalid function (invalid name): %s", function.dump(2).c_str());
                    continue;
                }
                if (!function.contains("parameters") || !function.at("parameters").is_object()) {
                    LOG_WRN("Skipping invalid function (invalid parameters): %s", function.dump(2).c_str());
                    continue;
                }
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);

                struct parameter_rule {
                    std::string symbol_name;
                    bool is_required;
                };
                std::vector<parameter_rule> arg_rules;
                if (!parameters.contains("properties") || !parameters.at("properties").is_object()) {
                    LOG_WRN("Skipping invalid function (invalid properties): %s", function.dump(2).c_str());
                    continue;
                } else {
                    std::vector<std::string> requiredParameters;
                    if (parameters.contains("required")) {
                        try { parameters.at("required").get_to(requiredParameters); }
                        catch (const std::runtime_error&) {
                            LOG_WRN("Invalid function required parameters, ignoring: %s", function.at("required").dump(2).c_str());
                        }
                    }
                    sort_uniq(requiredParameters);
                    for (const auto & [key, value] : parameters.at("properties").items()) {
                        std::string quoted_key = key;
                        bool required = std::binary_search(requiredParameters.begin(), requiredParameters.end(), key);
                        if (form.key_start.back() == '"' && key_val_sep[0] == '"') {
                            quoted_key = gbnf_format_literal(key);
                            quoted_key = quoted_key.substr(1, quoted_key.size() - 2);
                        }
                        arg_rules.push_back(parameter_rule {builder.add_rule("func-" + name + "-kv-" + key,
                            gbnf_format_literal(form.key_start) + " " +
                            gbnf_format_literal(quoted_key) + " " +
                            gbnf_format_literal(key_val_sep) + " " +
                            ((value.contains("type") && value["type"].is_string() && value["type"] == "string" && (!form.raw_argval || *form.raw_argval)) ?
                                    (form.raw_argval ?
                                            string_arg_val :
                                            "( " + string_arg_val + " | " + builder.add_schema(name + "-arg-" + key, value) + " )"
                                    ) :
                                    builder.add_schema(name + "-arg-" + key, value)
                            )
                        ), required});
                    }
                }

                auto next_arg_with_sep = builder.add_rule(name + "-last-arg-end", form.last_val_end ? gbnf_format_literal(*form.last_val_end) : gbnf_format_literal(form.val_end));
                decltype(next_arg_with_sep) next_arg = "\"\"";
                for (auto i = arg_rules.size() - 1; /* i >= 0 && */ i < arg_rules.size(); --i) {
                    std::string include_this_arg = arg_rules[i].symbol_name + " " + next_arg_with_sep;
                    next_arg = builder.add_rule(name + "-arg-after-" + std::to_string(i), arg_rules[i].is_required ?
                            include_this_arg : "( " + include_this_arg + " ) | " + next_arg
                    );
                    include_this_arg = gbnf_format_literal(form.val_end) + " " + include_this_arg;
                    next_arg_with_sep = builder.add_rule(name + "-arg-after-" + std::to_string(i) + "-with-sep", arg_rules[i].is_required ?
                            include_this_arg : "( " + include_this_arg + " ) | " + next_arg_with_sep
                    );
                }

                std::string quoted_name = name;
                if (form.tool_start.back() == '"' && form.tool_sep[0] == '"') {
                    quoted_name = gbnf_format_literal(name);
                    quoted_name = quoted_name.substr(1, quoted_name.size() - 2);
                }
                quoted_name = gbnf_format_literal(quoted_name);
                // Kimi-K2 uses functions.{{ tool_call['function']['name'] }}:{{ loop.index }} as function name
                if (data.format == COMMON_CHAT_FORMAT_KIMI_K2) {
                    quoted_name = "\"functions.\" " + quoted_name + " \":\" [0-9]+";
                }
                tool_rules.push_back(builder.add_rule(name + "-call",
                        gbnf_format_literal(form.tool_start) + " " +
                        quoted_name + " " +
                        gbnf_format_literal(form.tool_sep) + " " +
                        next_arg
                ));
            }

            auto tool_call_once = builder.add_rule("root-tool-call-once", string_join(tool_rules, " | "));
            auto tool_call_more = builder.add_rule("root-tool-call-more", gbnf_format_literal(form.tool_end) + " " + tool_call_once);
            auto call_end = builder.add_rule("root-call-end", form.last_tool_end ? gbnf_format_literal(*form.last_tool_end) : gbnf_format_literal(form.tool_end));
            auto tool_call_multiple_with_end = builder.add_rule("root-tool-call-multiple-with-end", tool_call_once + " " + tool_call_more + "* " + call_end);
            builder.add_rule("root",
                (form.scope_start.empty() ? "" : gbnf_format_literal(form.scope_start) + " ") +
                tool_call_multiple_with_end  + "?" +
                (form.scope_end.empty() ? "" : " " + gbnf_format_literal(form.scope_end))
            );
        });

        // grammar trigger for tool call
        data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, form.scope_start + form.tool_start });
    }
}

/**
 * Parse XML-Style tool call for given xml_tool_call_format. Return false for invalid syntax and get the position untouched.
 * Throws xml_toolcall_syntax_exception if there is invalid syntax and cannot recover the original status for common_chat_msg_parser.
 * form.scope_start, form.tool_sep and form.scope_end can be empty.
 */
inline bool parse_xml_tool_calls(common_chat_msg_parser & builder, const struct xml_tool_call_format & form) {
    GGML_ASSERT(!form.tool_start.empty());
    GGML_ASSERT(!form.key_start.empty());
    GGML_ASSERT(!form.key_val_sep.empty());
    GGML_ASSERT(!form.val_end.empty());
    GGML_ASSERT(!form.tool_end.empty());

    // Helper to choose return false or throw error
    constexpr auto return_error = [](common_chat_msg_parser & builder, auto &start_pos, const bool &recovery) {
        LOG_DBG("Failed to parse XML-Style tool call at position: %s\n", gbnf_format_literal(builder.consume_rest().substr(0, 20)).c_str());
        if (recovery) {
            builder.move_to(start_pos);
            return false;
        } else throw xml_toolcall_syntax_exception("Tool call parsing failed with unrecoverable errors. Try using a grammar to constrain the model’s output.");
    };
    // Drop substring from needle to end from a JSON
    constexpr auto partial_json = [](std::string &json_str, std::string_view needle = "XML_TOOL_CALL_PARTIAL_FLAG") {
        auto pos = json_str.rfind(needle);
        if (pos == std::string::npos) {
            return false;
        }
        for (auto i = pos + needle.size(); i < json_str.size(); ++i) {
            unsigned char ch = static_cast<unsigned char>(json_str[i]);
            if (ch != '\'' && ch != '"' && ch != '}' && ch != ':' && !std::isspace(ch)) {
                return false;
            }
        }
        if (pos != 0 && json_str[pos - 1] == '"') {
            --pos;
        }
        json_str.resize(pos);
        return true;
    };
    // Helper to generate a partial argument JSON
    constexpr auto gen_partial_json = [partial_json](auto set_partial_arg, auto &arguments, auto &builder, auto &function_name) {
        auto rest = builder.consume_rest();
        utf8_truncate_safe_resize(rest);
        set_partial_arg(rest, "XML_TOOL_CALL_PARTIAL_FLAG");
        auto tool_str = arguments.dump();
        if (partial_json(tool_str)) {
            if (builder.add_tool_call(function_name, "", tool_str)) {
                return;
            }
        }
        LOG_DBG("Failed to parse partial XML-Style tool call, fallback to non-partial: %s\n", tool_str.c_str());
    };
    // Helper to find a close (because there may be form.last_val_end or form.last_tool_end)
    constexpr auto try_find_close = [](
            common_chat_msg_parser & builder,
            const std::string & end,
            const std::optional<std::string> & alt_end,
            const std::string & end_next,
            const std::optional<std::string> & alt_end_next
    ) {
        auto saved_pos = builder.pos();
        auto tc = builder.try_find_literal(end);
        auto val_end_size = end.size();
        if (alt_end) {
            auto pos_1 = builder.pos();
            builder.move_to(saved_pos);
            auto tc2 = try_find_2_literal_splited_by_spaces(builder, *alt_end, end_next);
            if (alt_end_next) {
                builder.move_to(saved_pos);
                auto tc3 = try_find_2_literal_splited_by_spaces(builder, *alt_end, *alt_end_next);
                if (tc3 && (!tc2 || tc2->prelude.size() > tc3->prelude.size())) {
                    tc2 = tc3;
                }
            }
            if (tc2 && (!tc || tc->prelude.size() > tc2->prelude.size())) {
                tc = tc2;
                tc->groups[0].end = std::min(builder.input().size(), tc->groups[0].begin + alt_end->size());
                builder.move_to(tc->groups[0].end);
                val_end_size = alt_end->size();
            } else {
                builder.move_to(pos_1);
            }
        }
        return std::make_pair(val_end_size, tc);
    };
    // Helper to find a val_end or last_val_end, returns matched pattern size
    const auto try_find_val_end = [try_find_close, &builder, &form]() {
        return try_find_close(builder, form.val_end, form.last_val_end, form.tool_end, form.last_tool_end);
    };
    // Helper to find a tool_end or last_tool_end, returns matched pattern size
    const auto try_find_tool_end = [try_find_close, &builder, &form]() {
        return try_find_close(builder, form.tool_end, form.last_tool_end, form.scope_end, std::nullopt);
    };

    bool recovery = true;
    const auto start_pos = builder.pos();
    if (!all_space(form.scope_start)) {
        if (auto tc = builder.try_find_literal(form.scope_start)) {
            if (all_space(tc->prelude)) {
                if (form.scope_start.size() != tc->groups[0].end - tc->groups[0].begin)
                    throw common_chat_msg_partial_exception("Partial literal: " + gbnf_format_literal(form.scope_start));
            } else {
                builder.move_to(start_pos);
                return false;
            }
        } else return false;
    }
    while (auto tc = builder.try_find_literal(form.tool_start)) {
        if (!all_space(tc->prelude)) {
            LOG_DBG("XML-Style tool call: Expected %s, but found %s, trying to match next pattern\n",
                    gbnf_format_literal(form.tool_start).c_str(),
                    gbnf_format_literal(tc->prelude).c_str()
            );
            builder.move_to(tc->groups[0].begin - tc->prelude.size());
            break;
        }

        // Find tool name
        auto func_name = builder.try_find_literal(all_space(form.tool_sep) ? form.key_start : form.tool_sep);
        if (!func_name) {
            auto [sz, tc] = try_find_tool_end();
            func_name = tc;
        }
        if (!func_name) {
            // Partial tool name not supported
            throw common_chat_msg_partial_exception("incomplete tool_call");
        }
        // If the model generate multiple tool call and the first tool call has no argument
        if (func_name->prelude.find(form.tool_end) != std::string::npos || (form.last_tool_end ? func_name->prelude.find(*form.last_tool_end) != std::string::npos : false)) {
            builder.move_to(func_name->groups[0].begin - func_name->prelude.size());
            auto [sz, tc] = try_find_tool_end();
            func_name = tc;
        }

        // Parse tool name
        builder.move_to(all_space(form.tool_sep) ? func_name->groups[0].begin : func_name->groups[0].end);
        std::string function_name = string_strip(func_name->prelude);
        // Kimi-K2 uses functions.{{ tool_call['function']['name'] }}:{{ loop.index }} as function name
        if (builder.syntax().format == COMMON_CHAT_FORMAT_KIMI_K2) {
            if (string_starts_with(function_name, "functions.")) {
                static const std::regex re(":\\d+$");
                if (std::regex_search(function_name, re)) {
                    function_name = function_name.substr(10, function_name.rfind(":") - 10);
                }
            }
        }

        // Argument JSON
        json arguments = json::object();

        // Helper to generate a partial argument JSON
        const auto gen_partial_args = [&](auto set_partial_arg) {
            gen_partial_json(set_partial_arg, arguments, builder, function_name);
        };

        // Parse all arg_key/arg_value pairs
        while (auto tc = builder.try_find_literal(form.key_start)) {
            if (!all_space(tc->prelude)) {
                LOG_DBG("XML-Style tool call: Expected %s, but found %s, trying to match next pattern\n",
                        gbnf_format_literal(form.key_start).c_str(),
                        gbnf_format_literal(tc->prelude).c_str()
                );
                builder.move_to(tc->groups[0].begin - tc->prelude.size());
                break;
            }
            if (tc->groups[0].end - tc->groups[0].begin != form.key_start.size()) {
                auto tool_call_arg = arguments.dump();
                if (tool_call_arg.size() != 0 && tool_call_arg[tool_call_arg.size() - 1] == '}') {
                    tool_call_arg.resize(tool_call_arg.size() - 1);
                }
                builder.add_tool_call(function_name, "", tool_call_arg);
                throw common_chat_msg_partial_exception("Partial literal: " + gbnf_format_literal(form.key_start));
            }

            // Parse arg_key
            auto key_res = builder.try_find_literal(form.key_val_sep);
            if (!key_res) {
                gen_partial_args([&](auto &rest, auto &needle) {arguments[rest + needle] = "";});
                throw common_chat_msg_partial_exception("Expected " + gbnf_format_literal(form.key_val_sep) + " after " + gbnf_format_literal(form.key_start));
            }
            if (key_res->groups[0].end - key_res->groups[0].begin != form.key_val_sep.size()) {
                gen_partial_args([&](auto &, auto &needle) {arguments[key_res->prelude + needle] = "";});
                throw common_chat_msg_partial_exception("Partial literal: " + gbnf_format_literal(form.key_val_sep));
            }
            auto &key = key_res->prelude;
            recovery = false;

            // Parse arg_value
            if (form.key_val_sep2) {
                if (auto tc = builder.try_find_literal(*form.key_val_sep2)) {
                    if (!all_space(tc->prelude)) {
                        LOG_DBG("Failed to parse XML-Style tool call: Unexcepted %s between %s and %s\n",
                                gbnf_format_literal(tc->prelude).c_str(),
                                gbnf_format_literal(form.key_val_sep).c_str(),
                                gbnf_format_literal(*form.key_val_sep2).c_str()
                        );
                        return return_error(builder, start_pos, false);
                    }
                    if (tc->groups[0].end - tc->groups[0].begin != form.key_val_sep2->size()) {
                        gen_partial_args([&](auto &, auto &needle) {arguments[key] = needle;});
                        throw common_chat_msg_partial_exception("Partial literal: " + gbnf_format_literal(*form.key_val_sep2));
                    }
                } else {
                    gen_partial_args([&](auto &, auto &needle) {arguments[key] = needle;});
                    throw common_chat_msg_partial_exception("Expected " + gbnf_format_literal(*form.key_val_sep2) + " after " + gbnf_format_literal(form.key_val_sep));
                }
            }
            auto val_start = builder.pos();

            // Test if arg_val is a partial JSON
            std::optional<common_json> value_json = std::nullopt;
            if (!form.raw_argval || !*form.raw_argval) {
                try { value_json = builder.try_consume_json(); }
                catch (const std::runtime_error&) { builder.move_to(val_start); }
                // TODO: Delete this when json_partial adds top-level support for null/true/false
                if (builder.pos() == val_start) {
                    const static std::regex number_regex(R"([0-9-][0-9]*(\.\d*)?([eE][+-]?\d*)?)");
                    builder.consume_spaces();
                    std::string_view sv = utf8_truncate_safe_view(builder.input());
                    sv.remove_prefix(builder.pos());
                    std::string rest = "a";
                    if (sv.size() < 6) rest = sv;
                    if (string_starts_with("null", rest) || string_starts_with("true", rest) || string_starts_with("false", rest) || std::regex_match(sv.begin(), sv.end(), number_regex)) {
                        value_json = {123, {"123", "123"}};
                        builder.consume_rest();
                    } else {
                        builder.move_to(val_start);
                    }
                }
            }

            // If it is a JSON and followed by </arg_value>, parse as json
            // cannot support streaming because it may be a plain text starting with JSON
            if (value_json) {
                auto json_end = builder.pos();
                builder.consume_spaces();
                if (builder.pos() == builder.input().size()) {
                    if (form.raw_argval && !*form.raw_argval && (value_json->json.is_string() || value_json->json.is_object() || value_json->json.is_array())) {
                        arguments[key] = value_json->json;
                        auto json_str = arguments.dump();
                        if (!value_json->healing_marker.json_dump_marker.empty()) {
                            GGML_ASSERT(std::string::npos != json_str.rfind(value_json->healing_marker.json_dump_marker));
                            json_str.resize(json_str.rfind(value_json->healing_marker.json_dump_marker));
                        } else {
                            GGML_ASSERT(json_str.back() == '}');
                            json_str.resize(json_str.size() - 1);
                        }
                        builder.add_tool_call(function_name, "", json_str);
                    } else {
                        gen_partial_args([&](auto &, auto &needle) {arguments[key] = needle;});
                    }
                    LOG_DBG("Possible JSON arg_value: %s\n", value_json->json.dump().c_str());
                    throw common_chat_msg_partial_exception("JSON arg_value detected. Waiting for more tokens for validations.");
                }
                builder.move_to(json_end);
                auto [val_end_size, tc] = try_find_val_end();
                if (tc && all_space(tc->prelude) && value_json->healing_marker.marker.empty()) {
                    if (tc->groups[0].end - tc->groups[0].begin != val_end_size) {
                        gen_partial_args([&](auto &, auto &needle) {arguments[key] = needle;});
                        LOG_DBG("Possible terminated JSON arg_value: %s\n", value_json->json.dump().c_str());
                        throw common_chat_msg_partial_exception("Partial literal: " + gbnf_format_literal(form.val_end) + (form.last_val_end ? gbnf_format_literal(*form.last_val_end) : ""));
                    } else arguments[key] = value_json->json;
                } else builder.move_to(val_start);
            }

            // If not, parse as plain text
            if (val_start == builder.pos()) {
                if (auto [val_end_size, value_plain] = try_find_val_end(); value_plain) {
                    auto &value_str = value_plain->prelude;
                    if (form.trim_raw_argval) value_str = string_strip(value_str);
                    if (value_plain->groups[0].end - value_plain->groups[0].begin != val_end_size) {
                        gen_partial_args([&](auto &, auto &needle) {arguments[key] = value_str + needle;});
                        throw common_chat_msg_partial_exception(
                                "Expected " + gbnf_format_literal(form.val_end) +
                                " after " + gbnf_format_literal(form.key_val_sep) +
                                (form.key_val_sep2 ? " " + gbnf_format_literal(*form.key_val_sep2) : "")
                        );
                    }
                    arguments[key] = value_str;
                } else {
                    if (form.trim_raw_argval) {
                        gen_partial_args([&](auto &rest, auto &needle) {arguments[key] = string_strip(rest) + needle;});
                    } else {
                        gen_partial_args([&](auto &rest, auto &needle) {arguments[key] = rest + needle;});
                    }
                    throw common_chat_msg_partial_exception(
                            "Expected " + gbnf_format_literal(form.val_end) +
                            " after " + gbnf_format_literal(form.key_val_sep) +
                            (form.key_val_sep2 ? " " + gbnf_format_literal(*form.key_val_sep2) : "")
                    );
                }
            }
        }

        // Consume closing tag
        if (auto [tool_end_size, tc] = try_find_tool_end(); tc) {
            if (!all_space(tc->prelude)) {
                LOG_DBG("Failed to parse XML-Style tool call: Expected %s, but found %s\n",
                        gbnf_format_literal(form.tool_end).c_str(),
                        gbnf_format_literal(tc->prelude).c_str()
                );
                return return_error(builder, start_pos, recovery);
            }
            if (tc->groups[0].end - tc->groups[0].begin == tool_end_size) {
                // Add the parsed tool call
                if (!builder.add_tool_call(function_name, "", arguments.dump())) {
                    throw common_chat_msg_partial_exception("Failed to add XML-Style tool call");
                }
                recovery = false;
                continue;
            }
        }

        auto tool_call_arg = arguments.dump();
        if (tool_call_arg.size() != 0 && tool_call_arg[tool_call_arg.size() - 1] == '}') {
            tool_call_arg.resize(tool_call_arg.size() - 1);
        }
        builder.add_tool_call(function_name, "", tool_call_arg);
        throw common_chat_msg_partial_exception("Expected " + gbnf_format_literal(form.tool_end) + " after " + gbnf_format_literal(form.val_end));
    }
    if (auto tc = builder.try_find_literal(form.scope_end)) {
        if (!all_space(tc->prelude)) {
            LOG_DBG("Failed to parse XML-Style tool call: Expected %s, but found %s\n",
                    gbnf_format_literal(form.scope_end).c_str(),
                    gbnf_format_literal(tc->prelude).c_str()
            );
            return return_error(builder, start_pos, recovery);
        }
    } else {
        if (all_space(form.scope_end)) return true;
        builder.consume_spaces();
        if (builder.pos() == builder.input().size())
            throw common_chat_msg_partial_exception("incomplete tool calls");
        LOG_DBG("Failed to parse XML-Style tool call: Expected %s, but found %s\n",
                gbnf_format_literal(form.scope_end).c_str(),
                gbnf_format_literal(builder.consume_rest()).c_str()
        );
        return return_error(builder, start_pos, recovery);
    }

    return true;
}

/**
 * Parse XML-Style tool call for given xml_tool_call_format. Return false for invalid syntax and get the position untouched.
 * May cause std::runtime_error if there is invalid syntax because partial valid tool call is already sent out to client.
 * form.scope_start, form.tool_sep and form.scope_end can be empty.
 */
bool common_chat_msg_parser::try_consume_xml_tool_calls(const struct xml_tool_call_format & form) {
    auto pos = pos_;
    auto tsize = result_.tool_calls.size();
    try { return parse_xml_tool_calls(*this, form); }
    catch (const xml_toolcall_syntax_exception&) {}
    move_to(pos);
    result_.tool_calls.resize(tsize);
    return false;
}

/**
 * Parse content uses reasoning and XML-Style tool call
 * TODO: Note that form.allow_toolcall_in_think is not tested yet. If anyone confirms it works, this comment can be removed.
 */
inline void parse_msg_with_xml_tool_calls(common_chat_msg_parser & builder, const struct xml_tool_call_format & form, const std::string & start_think = "<think>", const std::string & end_think = "</think>") {
    constexpr auto rstrip = [](std::string &s) {
        s.resize(std::distance(s.begin(), std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base()));
    };
    // Erase substring from l to r, along with additional spaces nearby
    constexpr auto erase_spaces = [](auto &str, size_t l, size_t r) {
        while (/* l > -1 && */ --l < str.size() && std::isspace(static_cast<unsigned char>(str[l])));
        ++l;
        while (++r < str.size() && std::isspace(static_cast<unsigned char>(str[r])));
        if (l < r) str[l] = '\n';
        if (l + 1 < r) str[l + 1] = '\n';
        if (l != 0) l += 2;
        str.erase(l, r - l);
        return l;
    };
    constexpr auto trim_suffix = [](std::string &content, std::initializer_list<std::string_view> list) {
        auto best_match = content.size();
        for (auto pattern: list) {
            if (pattern.size() == 0) continue;
            for (auto match_idx = content.size() - std::min(pattern.size(), content.size()); content.size() > match_idx; match_idx++) {
                auto match_len = content.size() - match_idx;
                if (content.compare(match_idx, match_len, pattern.data(), match_len) == 0 && best_match > match_idx) {
                    best_match = match_idx;
                }
            }
        }
        if (content.size() > best_match) {
            content.erase(best_match);
        }
    };
    const auto trim_potential_partial_word = [&start_think, &end_think, &form, trim_suffix](std::string &content) {
        return trim_suffix(content, {
            start_think, end_think, form.scope_start, form.tool_start, form.tool_sep, form.key_start,
            form.key_val_sep, form.key_val_sep2 ? form.key_val_sep2->c_str() : "",
            form.val_end, form.last_val_end ? form.last_val_end->c_str() : "",
            form.tool_end, form.last_tool_end ? form.last_tool_end->c_str() : "",
            form.scope_end
        });
    };


    // Trim leading spaces without affecting keyword matching
    static const common_regex spaces_regex("\\s*");
    {
        auto tc = builder.consume_regex(spaces_regex);
        auto spaces = builder.str(tc.groups[0]);
        auto s1 = spaces.size();
        trim_potential_partial_word(spaces);
        auto s2 = spaces.size();
        builder.move_to(builder.pos() - (s1 - s2));
    }

    // Parse content
    bool reasoning_unclosed = builder.syntax().thinking_forced_open;
    std::string unclosed_reasoning_content("");
    for (;;) {
        auto tc = try_find_2_literal_splited_by_spaces(builder, form.scope_start, form.tool_start);
        std::string content;
        std::string tool_call_start;

        if (tc) {
            content = std::move(tc->prelude);
            tool_call_start = builder.str(tc->groups[0]);
            LOG_DBG("Matched tool start: %s\n", gbnf_format_literal(tool_call_start).c_str());
        } else {
            content = builder.consume_rest();
            utf8_truncate_safe_resize(content);
        }

        // Handle unclosed think block
        if (reasoning_unclosed) {
            if (auto pos = content.find(end_think); pos == std::string::npos && builder.pos() != builder.input().size()) {
                unclosed_reasoning_content += content;
                if (!(form.allow_toolcall_in_think && tc)) {
                    unclosed_reasoning_content += tool_call_start;
                    continue;
                }
            } else {
                reasoning_unclosed = false;
                std::string reasoning_content;
                if (pos == std::string::npos) {
                    reasoning_content = std::move(content);
                } else {
                    reasoning_content = content.substr(0, pos);
                    content.erase(0, pos + end_think.size());
                }
                if (builder.pos() == builder.input().size() && all_space(content)) {
                    rstrip(reasoning_content);
                    trim_potential_partial_word(reasoning_content);
                    rstrip(reasoning_content);
                    if (reasoning_content.empty()) {
                        rstrip(unclosed_reasoning_content);
                        trim_potential_partial_word(unclosed_reasoning_content);
                        rstrip(unclosed_reasoning_content);
                        if (unclosed_reasoning_content.empty()) continue;
                    }
                }
                if (builder.syntax().reasoning_format == COMMON_REASONING_FORMAT_NONE || builder.syntax().reasoning_in_content) {
                    builder.add_content(start_think);
                    builder.add_content(unclosed_reasoning_content);
                    builder.add_content(reasoning_content);
                    if (builder.pos() != builder.input().size() || !all_space(content))
                        builder.add_content(end_think);
                } else {
                    builder.add_reasoning_content(unclosed_reasoning_content);
                    builder.add_reasoning_content(reasoning_content);
                }
                unclosed_reasoning_content.clear();
            }
        }

        // Handle multiple think block
        bool toolcall_in_think = false;
        for (auto think_start = content.find(start_think); think_start != std::string::npos; think_start = content.find(start_think, think_start)) {
            if (auto think_end = content.find(end_think, think_start + start_think.size()); think_end != std::string::npos) {
                if (builder.syntax().reasoning_format != COMMON_REASONING_FORMAT_NONE && !builder.syntax().reasoning_in_content) {
                    auto reasoning_content = content.substr(think_start + start_think.size(), think_end - think_start - start_think.size());
                    builder.add_reasoning_content(reasoning_content);
                    think_start = erase_spaces(content, think_start, think_end + end_think.size() - 1);
                } else {
                    think_start = think_end + end_think.size() - 1;
                }
            } else {
                // This <tool_call> start is in thinking block, skip this tool call
                // This <tool_call> start is in thinking block
                if (form.allow_toolcall_in_think) {
                    unclosed_reasoning_content = content.substr(think_start + start_think.size());
                } else {
                    unclosed_reasoning_content = content.substr(think_start + start_think.size()) + tool_call_start;
                }
                reasoning_unclosed = true;
                content.resize(think_start);
                toolcall_in_think = true;
            }
        }

        if (builder.syntax().reasoning_format != COMMON_REASONING_FORMAT_NONE && !builder.syntax().reasoning_in_content) {
            rstrip(content);
            // Handle unclosed </think> token from content: delete all </think> token
            if (auto pos = content.rfind(end_think); pos != std::string::npos) {
                while (pos != std::string::npos) {
                    pos = erase_spaces(content, pos, pos + end_think.size() - 1);
                    pos = content.rfind(end_think, pos);
                }
            }
            // Strip if needed
            if (content.size() > 0 && std::isspace(static_cast<unsigned char>(content[0]))) {
                content = string_strip(content);
            }
        }

        // remove potential partial suffix
        if (builder.pos() == builder.input().size()) {
            if (unclosed_reasoning_content.empty()) {
                rstrip(content);
                trim_potential_partial_word(content);
                rstrip(content);
            } else {
                rstrip(unclosed_reasoning_content);
                trim_potential_partial_word(unclosed_reasoning_content);
                rstrip(unclosed_reasoning_content);
            }
        }

        // consume unclosed_reasoning_content if allow_toolcall_in_think is set
        if (form.allow_toolcall_in_think && !unclosed_reasoning_content.empty()) {
            if (builder.syntax().reasoning_format != COMMON_REASONING_FORMAT_NONE && !builder.syntax().reasoning_in_content) {
                builder.add_reasoning_content(unclosed_reasoning_content);
            } else {
                if (content.empty()) {
                    content = start_think + unclosed_reasoning_content;
                } else {
                    content += "\n\n" + start_think;
                    content += unclosed_reasoning_content;
                }
            }
            unclosed_reasoning_content.clear();
        }

        // Add content
        if (!content.empty()) {
            // If there are multiple content blocks
            if (builder.syntax().reasoning_format != COMMON_REASONING_FORMAT_NONE && !builder.syntax().reasoning_in_content && builder.result().content.size() != 0) {
                builder.add_content("\n\n");
            }
            builder.add_content(content);
        }

        // This <tool_call> start is in thinking block and toolcall_in_think not set, skip this tool call
        if (toolcall_in_think && !form.allow_toolcall_in_think) {
            continue;
        }

        // There is no tool call and all content is parsed
        if (!tc) {
            GGML_ASSERT(builder.pos() == builder.input().size());
            GGML_ASSERT(unclosed_reasoning_content.empty());
            if (!form.allow_toolcall_in_think) GGML_ASSERT(!reasoning_unclosed);
            break;
        }

        builder.move_to(tc->groups[0].begin);
        if (builder.try_consume_xml_tool_calls(form)) {
            auto end_of_tool = builder.pos();
            builder.consume_spaces();
            if (builder.pos() != builder.input().size()) {
                builder.move_to(end_of_tool);
                if (!builder.result().content.empty()) {
                    builder.add_content("\n\n");
                }
            }
        } else {
            static const common_regex next_char_regex(".");
            auto c = builder.str(builder.consume_regex(next_char_regex).groups[0]);
            rstrip(c);
            builder.add_content(c);
        }
    }
}

/**
 * Parse content uses reasoning and XML-Style tool call
 */
void common_chat_msg_parser::consume_reasoning_with_xml_tool_calls(const struct xml_tool_call_format & form, const std::string & start_think, const std::string & end_think) {
    parse_msg_with_xml_tool_calls(*this, form, start_think, end_think);
}
