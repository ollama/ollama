#include "arg.h"
#include "preset.h"
#include "peg-parser.h"
#include "log.h"

#include <fstream>
#include <sstream>
#include <filesystem>

static std::string rm_leading_dashes(const std::string & str) {
    size_t pos = 0;
    while (pos < str.size() && str[pos] == '-') {
        ++pos;
    }
    return str.substr(pos);
}

std::vector<std::string> common_preset::to_args() const {
    std::vector<std::string> args;

    for (const auto & [opt, value] : options) {
        args.push_back(opt.args.back()); // use the last arg as the main arg
        if (opt.value_hint == nullptr && opt.value_hint_2 == nullptr) {
            // flag option, no value
            if (common_arg_utils::is_falsey(value)) {
                // use negative arg if available
                if (!opt.args_neg.empty()) {
                    args.back() = opt.args_neg.back();
                } else {
                    // otherwise, skip the flag
                    // TODO: maybe throw an error instead?
                    args.pop_back();
                }
            }
        }
        if (opt.value_hint != nullptr) {
            // single value
            args.push_back(value);
        }
        if (opt.value_hint != nullptr && opt.value_hint_2 != nullptr) {
            throw std::runtime_error(string_format(
                "common_preset::to_args(): option '%s' has two values, which is not supported yet",
                opt.args.back()
            ));
        }
    }

    return args;
}

std::string common_preset::to_ini() const {
    std::ostringstream ss;

    ss << "[" << name << "]\n";
    for (const auto & [opt, value] : options) {
        auto espaced_value = value;
        string_replace_all(espaced_value, "\n", "\\\n");
        ss << rm_leading_dashes(opt.args.back()) << " = ";
        ss << espaced_value << "\n";
    }
    ss << "\n";

    return ss.str();
}

static std::map<std::string, std::map<std::string, std::string>> parse_ini_from_file(const std::string & path) {
    std::map<std::string, std::map<std::string, std::string>> parsed;

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("preset file does not exist: " + path);
    }

    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open server preset file: " + path);
    }

    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    static const auto parser = build_peg_parser([](auto & p) {
        // newline ::= "\r\n" / "\n" / "\r"
        auto newline = p.rule("newline", p.literal("\r\n") | p.literal("\n") | p.literal("\r"));

        // ws ::= [ \t]*
        auto ws = p.rule("ws", p.chars("[ \t]", 0, -1));

        // comment ::= [;#] (!newline .)*
        auto comment = p.rule("comment", p.chars("[;#]", 1, 1) + p.zero_or_more(p.negate(newline) + p.any()));

        // eol ::= ws comment? (newline / EOF)
        auto eol = p.rule("eol", ws + p.optional(comment) + (newline | p.end()));

        // ident ::= [a-zA-Z_] [a-zA-Z0-9_.-]*
        auto ident = p.rule("ident", p.chars("[a-zA-Z_]", 1, 1) + p.chars("[a-zA-Z0-9_.-]", 0, -1));

        // value ::= (!eol-start .)*
        auto eol_start = p.rule("eol-start", ws + (p.chars("[;#]", 1, 1) | newline | p.end()));
        auto value = p.rule("value", p.zero_or_more(p.negate(eol_start) + p.any()));

        // header-line ::= "[" ws ident ws "]" eol
        auto header_line = p.rule("header-line", "[" + ws + p.tag("section-name", p.chars("[^]]")) + ws + "]" + eol);

        // kv-line ::= ident ws "=" ws value eol
        auto kv_line = p.rule("kv-line", p.tag("key", ident) + ws + "=" + ws + p.tag("value", value) + eol);

        // comment-line ::= ws comment (newline / EOF)
        auto comment_line = p.rule("comment-line", ws + comment + (newline | p.end()));

        // blank-line ::= ws (newline / EOF)
        auto blank_line = p.rule("blank-line", ws + (newline | p.end()));

        // line ::= header-line / kv-line / comment-line / blank-line
        auto line = p.rule("line", header_line | kv_line | comment_line | blank_line);

        // ini ::= line* EOF
        auto ini = p.rule("ini", p.zero_or_more(line) + p.end());

        return ini;
    });

    common_peg_parse_context ctx(contents);
    const auto result = parser.parse(ctx);
    if (!result.success()) {
        throw std::runtime_error("failed to parse server config file: " + path);
    }

    std::string current_section = COMMON_PRESET_DEFAULT_NAME;
    std::string current_key;

    ctx.ast.visit(result, [&](const auto & node) {
        if (node.tag == "section-name") {
            const std::string section = std::string(node.text);
            current_section = section;
            parsed[current_section] = {};
        } else if (node.tag == "key") {
            const std::string key = std::string(node.text);
            current_key = key;
        } else if (node.tag == "value" && !current_key.empty() && !current_section.empty()) {
            parsed[current_section][current_key] = std::string(node.text);
            current_key.clear();
        }
    });

    return parsed;
}

static std::map<std::string, common_arg> get_map_key_opt(common_params_context & ctx_params) {
    std::map<std::string, common_arg> mapping;
    for (const auto & opt : ctx_params.options) {
        for (const auto & env : opt.get_env()) {
            mapping[env] = opt;
        }
        for (const auto & arg : opt.get_args()) {
            mapping[rm_leading_dashes(arg)] = opt;
        }
    }
    return mapping;
}

static bool is_bool_arg(const common_arg & arg) {
    return !arg.args_neg.empty();
}

static std::string parse_bool_arg(const common_arg & arg, const std::string & key, const std::string & value) {
    // if this is a negated arg, we need to reverse the value
    for (const auto & neg_arg : arg.args_neg) {
        if (rm_leading_dashes(neg_arg) == key) {
            return common_arg_utils::is_truthy(value) ? "false" : "true";
        }
    }
    // otherwise, not negated
    return value;
}

common_presets common_presets_load(const std::string & path, common_params_context & ctx_params) {
    common_presets out;
    auto key_to_opt = get_map_key_opt(ctx_params);
    auto ini_data = parse_ini_from_file(path);

    for (auto section : ini_data) {
        common_preset preset;
        if (section.first.empty()) {
            preset.name = COMMON_PRESET_DEFAULT_NAME;
        } else {
            preset.name = section.first;
        }
        LOG_DBG("loading preset: %s\n", preset.name.c_str());
        for (const auto & [key, value] : section.second) {
            LOG_DBG("option: %s = %s\n", key.c_str(), value.c_str());
            if (key_to_opt.find(key) != key_to_opt.end()) {
                auto & opt = key_to_opt[key];
                if (is_bool_arg(opt)) {
                    preset.options[opt] = parse_bool_arg(opt, key, value);
                } else {
                    preset.options[opt] = value;
                }
                LOG_DBG("accepted option: %s = %s\n", key.c_str(), preset.options[opt].c_str());
            } else {
                // TODO: maybe warn about unknown key?
            }
        }
        out[preset.name] = preset;
    }

    return out;
}
