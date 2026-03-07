// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#include "sampling.h"
#include "sampling_ext.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-grammar.h"
#include "nlohmann/json.hpp"

#include <map>
#include <set>
#include <cstring>

struct common_sampler *common_sampler_cinit(const struct llama_model *model, struct common_sampler_cparams *params) {
    try {
        common_params_sampling sparams;
        sparams.top_k = params->top_k;
        sparams.top_p = params->top_p;
        sparams.min_p = params->min_p;
        sparams.typ_p = params->typical_p;
        sparams.temp = params->temp;
        sparams.penalty_last_n = params->penalty_last_n;
        sparams.penalty_repeat = params->penalty_repeat;
        sparams.penalty_freq = params->penalty_freq;
        sparams.penalty_present = params->penalty_present;
        sparams.seed = params->seed;
        sparams.grammar = params->grammar;
        sparams.xtc_probability = 0.0;
        sparams.xtc_threshold = 0.5;
        return common_sampler_init(model, sparams);
    } catch (const std::exception &err) {
        return nullptr;
    }
}

void common_sampler_cfree(struct common_sampler *sampler) {
    common_sampler_free(sampler);
}

void common_sampler_creset(struct common_sampler *sampler) {
    common_sampler_reset(sampler);
}

void common_sampler_caccept(struct common_sampler *sampler, llama_token id, bool apply_grammar) {
    common_sampler_accept(sampler, id, apply_grammar);
}

llama_token common_sampler_csample(struct common_sampler *sampler, struct llama_context *ctx, int idx) {
    return common_sampler_sample(sampler, ctx, idx);
}

int schema_to_grammar(const char *json_schema, char *grammar, size_t max_len)
{
    try
    {
        nlohmann::ordered_json schema = nlohmann::ordered_json::parse(json_schema);
        std::string grammar_str = json_schema_to_grammar(schema);
        size_t len = grammar_str.length();
        if (len >= max_len)
        {
            len = max_len - 1;
        }
        strncpy(grammar, grammar_str.c_str(), len);
        return len;
    }
    catch (const std::exception &e)
    {
        strncpy(grammar, "", max_len - 1);
        return 0;
    }
}

struct llama_vocab * llama_load_vocab_from_file(const char * fname) {
    llama_vocab * vocab = new llama_vocab();
    try {
        const auto kv = LLM_KV(LLM_ARCH_UNKNOWN);
        std::vector<std::string> splits = {};
        llama_model_loader ml(std::string(fname), splits, false, false, false, nullptr, nullptr);
        vocab->load(ml, kv);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return nullptr;
    }

    return vocab;
}

void llama_free_vocab(struct llama_vocab * vocab) {
    delete vocab;
}
struct llama_grammar *grammar_init(char* grammar, uint32_t* tokens, size_t n_tokens, const char** pieces, uint32_t* eog_tokens, size_t n_eog_tokens) {
    try {
        if (grammar == nullptr) {
            LLAMA_LOG_ERROR("%s: null grammar input\n", __func__);
            return nullptr;
        }

        ollama_vocab *vocab = new ollama_vocab();
        vocab->set_eog_tokens(eog_tokens, n_eog_tokens);
        vocab->add_token_pieces(tokens, n_tokens, pieces);

        struct llama_grammar *g = llama_grammar_init_impl(nullptr, vocab, grammar, "root", false, nullptr, 0, nullptr, 0);
        if (g == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize grammar\n", __func__);
            delete vocab;
            return nullptr;
        }
        return g;

    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception during initialization: %s\n", __func__, e.what());
        return nullptr;
    }
}

void grammar_free(struct llama_grammar *g) {
    if (g != nullptr) {
        if (g->vocab != nullptr) {
            delete g->vocab;
        }
        if (g->o_vocab != nullptr) {
                delete g->o_vocab;
        }
        llama_grammar_free_impl(g);
    }
}

void grammar_apply(struct llama_grammar *g, struct llama_token_data_array *tokens) {
    if (g == nullptr || tokens == nullptr) {
        LLAMA_LOG_ERROR("%s: null grammar or tokens input\n", __func__);
        return;
    }
    llama_grammar_apply_impl(*g, tokens);
}


void grammar_accept(struct llama_grammar *g, llama_token id) {
    llama_grammar_accept_impl(*g, id);
}

// =========================================================================
// Tool Call Grammar Builder — internals
// =========================================================================
//
// Ported from llama.cpp peg-parser.cpp (trie + gbnf_excluding_pattern) and
// chat.cpp (tool call grammar construction).  This is a self-contained
// implementation that uses only the public build_grammar() / add_rule() /
// gbnf_format_literal() API from json-schema-to-grammar.h.

namespace {

using json = nlohmann::ordered_json;

// ---------- Trie for GBNF exclusion patterns ----------
//
// Given a set of delimiter strings, produces a GBNF pattern that matches any
// character sequence NOT containing any of the delimiters.  Identical to the
// trie + gbnf_excluding_pattern in peg-parser.cpp, inlined here so we don't
// depend on the PEG parser library.

struct exclusion_trie {
    struct node {
        std::map<unsigned char, size_t> children;
        bool is_word = false;
    };

    std::vector<node> nodes;

    explicit exclusion_trie(const std::vector<std::string> & words) {
        nodes.emplace_back();  // root
        for (const auto & w : words) {
            if (w.empty()) {
                continue;
            }
            size_t cur = 0;
            for (unsigned char ch : w) {
                auto it = nodes[cur].children.find(ch);
                if (it == nodes[cur].children.end()) {
                    size_t child = nodes.size();
                    nodes.emplace_back();
                    nodes[cur].children[ch] = child;
                    cur = child;
                } else {
                    cur = it->second;
                }
            }
            nodes[cur].is_word = true;
        }
    }

    struct prefix_and_next {
        std::string prefix;
        std::string next_chars;
    };

    std::vector<prefix_and_next> collect() const {
        std::vector<prefix_and_next> result;
        std::string prefix;
        collect_impl(0, prefix, result);
        return result;
    }

private:
    void collect_impl(size_t idx, std::string & prefix,
                      std::vector<prefix_and_next> & out) const {
        const auto & n = nodes[idx];
        if (!n.is_word && !n.children.empty()) {
            std::string chars;
            chars.reserve(n.children.size());
            for (const auto & kv : n.children) {
                chars.push_back(static_cast<char>(kv.first));
            }
            out.push_back({prefix, chars});
        }
        for (const auto & kv : n.children) {
            prefix.push_back(static_cast<char>(kv.first));
            collect_impl(kv.second, prefix, out);
            prefix.pop_back();
        }
    }
};

static std::string escape_char_class(char c) {
    switch (c) {
        case '\n': return "\\n";
        case '\t': return "\\t";
        case '\r': return "\\r";
        case '\\': return "\\\\";
        case ']':  return "\\]";
        case '[':  return "\\[";
        default:   return std::string(1, c);
    }
}

// Build a GBNF pattern that matches any text NOT containing any of the
// given delimiter strings.  Zero-width match is possible (the star).
static std::string gbnf_excluding(const std::vector<std::string> & delimiters) {
    exclusion_trie trie(delimiters);
    auto pieces = trie.collect();

    std::string pattern;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i > 0) {
            pattern += " | ";
        }
        const auto & pre  = pieces[i].prefix;
        const auto & chars = pieces[i].next_chars;

        std::string cls;
        for (char ch : chars) {
            cls += escape_char_class(ch);
        }
        if (!pre.empty()) {
            pattern += gbnf_format_literal(pre) + " [^" + cls + "]";
        } else {
            pattern += "[^" + cls + "]";
        }
    }
    return "(" + pattern + ")*";
}

// ---------- Tool JSON validation ----------

static std::string validate_tool_json(const json & tool, size_t index) {
    const std::string idx = "tools[" + std::to_string(index) + "]";

    if (!tool.is_object()) {
        return idx + ": not an object";
    }
    if (!tool.contains("function")) {
        return idx + ": missing 'function'";
    }
    const auto & func = tool["function"];
    if (!func.is_object()) {
        return idx + ": 'function' is not an object";
    }
    if (!func.contains("name")) {
        return idx + ": function missing 'name'";
    }
    if (!func["name"].is_string()) {
        return idx + ": function 'name' is not a string";
    }
    const std::string name = func["name"];
    if (name.empty()) {
        return idx + ": function name is empty";
    }
    if (name.find('\0') != std::string::npos) {
        return idx + ": function name contains null byte";
    }
    // Reject names with characters that break XML tag syntax.
    // The Qwen format uses <function=NAME> — the name ends at '>'.
    for (char c : name) {
        if (c == '>' || c == '<' || c == '\n' || c == '\r') {
            return idx + ": function name contains forbidden character (one of > < \\n \\r)";
        }
    }

    if (func.contains("parameters")) {
        const auto & params = func["parameters"];
        if (!params.is_object()) {
            return idx + ": 'parameters' is not an object";
        }
        if (params.contains("properties")) {
            const auto & props = params["properties"];
            if (!props.is_object()) {
                return idx + ": 'properties' is not an object";
            }
            std::set<std::string> seen;
            for (auto it = props.begin(); it != props.end(); ++it) {
                const std::string & pname = it.key();
                if (pname.empty()) {
                    return idx + ": empty parameter name";
                }
                if (pname.find('\0') != std::string::npos) {
                    return idx + ": parameter name '" + pname + "' contains null byte";
                }
                for (char c : pname) {
                    if (c == '>' || c == '<' || c == '\n' || c == '\r') {
                        return idx + ": parameter name '" + pname +
                               "' contains forbidden character (one of > < \\n \\r)";
                    }
                }
                if (!seen.insert(pname).second) {
                    return idx + ": duplicate parameter '" + pname + "'";
                }
            }
        }

        // Validate required array if present
        if (params.contains("required")) {
            const auto & req = params["required"];
            if (!req.is_array()) {
                return idx + ": 'required' is not an array";
            }
            for (size_t r = 0; r < req.size(); ++r) {
                if (!req[r].is_string()) {
                    return idx + ": required[" + std::to_string(r) + "] is not a string";
                }
                // Verify each required parameter actually exists in properties
                if (params.contains("properties")) {
                    const std::string rname = req[r];
                    if (!params["properties"].contains(rname)) {
                        return idx + ": required parameter '" + rname +
                               "' not found in properties";
                    }
                }
            }
        }
    }
    return "";  // valid
}

// ---------- Write error helper ----------

static void write_error(char * error_out, size_t error_max_len,
                        const std::string & msg) {
    if (error_out == nullptr || error_max_len == 0) {
        return;
    }
    size_t len = msg.size();
    if (len >= error_max_len) {
        len = error_max_len - 1;
    }
    std::memcpy(error_out, msg.c_str(), len);
    error_out[len] = '\0';
}

}  // anonymous namespace

// =========================================================================
// tool_call_grammar_from_json — public C bridge
// =========================================================================

int tool_call_grammar_from_json(
        const char  * tools_json,
        char        * grammar_out,
        size_t        grammar_max_len,
        char        * error_out,
        size_t        error_max_len)
{
    auto err = [&](int code, const std::string & msg) -> int {
        write_error(error_out, error_max_len, msg);
        return code;
    };

    // ---- pointer / length checks ----

    if (tools_json == nullptr) {
        return err(TOOL_GRAMMAR_ERR_NULL_INPUT, "tools_json is null");
    }
    if (grammar_out == nullptr) {
        return err(TOOL_GRAMMAR_ERR_NULL_OUTPUT, "grammar_out is null");
    }
    if (grammar_max_len == 0) {
        return err(TOOL_GRAMMAR_ERR_ZERO_LENGTH, "grammar_max_len is zero");
    }
    grammar_out[0] = '\0';

    // ---- parse JSON ----

    json tools;
    try {
        tools = json::parse(tools_json);
    } catch (const std::exception & e) {
        return err(TOOL_GRAMMAR_ERR_INVALID_JSON,
                   std::string("invalid JSON: ") + e.what());
    }

    if (!tools.is_array()) {
        return err(TOOL_GRAMMAR_ERR_NOT_ARRAY, "tools JSON is not an array");
    }
    if (tools.empty()) {
        return err(TOOL_GRAMMAR_ERR_EMPTY_TOOLS, "tools array is empty");
    }

    // ---- validate every tool, check for duplicate names ----

    std::set<std::string> seen_names;
    for (size_t i = 0; i < tools.size(); ++i) {
        std::string verr = validate_tool_json(tools[i], i);
        if (!verr.empty()) {
            return err(TOOL_GRAMMAR_ERR_INVALID_TOOL, verr);
        }
        const std::string name = tools[i]["function"]["name"];
        if (!seen_names.insert(name).second) {
            return err(TOOL_GRAMMAR_ERR_DUPLICATE_NAME,
                       "duplicate function name: '" + name + "'");
        }
    }

    // ---- build grammar ----
    //
    // Target GBNF (example with get_weather(location, unit?) and search(query)):
    //
    //   root             ::= tool-call+
    //   tool-call        ::= "<tool_call>\n" tool-choice "</tool_call>" ws
    //   tool-choice      ::= tool-get-weather | tool-search
    //   tool-get-weather ::= "<function=get_weather>\n" tool-get-weather-arg-location
    //                        tool-get-weather-arg-unit? "</function>\n"
    //   tool-get-weather-arg-location ::= "<parameter=location>\n"
    //                                     xml-arg-string "\n"
    //                                     ("</parameter>\n")?
    //   tool-get-weather-arg-unit     ::= "<parameter=unit>\n"
    //                                     xml-arg-string "\n"
    //                                     ("</parameter>\n")?
    //   tool-search      ::= "<function=search>\n" tool-search-arg-query
    //                        "</function>\n"
    //   tool-search-arg-query ::= "<parameter=query>\n" xml-arg-string "\n"
    //                             ("</parameter>\n")?
    //   xml-arg-string   ::= (free text excluding Qwen XML delimiters)
    //   ws               ::= [ \t\n]*

    std::string grammar_str;
    try {
        grammar_str = build_grammar([&](const common_grammar_builder & builder) {
            // Free-text rule for parameter values.
            // Matches any text that does NOT contain the three XML delimiter
            // sequences that can follow a parameter value in the Qwen format.
            builder.add_rule("xml-arg-string", gbnf_excluding({
                "\n</parameter>",
                "\n<parameter=",
                "\n</function>"
            }));

            // Whitespace after </tool_call> (between parallel calls, or trailing).
            builder.add_rule("ws", "[ \\t\\n]*");

            // Per-tool rules
            std::vector<std::string> tool_rule_names;
            tool_rule_names.reserve(tools.size());

            for (size_t i = 0; i < tools.size(); ++i) {
                const auto & func = tools[i]["function"];
                const std::string name = func["name"];

                // Collect required parameter names into a set for O(1) lookup.
                std::set<std::string> required_set;
                if (func.contains("parameters") &&
                    func["parameters"].contains("required") &&
                    func["parameters"]["required"].is_array()) {
                    for (const auto & r : func["parameters"]["required"]) {
                        if (r.is_string()) {
                            required_set.insert(r.get<std::string>());
                        }
                    }
                }

                // Build parameter rules (ordered as declared in properties).
                std::string args_body;
                if (func.contains("parameters") &&
                    func["parameters"].contains("properties") &&
                    func["parameters"]["properties"].is_object()) {
                    const auto & props = func["parameters"]["properties"];
                    for (auto it = props.begin(); it != props.end(); ++it) {
                        const std::string & pname = it.key();
                        bool is_required = required_set.count(pname) > 0;

                        // <parameter=pname>\n xml-arg-string \n (</parameter>\n)?
                        std::string arg_body =
                            gbnf_format_literal("<parameter=" + pname + ">\n") +
                            " xml-arg-string " +
                            gbnf_format_literal("\n") +
                            " (" + gbnf_format_literal("</parameter>\n") + ")?";

                        std::string actual_name = builder.add_rule(
                            "tool-" + name + "-arg-" + pname, arg_body);

                        if (!args_body.empty()) {
                            args_body += " ";
                        }
                        args_body += actual_name;
                        if (!is_required) {
                            args_body += "?";
                        }
                    }
                }

                // <function=name>\n [args] </function>\n
                std::string tool_body =
                    gbnf_format_literal("<function=" + name + ">\n");
                if (!args_body.empty()) {
                    tool_body += " " + args_body;
                }
                tool_body += " " + gbnf_format_literal("</function>\n");

                std::string actual_name = builder.add_rule(
                    "tool-" + name, tool_body);
                tool_rule_names.push_back(actual_name);
            }

            // tool-choice: alternation of all per-tool rules
            std::string choice_body;
            for (size_t i = 0; i < tool_rule_names.size(); ++i) {
                if (i > 0) {
                    choice_body += " | ";
                }
                choice_body += tool_rule_names[i];
            }
            builder.add_rule("tool-choice", choice_body);

            // tool-call: one complete <tool_call>…</tool_call> block
            builder.add_rule("tool-call",
                gbnf_format_literal("<tool_call>\n") +
                " tool-choice " +
                gbnf_format_literal("</tool_call>") +
                " ws");

            // root: one or more parallel tool calls
            builder.add_rule("root", "tool-call+");
        });
    } catch (const std::exception & e) {
        return err(TOOL_GRAMMAR_ERR_GRAMMAR_BUILD,
                   std::string("grammar build failed: ") + e.what());
    }

    // ---- write output ----

    if (grammar_str.length() >= grammar_max_len) {
        std::memcpy(grammar_out, grammar_str.c_str(), grammar_max_len - 1);
        grammar_out[grammar_max_len - 1] = '\0';
        return err(TOOL_GRAMMAR_ERR_TRUNCATED,
                   "grammar output truncated (" +
                   std::to_string(grammar_str.length()) + " bytes, buffer is " +
                   std::to_string(grammar_max_len) + ")");
    }

    std::memcpy(grammar_out, grammar_str.c_str(), grammar_str.length() + 1);
    return TOOL_GRAMMAR_OK;
}
