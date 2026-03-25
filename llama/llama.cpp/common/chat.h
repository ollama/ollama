// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include "peg-parser.h"
#include <functional>
#include <chrono>
#include <string>
#include <vector>
#include <map>

struct common_chat_templates;

struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;

    bool operator==(const common_chat_tool_call & other) const {
        return name == other.name && arguments == other.arguments && id == other.id;
    }
};

struct common_chat_msg_content_part {
    std::string type;
    std::string text;

    bool operator==(const common_chat_msg_content_part & other) const {
        return type == other.type && text == other.text;
    }
};

struct common_chat_msg {
    std::string role;
    std::string content;
    std::vector<common_chat_msg_content_part> content_parts;
    std::vector<common_chat_tool_call> tool_calls;
    std::string reasoning_content;
    std::string tool_name;
    std::string tool_call_id;

    template <class T> T to_json_oaicompat() const;

    bool empty() const {
        return content.empty() && content_parts.empty() && tool_calls.empty() && reasoning_content.empty() && tool_name.empty() && tool_call_id.empty();
    }
    void set_tool_call_ids(std::vector<std::string> & ids_cache, const std::function<std::string()> & gen_tool_call_id) {
        for (auto i = 0u; i < tool_calls.size(); i++) {
            if (ids_cache.size() <= i) {
                auto id = tool_calls[i].id;
                if (id.empty()) {
                    id = gen_tool_call_id();
                }
                ids_cache.push_back(id);
            }
            tool_calls[i].id = ids_cache[i];
        }
    }
    bool operator==(const common_chat_msg & other) const {
        return role == other.role
            && content == other.content
            && content_parts == other.content_parts
            && tool_calls == other.tool_calls
            && reasoning_content == other.reasoning_content
            && tool_name == other.tool_name
            && tool_call_id == other.tool_call_id;
    }
    bool operator!=(const common_chat_msg & other) const {
        return !(*this == other);
    }
};

struct common_chat_msg_diff {
    std::string reasoning_content_delta;
    std::string content_delta;
    size_t tool_call_index = std::string::npos;
    common_chat_tool_call tool_call_delta;

    static std::vector<common_chat_msg_diff> compute_diffs(const common_chat_msg & msg_prv, const common_chat_msg & msg_new);

    bool operator==(const common_chat_msg_diff & other) const {
        return content_delta == other.content_delta
        && tool_call_index == other.tool_call_index
        && tool_call_delta == other.tool_call_delta;
    }
};

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;
};

enum common_chat_tool_choice {
    COMMON_CHAT_TOOL_CHOICE_AUTO,
    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
    COMMON_CHAT_TOOL_CHOICE_NONE,
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,
    COMMON_CHAT_FORMAT_GENERIC,
    COMMON_CHAT_FORMAT_MISTRAL_NEMO,
    COMMON_CHAT_FORMAT_MAGISTRAL,
    COMMON_CHAT_FORMAT_LLAMA_3_X,
    COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
    COMMON_CHAT_FORMAT_DEEPSEEK_R1,
    COMMON_CHAT_FORMAT_FIREFUNCTION_V2,
    COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2,
    COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
    COMMON_CHAT_FORMAT_DEEPSEEK_V3_1,
    COMMON_CHAT_FORMAT_HERMES_2_PRO,
    COMMON_CHAT_FORMAT_COMMAND_R7B,
    COMMON_CHAT_FORMAT_GRANITE,
    COMMON_CHAT_FORMAT_GPT_OSS,
    COMMON_CHAT_FORMAT_SEED_OSS,
    COMMON_CHAT_FORMAT_NEMOTRON_V2,
    COMMON_CHAT_FORMAT_APERTUS,
    COMMON_CHAT_FORMAT_LFM2_WITH_JSON_TOOLS,
    COMMON_CHAT_FORMAT_GLM_4_5,
    COMMON_CHAT_FORMAT_MINIMAX_M2,
    COMMON_CHAT_FORMAT_KIMI_K2,
    COMMON_CHAT_FORMAT_QWEN3_CODER_XML,
    COMMON_CHAT_FORMAT_APRIEL_1_5,
    COMMON_CHAT_FORMAT_XIAOMI_MIMO,

    // These are intended to be parsed by the PEG parser
    COMMON_CHAT_FORMAT_PEG_SIMPLE,
    COMMON_CHAT_FORMAT_PEG_NATIVE,
    COMMON_CHAT_FORMAT_PEG_CONSTRUCTED,

    COMMON_CHAT_FORMAT_COUNT, // Not a format, just the # formats
};

struct common_chat_templates_inputs {
    std::vector<common_chat_msg> messages;
    std::string grammar;
    std::string json_schema;
    bool add_generation_prompt = true;
    bool use_jinja = true;
    // Parameters below only supported when use_jinja is true
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool parallel_tool_calls = false;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
    bool enable_thinking = true;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::map<std::string, std::string> chat_template_kwargs;
    bool add_bos = false;
    bool add_eos = false;
};

struct common_chat_params {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string                         prompt;
    std::string                         grammar;
    bool                                grammar_lazy = false;
    bool                                thinking_forced_open = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
    std::string                         parser;
};

struct common_chat_syntax {
    common_chat_format       format                = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format  reasoning_format      = COMMON_REASONING_FORMAT_NONE;
    // Whether reasoning_content should be inlined in the content (e.g. for reasoning_format=deepseek in stream mode)
    bool                     reasoning_in_content  = false;
    bool                     thinking_forced_open  = false;
    bool                     parse_tool_calls      = true;
    common_peg_arena         parser                = {};
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool common_chat_verify_template(const std::string & tmpl, bool use_jinja);

void common_chat_templates_free(struct common_chat_templates * tmpls);

struct common_chat_templates_deleter { void operator()(common_chat_templates * tmpls) { common_chat_templates_free(tmpls); } };

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

common_chat_templates_ptr common_chat_templates_init(
                                    const struct llama_model * model,
                                           const std::string & chat_template_override,
                                           const std::string & bos_token_override = "",
                                           const std::string & eos_token_override = "");

bool         common_chat_templates_was_explicit(const struct common_chat_templates * tmpls);
const char * common_chat_templates_source(const struct common_chat_templates * tmpls, const char * variant = nullptr);


struct common_chat_params      common_chat_templates_apply(
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs);

// Format single message, while taking into account the position of that message in chat history
std::string common_chat_format_single(
        const struct common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & past_msg,
        const common_chat_msg & new_msg,
        bool add_ass,
        bool use_jinja);

// Returns an example of formatted chat
std::string common_chat_format_example(
    const struct common_chat_templates * tmpls,
    bool use_jinja,
    const std::map<std::string, std::string> & chat_template_kwargs);

const char*               common_chat_format_name(common_chat_format format);
const char*               common_reasoning_format_name(common_reasoning_format format);
common_reasoning_format   common_reasoning_format_from_name(const std::string & format);
common_chat_msg           common_chat_parse(const std::string & input, bool is_partial, const common_chat_syntax & syntax);
common_chat_msg           common_chat_peg_parse(const common_peg_arena & parser, const std::string & input, bool is_partial, const common_chat_syntax & syntax);

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice);

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates);

// Parses a JSON array of messages in OpenAI's chat completion API format.
// T can be std::string containing JSON or nlohmann::ordered_json
template <class T> std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const T & messages);
template <class T> T common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text = false);

// Parses a JSON array of tools in OpenAI's chat completion tool call API format.
// T can be std::string containing JSON or nlohmann::ordered_json
template <class T> std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const T & tools);
template <class T> T common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools);

template <class T> T common_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
