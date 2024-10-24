#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
#include "httplib.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

#include <random>
#include <sstream>
#include <string>
#include <vector>

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

using json = nlohmann::ordered_json;

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value\n", key.c_str(), json(default_value).type_name());
            return default_value;
        }
    } else {
        return default_value;
    }
}

//
// chat template utils
//

// Format given chat. If tmpl is empty, we take the template from model metadata
inline std::string format_chat(const struct llama_model * model, const std::string & tmpl, const std::vector<json> & messages) {
    std::vector<common_chat_msg> chat;

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & curr_msg = messages[i];

        std::string role = json_value(curr_msg, "role", std::string(""));

        std::string content;
        if (curr_msg.contains("content")) {
            if (curr_msg["content"].is_string()) {
                content = curr_msg["content"].get<std::string>();
            } else if (curr_msg["content"].is_array()) {
                for (const auto & part : curr_msg["content"]) {
                    if (part.contains("text")) {
                        content += "\n" + part["text"].get<std::string>();
                    }
                }
            } else {
                throw std::runtime_error("Invalid 'content' type (ref: https://github.com/ggerganov/llama.cpp/issues/8367)");
            }
        } else {
            throw std::runtime_error("Missing 'content' (ref: https://github.com/ggerganov/llama.cpp/issues/8367)");
        }

        chat.push_back({role, content});
    }

    const auto formatted_chat = common_chat_apply_template(model, tmpl, chat, true);
    LOG_DBG("formatted_chat: '%s'\n", formatted_chat.c_str());

    return formatted_chat;
}

static std::string llama_get_chat_template(const struct llama_model * model) {
    std::string template_key = "tokenizer.chat_template";
    // call with NULL buffer to get the total size of the string
    int32_t res = llama_model_meta_val_str(model, template_key.c_str(), NULL, 0);
    if (res < 0) {
        return "";
    } else {
        std::vector<char> model_template(res, 0);
        llama_model_meta_val_str(model, template_key.c_str(), model_template.data(), model_template.size());
        return std::string(model_template.data(), model_template.size());
    }
}

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline std::vector<uint8_t> base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// random string / id
//

static std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

static std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

//
// other common utils
//

static size_t longest_common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
}

static size_t longest_common_prefix(const std::string & a, const std::string & b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop, const std::string &text) {
    if (!text.empty() && !stop.empty()) {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial)) {
                    return text.size() - char_index - 1;
                }
            }
        }
    }

    return std::string::npos;
}

static bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

struct completion_token_output {
    llama_token tok;
    std::string text_to_send;

    struct token_prob {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
};

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context * ctx, const std::vector<completion_token_output> & probs) {
    json out = json::array();

    for (const auto & prob : probs) {
        json probs_for_token = json::array();

        for (const auto & p : prob.probs) {
            const std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
            probs_for_token.push_back(json {
                {"tok_str", tok_str},
                {"prob",    p.prob},
            });
        }

        const std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
        out.push_back(json {
            {"content", tok_str},
            {"probs",   probs_for_token},
        });
    }

    return out;
}

static bool server_sent_event(httplib::DataSink & sink, const char * event, const json & data) {
    const std::string str =
        std::string(event) + ": " +
        data.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n"; // note: these newlines are important (not sure why though, if you know, add a comment to explain)

    LOG_DBG("data stream, to_send: %s", str.c_str());

    return sink.write(str.c_str(), str.size());
}

//
// OAI utils
//

static json oaicompat_completion_params_parse(
    const struct llama_model * model,
    const json & body, /* openai api json semantics */
    const std::string & chat_template) {
    json llama_params;

    llama_params["__oaicompat"] = true;

    // Apply chat template to the list of messages
    llama_params["prompt"] = format_chat(model, chat_template, body.at("messages"));

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            llama_params["json_schema"] = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            json json_schema = json_value(response_format, "json_schema", json::object());
            llama_params["json_schema"] = json_value(json_schema, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::runtime_error("top_logprobs requires logprobs to be set to true");
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params { "tools", "tool_choice" };
    for (const auto & param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", "tfs_z",... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

static json format_final_response_oaicompat(const json & request, const json & result, const std::string & completion_id, bool streaming = false, bool verbose = false) {
    bool stopped_word        = result.count("stopped_word") != 0;
    bool stopped_eos         = json_value(result, "stopped_eos", false);
    int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
    int num_prompt_tokens    = json_value(result, "tokens_evaluated", 0);
    std::string content      = json_value(result, "content", std::string(""));

    std::string finish_reason = "length";
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }

    json choices =
        streaming ? json::array({json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"delta", json::object()}}})
                  : json::array({json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"message", json{{"content", content},
                                                         {"role", "assistant"}}}}});

    std::time_t t = std::time(0);

    json res = json {
        {"choices", choices},
        {"created", t},
        {"model",
            json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", streaming ? "chat.completion.chunk" : "chat.completion"},
        {"usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        }},
        {"id", completion_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = result;
    }

    if (result.contains("completion_probabilities")) {
        res["completion_probabilities"] = json_value(result, "completion_probabilities", json::array());
    }

    return res;
}

// return value is vector as there is one case where we might need to generate two responses
static std::vector<json> format_partial_response_oaicompat(const json & result, const std::string & completion_id) {
    if (!result.contains("model") || !result.contains("oaicompat_token_ctr")) {
        return std::vector<json>({result});
    }

    bool first = json_value(result, "oaicompat_token_ctr", 0) == 0;
    std::string modelname = json_value(result, "model", std::string(DEFAULT_OAICOMPAT_MODEL));

    bool stopped_word   = json_value(result, "stopped_word",  false);
    bool stopped_eos    = json_value(result, "stopped_eos",   false);
    bool stopped_limit  = json_value(result, "stopped_limit", false);
    std::string content = json_value(result, "content",       std::string(""));

    std::string finish_reason;
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }
    if (stopped_limit) {
        finish_reason = "length";
    }

    std::time_t t = std::time(0);

    json choices;

    if (!finish_reason.empty()) {
        choices = json::array({json{{"finish_reason", finish_reason},
                                    {"index", 0},
                                    {"delta", json::object()}}});
    } else {
        if (first) {
            if (content.empty()) {
                choices = json::array({json{{"finish_reason", nullptr},
                                            {"index", 0},
                                            {"delta", json{{"role", "assistant"}}}}});
            } else {
                // We have to send this as two updates to conform to openai behavior
                json initial_ret = json{{"choices", json::array({json{
                                        {"finish_reason", nullptr},
                                        {"index", 0},
                                        {"delta", json{
                                            {"role", "assistant"}
                                        }}}})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"}};

                json second_ret = json{
                            {"choices", json::array({json{{"finish_reason", nullptr},
                                                            {"index", 0},
                                                            {"delta", json{
                                                            {"content", content}}}
                                                            }})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"}};

                return std::vector<json>({initial_ret, second_ret});
            }
        } else {
            // Some idiosyncrasy in task processing logic makes several trailing calls
            // with empty content, we ignore these at the calee site.
            if (content.empty()) {
                return std::vector<json>({json::object()});
            }

            choices = json::array({json{
                {"finish_reason", nullptr},
                {"index", 0},
                {"delta",
                json{
                    {"content", content},
                }},
            }});
        }
    }

    json ret = json {
        {"choices", choices},
        {"created", t},
        {"id",      completion_id},
        {"model",   modelname},
        {"object",  "chat.completion.chunk"}
    };
    if (!finish_reason.empty()) {
        int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
        int num_prompt_tokens    = json_value(result, "tokens_evaluated", 0);
        ret.push_back({"usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        }});
    }

    return std::vector<json>({ret});
}

static json format_embeddings_response_oaicompat(const json & request, const json & embeddings) {
    json data = json::array();
    int i = 0;
    for (const auto & elem : embeddings) {
        data.push_back(json{
            {"embedding", json_value(elem, "embedding", json::array())},
            {"index",     i++},
            {"object",    "embedding"}
        });
    }

    json res = json {
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json { // TODO: fill
            {"prompt_tokens", 0},
            {"total_tokens", 0}
        }},
        {"data", data}
    };

    return res;
}

static json format_response_rerank(const json & request, const json & ranks) {
    json data = json::array();
    int i = 0;
    for (const auto & rank : ranks) {
        data.push_back(json{
            {"index",    i++},
            {"relevance_score", json_value(rank, "score", 0.0)},
        });
    }

    json res = json {
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json { // TODO: fill
            {"prompt_tokens", 0},
            {"total_tokens", 0}
        }},
        {"results", data}
    };

    return res;
}

static bool is_valid_utf8(const std::string & str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    const unsigned char* end = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 ||
                (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

static json format_tokenizer_response(const json & tokens) {
    return json {
        {"tokens", tokens}
    };
}

static json format_detokenized_response(const std::string & content) {
    return json {
        {"content", content}
    };
}

static json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}
