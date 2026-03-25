#include "common.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"
#include "arg.h" // for common_remote_get_content; TODO: use download.h only
#include "base64.hpp"

#include "server-common.h"

#include <random>
#include <sstream>
#include <fstream>

json format_error_response(const std::string & message, const enum error_type type) {
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
        case ERROR_TYPE_EXCEED_CONTEXT_SIZE:
            type_str = "exceed_context_size_error";
            code = 400;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

//
// random string / id
//

std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

std::string gen_tool_call_id() {
    return random_string();
}

//
// lora utils
//

bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras) {
    bool found_alora = false;
    for (const auto & lora : loras) {
        if (lora.scale != 0) {
            if (llama_adapter_get_alora_n_invocation_tokens(lora.ptr) == 0) {
                return false;
            }
            found_alora = true;
        }
    }
    return found_alora;
}

bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next) {

    // This should always be called after determining that the two sets are
    // _not_ equal. This assert is therefore some slightly wasted work and
    // should be safe to remove as long as this method is called correctly.
    GGML_ASSERT(!are_lora_equal(current, next));

    return (
        !(lora_get_enabled_ids(current).empty() || lora_all_alora(current)) ||
        !lora_all_alora(next));
}

std::vector<common_adapter_lora_info> parse_lora_request(
        const std::vector<common_adapter_lora_info> & lora_base,
        const json & data) {
    std::vector<common_adapter_lora_info> lora(lora_base);
    int max_idx = lora.size();

    // clear existing value
    for (auto & entry : lora) {
        entry.scale = 0.0f;
    }

    // set value
    for (const auto & entry : data) {
        int id      = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        if (0 <= id && id < max_idx) {
            lora[id].scale = scale;
        } else {
            throw std::runtime_error("invalid adapter id");
        }
    }

    return lora;
}

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2) {
    if (l1.size() != l2.size()) {
        return false;
    }
    for (size_t i = 0; i < l1.size(); ++i) {
        // we don't check lora.path to reduce the time complexity
        if (l1[i].scale != l2[i].scale || l1[i].ptr != l2[i].ptr) {
            return false;
        }
    }
    return true;
}

std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras) {
    std::vector<size_t> enabled_ids;
    for (size_t i = 0; i < loras.size(); ++i) {
        if (loras[i].scale > 0) {
            enabled_ids.push_back(i);
        }
    }
    return enabled_ids;
}

//
// base64 utils (TODO: use the base64::decode from base64.hpp)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline raw_buffer base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    raw_buffer ret;

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
// server_tokens implementation
//

server_tokens::server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd) : has_mtmd(has_mtmd) {
    for (size_t i = 0; i < mtmd_chunks.size(); ++i) {
        push_back(mtmd_chunks[i]);
    }
}

server_tokens::server_tokens(const llama_tokens & tokens, bool has_mtmd) : has_mtmd(has_mtmd), tokens(tokens) {
}

llama_pos server_tokens::pos_next() const {
    if (!has_mtmd) {
        return tokens.size();
    }

    llama_pos res = tokens.size();

    for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ++it) {
        const auto & chunk = it->second;
        res += mtmd_input_chunk_get_n_pos(chunk.get()) - mtmd_input_chunk_get_n_tokens(chunk.get());
    }

    return res;
}

std::string server_tokens::str() const {
    std::ostringstream oss;
    oss << "tokens: ";
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
        llama_token t = tokens[idx];
        oss << "idx:" << idx << " ";
        if (t == LLAMA_TOKEN_NULL) {
            oss << "<embd> ";
        } else {
            oss << t << " ";
        }
    }
    oss << "\n";
    oss << "image idx: ";
    for (const auto & it : map_idx_to_media) {
        oss << it.first << ", ";
    }
    return oss.str();
}

const mtmd::input_chunk_ptr & server_tokens::find_chunk(size_t idx) const {
    auto it = map_idx_to_media.find(idx);
    if (it != map_idx_to_media.end()) {
        return it->second;
    }
    throw std::runtime_error("Chunk not found");
}

void server_tokens::push_back(llama_token tok) {
    if (tok == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("Invalid token");
    }
    tokens.emplace_back(tok);
}

void server_tokens::push_back(const mtmd_input_chunk * chunk) {
    auto type = mtmd_input_chunk_get_type(chunk);
    if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        GGML_ASSERT(has_mtmd);
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        size_t start_idx = tokens.size();
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens.emplace_back(LLAMA_TOKEN_NULL);
        }
        mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
        map_idx_to_media[start_idx] = std::move(new_chunk);
    } else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto * text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        for (size_t i = 0; i < n_tokens; ++i) {
            push_back(text_tokens[i]);
        }
    } else {
        GGML_ABORT("Invalid chunk type");
    }
}

void server_tokens::push_back(server_tokens & tokens) {
    size_t start_idx = size();
    for (size_t i = 0; i < tokens.size(); i++) {
        push_back(tokens[i]);
    }
    if (tokens.has_mtmd) {
        // Assert if we are copying MTMD chunks to a server_tokens that does not have mtmd.
        // We could also just check, but this will prevent silently dropping MTMD data.
        GGML_ASSERT(has_mtmd);
        for (auto it = tokens.map_idx_to_media.begin(); it != tokens.map_idx_to_media.end(); ) {
            auto * chunk = tokens.map_idx_to_media[it->first].get();
            mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
            map_idx_to_media[start_idx + it->first] = std::move(new_chunk);
        }
    }
}

void server_tokens::insert(const llama_tokens & inp_tokens) {
    GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
    tokens.insert(tokens.end(), inp_tokens.begin(), inp_tokens.end());
}

const llama_tokens & server_tokens::get_text_tokens() const {
    GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
    return tokens;
}

void server_tokens::set_token(llama_pos pos, llama_token id) {
    GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
    tokens[pos] = id;
}

void server_tokens::keep_first(size_t n) {
    GGML_ASSERT(n <= tokens.size());
    if (has_mtmd) {
        if (n == tokens.size()) {
            return; // nothing to do
        }
        // we throw an error if we try to remove a token in the middle of an image
        // for ex. with input of 5 text tokens and 2 images:
        //    [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
        // n  1   2   3   4   5   6      7      8      9      10
        // allowed to resize      ^                    ^
        // disallowed to resize          ^      ^             ^
        if (n > 0) {
            // make sure we never remove tokens in the middle of an image
            // note that the case where we keep a full image at the end is allowed:
            //   tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] != LLAMA_TOKEN_NULL
            if (tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] == LLAMA_TOKEN_NULL) {
                find_chunk(n - 1); // will throw an error if the token is not begin-of-chunk
            }
        }
        // remove all image chunks that are not used anymore
        for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ) {
            size_t idx = it->first;
            if (idx >= n) {
                it = map_idx_to_media.erase(it);
            } else {
                ++it;
            }
        }
    }
    tokens.resize(n);
}

std::string server_tokens::detokenize(const llama_context * ctx, bool special) const {
    llama_tokens text_tokens;
    text_tokens.reserve(tokens.size());
    for (const auto & t : tokens) {
        if (t != LLAMA_TOKEN_NULL) {
            text_tokens.push_back(t);
        }
    }
    return common_detokenize(ctx, text_tokens, special);
}

size_t server_tokens::get_common_prefix(const server_tokens & b) const {
    const size_t max_idx = std::min(tokens.size(), b.tokens.size());

    if (!has_mtmd) {
        for (size_t i = 0; i < max_idx; ++i) {
            if (tokens[i] == b.tokens[i]) {
                continue;
            }

            return i;
        }

        return max_idx;
    }

    for (size_t i = 0; i < max_idx; ++i) {
        const llama_token ai =   tokens[i];
        const llama_token bi = b.tokens[i];

        if (ai == LLAMA_TOKEN_NULL && bi == LLAMA_TOKEN_NULL) {
            const auto & a_chunk =   find_chunk(i);
            const auto & b_chunk = b.find_chunk(i);

            GGML_ASSERT(a_chunk && b_chunk);

            const std::string id_ai = mtmd_input_chunk_get_id(a_chunk.get());
            const std::string id_bi = mtmd_input_chunk_get_id(b_chunk.get());

            const size_t n_tok_a = mtmd_input_chunk_get_n_tokens(a_chunk.get());
            const size_t n_tok_b = mtmd_input_chunk_get_n_tokens(b_chunk.get());

            if (id_ai == id_bi && n_tok_a == n_tok_b) {
                GGML_ASSERT(n_tok_a > 0 && "Invalid media chunk"); // should never happen
                i += n_tok_a - 1; // will be +1 by the for loop
                continue;
            }

            return i;
        }

        if (ai == bi) {
            continue;
        }

        return i;
    }

    return max_idx; // all tokens are equal
}

bool server_tokens::validate(const struct llama_context * ctx) const {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto & t = tokens[i];
        if (t == LLAMA_TOKEN_NULL) {
            try {
                const auto & chunk = find_chunk(i);
                size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk.get());
                i += n_tokens - 1; // will be +1 by the for loop
            } catch (const std::exception & e) {
                return false;
            }
        } else if (t < 0 || t >= n_vocab) {
            return false;
        }
    }
    return true;
}

int32_t server_tokens::process_chunk(
            llama_context * ctx,
            mtmd_context * mctx,
            size_t idx,
            llama_pos pos,
            int32_t seq_id,
            size_t & n_tokens_out) const {
    const auto & chunk = find_chunk(idx);
    const char * name = mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_IMAGE
                        ? "image" : "audio";
    SRV_INF("processing %s...\n", name);
    int32_t n_batch = llama_n_batch(ctx);
    int64_t t0 = ggml_time_ms();
    llama_pos new_n_past; // unused for now
    int32_t result = mtmd_helper_eval_chunk_single(mctx, ctx,
        chunk.get(),
        pos,
        seq_id,
        n_batch,
        true, // logits last
        &new_n_past);
    SRV_INF("%s processed in %" PRId64 " ms\n", name, ggml_time_ms() - t0);
    if (result != 0) {
        LOG_ERR("mtmd_helper_eval failed with status %d", result);
        n_tokens_out = 0;
        return result;
    }
    n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
    return 0;
}

server_tokens server_tokens::clone() const {
    server_tokens res;
    res.has_mtmd = has_mtmd;
    res.tokens   = tokens;
    for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ++it) {
        size_t idx = it->first;
        const mtmd::input_chunk_ptr & chunk = it->second;
        res.map_idx_to_media[idx] = mtmd::input_chunk_ptr(mtmd_input_chunk_copy(chunk.get()));
    }
    return res;
}

//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool json_is_array_of_mixed_numbers_strings(const json & data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto & e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

bool json_is_array_and_contains_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (e.is_number_integer()) {
                return true;
            }
        }
        return false;
    }
    return false;
}

json json_get_nested_values(const std::vector<std::string> & paths, const json & js) {
    json result = json::object();

    for (const std::string & path : paths) {
        json current = js;
        const auto keys = string_split<std::string>(path, /*separator*/ '/');
        bool valid_path = true;
        for (const std::string & k : keys) {
            if (valid_path && current.is_object() && current.contains(k)) {
                current = current[k];
            } else {
                valid_path = false;
            }
        }
        if (valid_path) {
            result[path] = current;
        }
    }
    return result;
}

llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                llama_tokens p;
                if (first) {
                    p = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    } else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

// Computes FNV-1a hash of the data
static std::string fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

server_tokens process_mtmd_prompt(mtmd_context * mctx, std::string prompt, std::vector<raw_buffer> files) {
    mtmd::bitmaps bitmaps;
    for (auto & file : files) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(mctx, file.data(), file.size()));
        if (!bmp.ptr) {
            throw std::runtime_error("Failed to load image or audio file");
        }
        // calculate bitmap hash (for KV caching)
        std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
        bmp.set_id(hash.c_str());
        bitmaps.entries.push_back(std::move(bmp));
    }
    // process prompt
    std::vector<server_tokens> inputs;
    // multimodal
    mtmd_input_text inp_txt = {
        prompt.c_str(),
        /* add_special */   true,
        /* parse_special */ true,
    };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t tokenized = mtmd_tokenize(mctx,
                                      chunks.ptr.get(),
                                      &inp_txt,
                                      bitmaps_c_ptr.data(),
                                      bitmaps_c_ptr.size());
    if (tokenized != 0) {
        throw std::runtime_error("Failed to tokenize prompt");
    }
    auto result = server_tokens(chunks, true);
    return result;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 */
static server_tokens tokenize_input_subprompt(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special) {
    constexpr char JSON_STRING_PROMPT_KEY[] = "prompt_string";
    constexpr char JSON_MTMD_DATA_KEY[] = "multimodal_data";
    const bool has_mtmd = mctx != nullptr;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        llama_tokens tmp = tokenize_mixed(vocab, json_prompt, add_special, parse_special);
        return server_tokens(tmp, false);
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        llama_tokens tmp = json_prompt.get<llama_tokens>();
        return server_tokens(tmp, false);
    } else if (json_prompt.contains(JSON_STRING_PROMPT_KEY)) {
        // JSON object with prompt key.
        if (json_prompt.contains(JSON_MTMD_DATA_KEY)) {
            if (!has_mtmd)
                throw std::runtime_error("Multimodal data provided, but model does not support multimodal requests.");

            // JSON object with prompt and multimodal key.
            std::vector<raw_buffer> files;
            for (const auto & entry : json_prompt.at(JSON_MTMD_DATA_KEY)) {
                files.push_back(base64_decode(entry));
            }
            return process_mtmd_prompt(mctx, json_prompt.at(JSON_STRING_PROMPT_KEY), files);
        } else {
            // Not multimodal, but contains a subobject.
            llama_tokens tmp = tokenize_mixed(vocab, json_prompt.at(JSON_STRING_PROMPT_KEY), add_special, parse_special);
            return server_tokens(tmp, false);
        }
   } else {
       throw std::runtime_error("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
   }
}

std::vector<server_tokens> tokenize_input_prompts(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special) {
    std::vector<server_tokens> result;
    if (json_prompt.is_array() && !json_is_array_and_contains_numbers(json_prompt)) {
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            result.push_back(tokenize_input_subprompt(vocab, mctx, p,add_special, parse_special));
        }
    } else {
        result.push_back(tokenize_input_subprompt(vocab, mctx, json_prompt, add_special, parse_special));
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

//
// OAI utils
//

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body) {
    json llama_params;

    if (!body.contains("prompt")) {
        throw std::runtime_error("\"prompt\" is required");
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "echo" field
    if (json_value(body, "echo", false)) {
        throw std::runtime_error("Only no echo is supported");
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params { "best_of", "suffix" };
    for (const auto & param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

// media_path always end with '/', see arg.cpp
static void handle_media(
        std::vector<raw_buffer> & out_files,
        json & media_obj,
        const std::string & media_path) {
    std::string url = json_value(media_obj, "url", std::string());
    if (string_starts_with(url, "http")) {
        // download remote image
        // TODO @ngxson : maybe make these params configurable
        common_remote_params params;
        params.headers.push_back("User-Agent: llama.cpp/" + build_info);
        params.max_size = 1024 * 1024 * 10; // 10MB
        params.timeout  = 10; // seconds
        SRV_INF("downloading image from '%s'\n", url.c_str());
        auto res = common_remote_get_content(url, params);
        if (200 <= res.first && res.first < 300) {
            SRV_INF("downloaded %zu bytes\n", res.second.size());
            raw_buffer data;
            data.insert(data.end(), res.second.begin(), res.second.end());
            out_files.push_back(data);
        } else {
            throw std::runtime_error("Failed to download image");
        }

    } else if (string_starts_with(url, "file://")) {
        if (media_path.empty()) {
            throw std::invalid_argument("file:// URLs are not allowed unless --media-path is specified");
        }
        // load local image file
        std::string file_path = url.substr(7); // remove "file://"
        raw_buffer data;
        if (!fs_validate_filename(file_path, true)) {
            throw std::invalid_argument("file path is not allowed: " + file_path);
        }
        SRV_INF("loading image from local file '%s'\n", (media_path + file_path).c_str());
        std::ifstream file(media_path + file_path, std::ios::binary);
        if (!file) {
            throw std::invalid_argument("file does not exist or cannot be opened: " + file_path);
        }
        data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        out_files.push_back(data);

    } else {
        // try to decode base64 image
        std::vector<std::string> parts = string_split<std::string>(url, /*separator*/ ',');
        if (parts.size() != 2) {
            throw std::runtime_error("Invalid url value");
        } else if (!string_starts_with(parts[0], "data:image/")) {
            throw std::runtime_error("Invalid url format: " + parts[0]);
        } else if (!string_ends_with(parts[0], "base64")) {
            throw std::runtime_error("url must be base64 encoded");
        } else {
            auto base64_data = parts[1];
            auto decoded_data = base64_decode(base64_data);
            out_files.push_back(decoded_data);
        }
    }
}

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const oaicompat_parser_options & opt,
    std::vector<raw_buffer> & out_files)
{
    json llama_params;

    auto tools = json_value(body, "tools", json());
    auto has_tools = tools.is_array() && !tools.empty();
    auto stream = json_value(body, "stream", false);
    auto tool_choice = json_value(body, "tool_choice", std::string("auto"));

    if (!opt.use_jinja) {
        if (has_tools) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
        if (tool_choice != "auto") {
            throw std::runtime_error("tool_choice param requires --jinja flag");
        }
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            json_schema = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::invalid_argument("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    // get input files
    if (!body.contains("messages")) {
        throw std::invalid_argument("'messages' is required");
    }
    json & messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::invalid_argument("Expected 'messages' to be an array");
    }
    for (auto & msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role != "assistant" && !msg.contains("content")) {
            throw std::invalid_argument("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content") && !msg.contains("tool_calls")) {
                throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        json & content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }

        if (!content.is_array()) {
            throw std::invalid_argument("Expected 'content' to be a string or an array");
        }

        for (auto & p : content) {
            std::string type      = json_value(p, "type", std::string());
            if (type == "image_url") {
                if (!opt.allow_image) {
                    throw std::runtime_error("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json image_url = json_value(p, "image_url", json::object());
                handle_media(out_files, image_url, opt.media_path);

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("image_url");

            } else if (type == "input_audio") {
                if (!opt.allow_audio) {
                    throw std::runtime_error("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json input_audio   = json_value(p, "input_audio", json::object());
                std::string data   = json_value(input_audio, "data", std::string());
                std::string format = json_value(input_audio, "format", std::string());
                // while we also support flac, we don't allow it here so we matches the OAI spec
                if (format != "wav" && format != "mp3") {
                    throw std::invalid_argument("input_audio.format must be either 'wav' or 'mp3'");
                }
                auto decoded_data = base64_decode(data); // expected to be base64 encoded
                out_files.push_back(decoded_data);

                // TODO: add audio_url support by reusing handle_media()

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("input_audio");

            } else if (type != "text") {
                throw std::invalid_argument("unsupported content[].type");
            }
        }
    }

    common_chat_templates_inputs inputs;
    inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                 = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice           = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema           = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar               = grammar;
    inputs.use_jinja             = opt.use_jinja;
    inputs.parallel_tool_calls   = json_value(body, "parallel_tool_calls", false);
    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    inputs.reasoning_format      = opt.reasoning_format;
    if (body.contains("reasoning_format")) {
        inputs.reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
    }
    inputs.enable_thinking       = opt.enable_thinking;
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        if (body.contains("grammar")) {
            throw std::invalid_argument("Cannot use custom grammar constraints with tools.");
        }
        llama_params["parse_tool_calls"] = true;
    }

    // merge the template args provided from command line with the args provided in the user request
    auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
    inputs.chat_template_kwargs = opt.chat_template_kwargs;
    for (const auto & item : chat_template_kwargs_object.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }

    // parse the "enable_thinking" kwarg to override the default value
    auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
    if (enable_thinking_kwarg == "true") {
        inputs.enable_thinking = true;
    } else if (enable_thinking_kwarg == "false") {
        inputs.enable_thinking = false;
    } else if (!enable_thinking_kwarg.empty() && enable_thinking_kwarg[0] == '"') {
        throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
    }

    // if the assistant message appears at the end of list, we do not add end-of-turn token
    // for ex. this can be useful to modify the reasoning process in reasoning models
    bool prefill_assistant_message = !inputs.messages.empty() && inputs.messages.back().role == "assistant" && opt.prefill_assistant;
    common_chat_msg last_message;
    if (prefill_assistant_message) {
        last_message = inputs.messages.back();
        inputs.messages.pop_back();

        /* sanity check, max one assistant message at the end of the list */
        if (!inputs.messages.empty() && inputs.messages.back().role == "assistant"){
            throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
        }

        /* TODO: test this properly */
        inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;

        if ( inputs.enable_thinking ) {
            throw std::invalid_argument("Assistant response prefill is incompatible with enable_thinking.");
        }

        inputs.add_generation_prompt = true;
    }

    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(opt.tmpls, inputs);

    /* Append assistant prefilled message */
    if (prefill_assistant_message) {
        if (!last_message.content_parts.empty()) {
            for (auto & p : last_message.content_parts) {
                chat_params.prompt += p.text;
            }
        } else {
            chat_params.prompt += last_message.content;
        }
    }

    llama_params["chat_format"]      = static_cast<int>(chat_params.format);
    llama_params["prompt"]           = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        llama_params["grammar"] = chat_params.grammar;
    }
    llama_params["grammar_lazy"]     = chat_params.grammar_lazy;
    auto grammar_triggers = json::array();
    for (const auto & trigger : chat_params.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }
    llama_params["grammar_triggers"] = grammar_triggers;
    llama_params["preserved_tokens"] = chat_params.preserved_tokens;
    llama_params["thinking_forced_open"]     = chat_params.thinking_forced_open;
    for (const auto & stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }
    if (!chat_params.parser.empty()) {
        llama_params["chat_parser"] = chat_params.parser;
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        if (has_tools && stream) {
            throw std::invalid_argument("logprobs is not supported with tools + stream");
        }
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::invalid_argument("top_logprobs requires logprobs to be set to true");
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

json convert_anthropic_to_oai(const json & body) {
    json oai_body;

    // Convert system prompt
    json oai_messages = json::array();
    auto system_param = json_value(body, "system", json());
    if (!system_param.is_null()) {
        std::string system_content;

        if (system_param.is_string()) {
            system_content = system_param.get<std::string>();
        } else if (system_param.is_array()) {
            for (const auto & block : system_param) {
                if (json_value(block, "type", std::string()) == "text") {
                    system_content += json_value(block, "text", std::string());
                }
            }
        }

        oai_messages.push_back({
            {"role", "system"},
            {"content", system_content}
        });
    }

    // Convert messages
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    const json & messages = body.at("messages");
    if (messages.is_array()) {
        for (const auto & msg : messages) {
            std::string role = json_value(msg, "role", std::string());

            if (!msg.contains("content")) {
                if (role == "assistant") {
                    continue;
                }
                oai_messages.push_back(msg);
                continue;
            }

            const json & content = msg.at("content");

            if (content.is_string()) {
                oai_messages.push_back(msg);
                continue;
            }

            if (!content.is_array()) {
                oai_messages.push_back(msg);
                continue;
            }

            json tool_calls = json::array();
            json converted_content = json::array();
            json tool_results = json::array();
            bool has_tool_calls = false;

            for (const auto & block : content) {
                std::string type = json_value(block, "type", std::string());

                if (type == "text") {
                    converted_content.push_back(block);
                } else if (type == "image") {
                    json source = json_value(block, "source", json::object());
                    std::string source_type = json_value(source, "type", std::string());

                    if (source_type == "base64") {
                        std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                        std::string data = json_value(source, "data", std::string());
                        std::ostringstream ss;
                        ss << "data:" << media_type << ";base64," << data;

                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", ss.str()}
                            }}
                        });
                    } else if (source_type == "url") {
                        std::string url = json_value(source, "url", std::string());
                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", url}
                            }}
                        });
                    }
                } else if (type == "tool_use") {
                    tool_calls.push_back({
                        {"id", json_value(block, "id", std::string())},
                        {"type", "function"},
                        {"function", {
                            {"name", json_value(block, "name", std::string())},
                            {"arguments", json_value(block, "input", json::object()).dump()}
                        }}
                    });
                    has_tool_calls = true;
                } else if (type == "tool_result") {
                    std::string tool_use_id = json_value(block, "tool_use_id", std::string());

                    auto result_content = json_value(block, "content", json());
                    std::string result_text;
                    if (result_content.is_string()) {
                        result_text = result_content.get<std::string>();
                    } else if (result_content.is_array()) {
                        for (const auto & c : result_content) {
                            if (json_value(c, "type", std::string()) == "text") {
                                result_text += json_value(c, "text", std::string());
                            }
                        }
                    }

                    tool_results.push_back({
                        {"role", "tool"},
                        {"tool_call_id", tool_use_id},
                        {"content", result_text}
                    });
                }
            }

            if (!converted_content.empty() || has_tool_calls) {
                json new_msg = {{"role", role}};
                if (!converted_content.empty()) {
                    new_msg["content"] = converted_content;
                } else if (has_tool_calls) {
                    new_msg["content"] = "";
                }
                if (!tool_calls.empty()) {
                    new_msg["tool_calls"] = tool_calls;
                }
                oai_messages.push_back(new_msg);
            }

            for (const auto & tool_msg : tool_results) {
                oai_messages.push_back(tool_msg);
            }
        }
    }

    oai_body["messages"] = oai_messages;

    // Convert tools
    if (body.contains("tools")) {
        const json & tools = body.at("tools");
        if (tools.is_array()) {
            json oai_tools = json::array();
            for (const auto & tool : tools) {
                oai_tools.push_back({
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(tool, "name", std::string())},
                        {"description", json_value(tool, "description", std::string())},
                        {"parameters", tool.contains("input_schema") ? tool.at("input_schema") : json::object()}
                    }}
                });
            }
            oai_body["tools"] = oai_tools;
        }
    }

    // Convert tool_choice
    if (body.contains("tool_choice")) {
        const json & tc = body.at("tool_choice");
        if (tc.is_object()) {
            std::string type = json_value(tc, "type", std::string());
            if (type == "auto") {
                oai_body["tool_choice"] = "auto";
            } else if (type == "any" || type == "tool") {
                oai_body["tool_choice"] = "required";
            }
        }
    }

    // Convert stop_sequences to stop
    if (body.contains("stop_sequences")) {
        oai_body["stop"] = body.at("stop_sequences");
    }

    // Handle max_tokens (required in Anthropic, but we're permissive)
    if (body.contains("max_tokens")) {
        oai_body["max_tokens"] = body.at("max_tokens");
    } else {
        oai_body["max_tokens"] = 4096;
    }

    // Pass through common params
    for (const auto & key : {"temperature", "top_p", "top_k", "stream"}) {
        if (body.contains(key)) {
            oai_body[key] = body.at(key);
        }
    }

    // Handle Anthropic-specific thinking param
    if (body.contains("thinking")) {
        json thinking = json_value(body, "thinking", json::object());
        std::string thinking_type = json_value(thinking, "type", std::string());
        if (thinking_type == "enabled") {
            int budget_tokens = json_value(thinking, "budget_tokens", 10000);
            oai_body["thinking_budget_tokens"] = budget_tokens;
        }
    }

    // Handle Anthropic-specific metadata param
    if (body.contains("metadata")) {
        json metadata = json_value(body, "metadata", json::object());
        std::string user_id = json_value(metadata, "user_id", std::string());
        if (!user_id.empty()) {
            oai_body["__metadata_user_id"] = user_id;
        }
    }

    return oai_body;
}

json format_embeddings_response_oaicompat(
        const json & request,
        const std::string & model_name,
        const json & embeddings,
        bool use_base64) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto & elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {
                {"embedding", base64::encode(data_ptr, data_size)},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", "base64"}
            };
        } else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);

        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }

    json res = json {
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n) {
    int32_t n_tokens = 0;
    bool return_text = is_tei_format && json_value(request, "return_text", false);
    std::vector<json> elements; // Temporary vector to hold unsorted elements
    std::string score_label = is_tei_format ? "score" : "relevance_score";
    for (const auto & rank : ranks) {
        int index = json_value(rank, "index", 0);
        json elem = json{
            {"index", index},
            {score_label, json_value(rank, "score", 0.0)},
        };
        n_tokens += json_value(rank, "tokens_evaluated", 0);
        if (return_text) {
            elem["text"] = std::move(texts[index]);
        }
        elements.push_back(elem);
    }

    std::sort(elements.begin(), elements.end(), [score_label](const json& a, const json& b) {
        return json_value(a, score_label, 0.0) > json_value(b, score_label, 0.0);
    });

    elements.resize(std::min(top_n, (int)elements.size()));
    json results = elements;

    if (is_tei_format) return results;

    json res = json{
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json{
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"results", results}
    };

    return res;
}


//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx) {
    std::vector<llama_token_data> cur;
    const auto * logits = llama_get_logits_ith(ctx, idx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_vocab = llama_vocab_n_tokens(vocab);

    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    // sort tokens by logits
    std::sort(cur.begin(), cur.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    });

    // apply softmax
    float max_l = cur[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < cur.size(); ++i) {
        float p = expf(cur[i].logit - max_l);
        cur[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < cur.size(); ++i) {
        cur[i].p /= cum_sum;
    }

    return cur;
}

std::string safe_json_to_str(const json & data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
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

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens) {
    return tokens_to_str(ctx, tokens.begin(), tokens.end());
}

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == LLAMA_TOKEN_NULL ? "" : common_token_to_piece(ctx, token);

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

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data) {
    std::ostringstream ss;
    auto send_single = [&ss](const json & data) {
        ss << "data: " <<
            safe_json_to_str(data) <<
            "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
    };

    if (data.is_array()) {
        for (const auto & item : data) {
            send_single(item);
        }
    } else {
        send_single(data);
    }

    return ss.str();
}

std::string format_anthropic_sse(const json & data) {
    std::ostringstream ss;

    auto send_event = [&ss](const json & event_obj) {
        if (event_obj.contains("event") && event_obj.contains("data")) {
            ss << "event: " << event_obj.at("event").get<std::string>() << "\n";
            ss << "data: " << safe_json_to_str(event_obj.at("data")) << "\n\n";
        } else {
            ss << "data: " << safe_json_to_str(event_obj) << "\n\n";
        }
    };

    if (data.is_array()) {
        for (const auto & event : data) {
            send_event(event);
        }
    } else {
        send_event(data);
    }

    return ss.str();
}

bool is_valid_utf8(const std::string & str) {
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

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt
    ) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto & chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text",     std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70, 0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(),                3*(n_batch/4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch/4) - (2 + tokens_prompt.size())));

    SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch) - 2*n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(),   tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int) extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}

server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc) {
    server_tokens result = {};

    const char * rerank_prompt = llama_model_chat_template(model, "rerank");

    if (rerank_prompt != nullptr) {
        std::string prompt = rerank_prompt;
        string_replace_all(prompt, "{query}"   , query);
        string_replace_all(prompt, "{document}", doc  );
        server_tokens tokens = tokenize_input_subprompt(vocab, mctx, prompt, false, true);
        result.push_back(tokens);
    } else {
        // Get EOS token - use SEP token as fallback if EOS is not available
        server_tokens query_tokens = tokenize_input_subprompt(vocab, mctx, query, false, false);
        server_tokens doc_tokens   = tokenize_input_subprompt(vocab, mctx, doc,   false, false);
        llama_token eos_token = llama_vocab_eos(vocab);
        if (eos_token == LLAMA_TOKEN_NULL) {
            eos_token = llama_vocab_sep(vocab);
        }

        if (llama_vocab_get_add_bos(vocab)) {
            result.push_back(llama_vocab_bos(vocab));
        }
        result.push_back(query_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
        if (llama_vocab_get_add_sep(vocab)) {
            result.push_back(llama_vocab_sep(vocab));
        }
        result.push_back(doc_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
    }

    return result;
}
