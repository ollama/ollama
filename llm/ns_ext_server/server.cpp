#include "utils.hpp"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
#include "httplib.h"
#include "json.hpp"

#include <cstddef>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include <signal.h>

using json = nlohmann::json;

struct server_params {
    std::string hostname = "127.0.0.1";
    std::vector<std::string> api_keys;
    std::string public_path = "examples/server/public";
    std::string chat_template = "";
    int32_t port = 8080;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
    bool slots_endpoint = true;
    bool metrics_endpoint = false;
    int n_threads_http = -1;
};

bool server_verbose = false;
bool server_log_json = true;

enum stop_type {
    STOP_FULL,
    STOP_PARTIAL,
};

// TODO: can become bool if we can't find use of more states
enum slot_state {
    IDLE,
    PROCESSING,
};

enum slot_command {
    NONE,
    LOAD_PROMPT,
    RELEASE,
};

struct slot_params {
    bool stream       = true;
    bool cache_prompt = false; // remember the prompt to avoid reprocessing all prompt

    uint32_t seed      = -1; // RNG seed
    int32_t  n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t  n_predict = -1; // new tokens to predict

    std::vector<std::string> antiprompt;

    json input_prefix;
    json input_suffix;
};

struct server_slot {
    int id;
    int task_id = -1;

    struct slot_params params;

    slot_state state = IDLE;
    slot_command command = NONE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_past      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;
    int32_t n_predict   = -1;

    int32_t n_prompt_tokens           = 0;
    int32_t n_prompt_tokens_processed = 0;

    json prompt;
    std::string generated_text;
    llama_token sampled;
    std::vector<llama_token> cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    bool infill = false;
    bool embedding = false;
    bool has_next_token = true;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;

    std::string stopping_word;

    // sampling
    struct llama_sampling_params sparams;
    llama_sampling_context *ctx_sampling = nullptr;

    // stats
    size_t n_sent_text = 0; // number of sent text character
    size_t n_sent_token_probs = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_genereration;

    double t_prompt_processing; // ms
    double t_token_generation; // ms

    // multitasks
    int multitask_id = -1;

    void reset() {
        n_prompt_tokens        = 0;
        generated_text         = "";
        truncated              = false;
        stopped_eos            = false;
        stopped_word           = false;
        stopped_limit          = false;
        stopping_word          = "";
        n_past                 = 0;
        n_sent_text            = 0;
        n_sent_token_probs     = 0;
        infill                 = false;

        generated_token_probs.clear();

    }

    bool has_budget(gpt_params &global_params) {
        if (params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (params.n_predict != -1) {
            n_remaining = params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool available() const {
        return state == IDLE && command == NONE;
    }

    bool is_processing() const {
        return (state == IDLE && command == LOAD_PROMPT) || state == PROCESSING;
    }

    void add_token_string(const completion_token_output &token) {
        if (command == RELEASE) {
            return;
        }
        cache_tokens.push_back(token.tok);
        generated_token_probs.push_back(token);
    }

    void release() {
        if (state == PROCESSING)
        {
            t_token_generation = (ne_time_us() - t_start_genereration) / 1e3;
            command = RELEASE;
        }
    }

    json get_formated_timings() {
        return json
        {
            {"prompt_n",               n_prompt_tokens_processed},
            {"prompt_ms",              t_prompt_processing},
            {"prompt_per_token_ms",    t_prompt_processing / n_prompt_tokens_processed},
            {"prompt_per_second",      1e3 / t_prompt_processing * n_prompt_tokens_processed},

            {"predicted_n",            n_decoded},
            {"predicted_ms",           t_token_generation},
            {"predicted_per_token_ms", t_token_generation / n_decoded},
            {"predicted_per_second",   1e3 / t_token_generation * n_decoded},
        };
    }

    void print_timings() const {
       char buffer[512];
        double t_token = t_prompt_processing / n_prompt_tokens_processed;
        double n_tokens_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        sprintf(buffer, "prompt eval time     = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)",
                t_prompt_processing, n_prompt_tokens_processed,
                t_token, n_tokens_second);
        LOG_INFO(buffer, {
            {"slot_id",                   id},
            {"task_id",                   task_id},
            {"t_prompt_processing",       t_prompt_processing},
            {"n_prompt_tokens_processed", n_prompt_tokens_processed},
            {"t_token",                   t_token},
            {"n_tokens_second",           n_tokens_second},
        });

        t_token = t_token_generation / n_decoded;
        n_tokens_second = 1e3 / t_token_generation * n_decoded;
        sprintf(buffer, "generation eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)",
                t_token_generation, n_decoded,
                t_token, n_tokens_second);
        LOG_INFO(buffer, {
            {"slot_id",            id},
            {"task_id",            task_id},
            {"t_token_generation", t_token_generation},
            {"n_decoded",          n_decoded},
            {"t_token",            t_token},
            {"n_tokens_second",    n_tokens_second},
        });

        sprintf(buffer, "          total time = %10.2f ms", t_prompt_processing + t_token_generation);
        LOG_INFO(buffer, {
            {"slot_id",             id},
            {"task_id",             task_id},
            {"t_prompt_processing", t_prompt_processing},
            {"t_token_generation",  t_token_generation},
            {"t_total",             t_prompt_processing + t_token_generation},
        });
    }
};

struct server_metrics {
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t n_tokens_predicted_total        = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted       = 0;
    uint64_t t_tokens_generation      = 0;


    void on_prompt_eval(const server_slot &slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
    }

    void on_prediction(const server_slot &slot) {
        n_tokens_predicted_total += slot.n_decoded;
        n_tokens_predicted       += slot.n_decoded;
        t_tokens_generation      += slot.t_token_generation;
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};

struct llama_server_context
{
    model_context *ctx = nullptr;

    gpt_params params;

    model_input  batch;

    bool multimodal         = false;
    bool all_slots_are_idle = false;
    bool add_bos_token      = true;

    int32_t n_ctx;  // total context for all clients / slots
    int32_t n_thread;

    // system prompt
    bool system_need_update = false;

    std::string              system_prompt;
    std::vector<llama_token> system_tokens;

    std::string name_user;      // this should be the antiprompt
    std::string name_assistant;

    // slots / clients
    std::vector<server_slot> slots;
    json default_generation_settings_for_props;

    llama_server_queue    queue_tasks;
    llama_server_response queue_results;

    server_metrics metrics;

    ~llama_server_context()
    {
        if (ctx)
        {
            model_free(ctx);
            ctx = nullptr;
        }
    }

    void batch_init(int32_t n_tokens_alloc) {
        batch.tokens = (model_token*) malloc(sizeof(model_token) * n_tokens_alloc);
    }

    void batch_free() {
        if (batch.tokens) free(batch.tokens);
    }

    void batch_clear() {
        batch.n_tokens = 0;
    }

    void batch_add(model_token id) {
        batch.tokens[batch.n_tokens] = id;
        batch.n_tokens++;
    }

    bool load_model(const gpt_params &params_)
    {
        params = params_;

        ctx = model_init_from_gpt_params(params);
        if (ctx == nullptr)
        {
            LOG_ERROR("unable to load model", {{"model", params.model}});
            return false;
        }

        n_ctx = model_n_ctx(ctx);
        n_thread = params.n_threads;

        add_bos_token = llama_should_add_bos_token(ctx);

        return true;
    }

    void initialize() {
        // create slots
        all_slots_are_idle = true;

        int32_t n_parallel = 1; //params.n_parallel
        const int32_t n_ctx_slot = n_ctx / n_parallel; //TODO: support n_parallel

        LOG_INFO("initializing slot!", {});
        for (int i = 0; i < n_parallel; i++)
        {
            server_slot slot;

            slot.id = i;
            slot.n_ctx = n_ctx_slot;
            slot.n_predict = params.n_predict;

            LOG_INFO("new slot", {
                {"slot_id",    slot.id},
                {"n_ctx_slot", slot.n_ctx}
            });

            slot.reset();

            slots.push_back(slot);
         }

        model_token *tokens = (model_token*) malloc(n_ctx * sizeof(model_token));
        memset((int32_t*)tokens, 0, n_ctx*sizeof(model_token));
        batch.tokens = tokens;
    }

    std::vector<llama_token> tokenize(const json & json_prompt, bool add_bos) const
    {
        // TODO: currently, we tokenize using special tokens by default
        //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
        //       but it's better compared to completely ignoring ChatML and other chat templates
        const bool TMP_FORCE_SPECIAL = true;

        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        std::vector<llama_token> prompt_tokens;

        if (json_prompt.is_array())
        {
            bool first = true;
            for (const auto& p : json_prompt)
            {
                if (p.is_string())
                {
                    auto s = p.template get<std::string>();
                    std::vector<llama_token> p;
                    if (first)
                    {
                        p = llama_tokenize(ctx, s, add_bos, TMP_FORCE_SPECIAL);
                        first = false;
                    }
                    else
                    {
                        p = llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
                    }
                    prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
                }
                else
                {
                    if (first)
                    {
                        first = false;
                    }
                    prompt_tokens.push_back(p.template get<llama_token>());
                }
            }
        }
        else
        {
            auto s = json_prompt.template get<std::string>();
            prompt_tokens = ::llama_tokenize(ctx, s, add_bos, TMP_FORCE_SPECIAL);
        }

        return prompt_tokens;
    }

    server_slot* get_slot(int id) {
        int64_t t_last = ne_time_us();
        server_slot *last_used = nullptr;

        for (server_slot & slot : slots)
        {
            if (slot.id == id && slot.available())
            {
                return &slot;
            }

            if (slot.available() && slot.t_last_used < t_last)
            {
                last_used = &slot;
                t_last = slot.t_last_used;
            }
        }

        return last_used;
    }

    bool launch_slot_with_data(server_slot* &slot, json data) {
        slot_params default_params;
        llama_sampling_params default_sparams;

        slot->params.stream             = json_value(data, "stream",            false);
        slot->params.cache_prompt       = json_value(data, "cache_prompt",      false);
        slot->params.n_predict          = json_value(data, "n_predict",         default_params.n_predict);
        slot->sparams.top_k             = json_value(data, "top_k",             default_sparams.top_k);
        slot->sparams.top_p             = json_value(data, "top_p",             default_sparams.top_p);
        slot->sparams.min_p             = json_value(data, "min_p",             default_sparams.min_p);
        slot->sparams.tfs_z             = json_value(data, "tfs_z",             default_sparams.tfs_z);
        slot->sparams.typical_p         = json_value(data, "typical_p",         default_sparams.typical_p);
        slot->sparams.temp              = json_value(data, "temperature",       default_sparams.temp);
        slot->sparams.dynatemp_range    = json_value(data, "dynatemp_range",    default_sparams.dynatemp_range);
        slot->sparams.dynatemp_exponent = json_value(data, "dynatemp_exponent", default_sparams.dynatemp_exponent);
        slot->sparams.penalty_last_n    = json_value(data, "repeat_last_n",     default_sparams.penalty_last_n);
        slot->sparams.penalty_repeat    = json_value(data, "repeat_penalty",    default_sparams.penalty_repeat);
        slot->sparams.penalty_freq      = json_value(data, "frequency_penalty", default_sparams.penalty_freq);
        slot->sparams.penalty_present   = json_value(data, "presence_penalty",  default_sparams.penalty_present);
        slot->sparams.mirostat          = json_value(data, "mirostat",          default_sparams.mirostat);
        slot->sparams.mirostat_tau      = json_value(data, "mirostat_tau",      default_sparams.mirostat_tau);
        slot->sparams.mirostat_eta      = json_value(data, "mirostat_eta",      default_sparams.mirostat_eta);
        slot->sparams.penalize_nl       = json_value(data, "penalize_nl",       default_sparams.penalize_nl);
        slot->params.n_keep             = json_value(data, "n_keep",            slot->params.n_keep);
        slot->params.seed               = json_value(data, "seed",              default_params.seed);
        slot->sparams.grammar           = json_value(data, "grammar",           default_sparams.grammar);
        slot->sparams.n_probs           = json_value(data, "n_probs",           default_sparams.n_probs);
        slot->sparams.min_keep          = json_value(data, "min_keep",          default_sparams.min_keep);

        if (slot->n_predict > 0 && slot->params.n_predict > slot->n_predict) {
            // Might be better to reject the request with a 400 ?
            LOG_WARNING("Max tokens to predict exceeds server configuration", {
                {"params.n_predict", slot->params.n_predict},
                {"slot.n_predict", slot->n_predict},
            });
            slot->params.n_predict = slot->n_predict;
        }

        // infill
        if (data.count("input_prefix") != 0)
        {
            slot->params.input_prefix = data["input_prefix"];
        }
        else
        {
            slot->params.input_prefix = "";
        }

        if (data.count("input_suffix") != 0)
        {
            slot->params.input_suffix = data["input_suffix"];
        }
        else
        {
            slot->params.input_suffix = "";
        }

        if (data.count("prompt") != 0)
        {
            slot->prompt = data["prompt"];
        }
        else
        {
            slot->prompt = "";
        }

        slot->sparams.penalty_prompt_tokens.clear();
        slot->sparams.use_penalty_prompt_tokens = false;
        const auto &penalty_prompt = data.find("penalty_prompt");
        if (penalty_prompt != data.end())
        {
            if (penalty_prompt->is_string())
            {
                const auto penalty_prompt_string = penalty_prompt->get<std::string>();
                auto penalty_tokens = llama_tokenize(ctx, penalty_prompt_string, false, false);
                slot->sparams.penalty_prompt_tokens.swap(penalty_tokens);
                if (slot->params.n_predict > 0)
                {
                    slot->sparams.penalty_prompt_tokens.reserve(slot->sparams.penalty_prompt_tokens.size() + slot->params.n_predict);
                }
                slot->sparams.use_penalty_prompt_tokens = true;
            }
            else if (penalty_prompt->is_array())
            {
                const auto n_tokens = penalty_prompt->size();
                slot->sparams.penalty_prompt_tokens.reserve(n_tokens + std::max(0, slot->params.n_predict));
                const int n_vocab = model_n_vocab(ctx);
                for (const auto &penalty_token : *penalty_prompt)
                {
                    if (penalty_token.is_number_integer())
                    {
                        const auto tok = penalty_token.get<llama_token>();
                        if (tok >= 0 && tok < n_vocab)
                        {
                            slot->sparams.penalty_prompt_tokens.push_back(tok);
                        }
                    }
                }
                slot->sparams.use_penalty_prompt_tokens = true;
            }
        }

        slot->sparams.logit_bias.clear();

        if (json_value(data, "ignore_eos", false))
        {
            slot->sparams.logit_bias[ctx->vocab.special_eos_id] = -INFINITY;
        }

        const auto &logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && logit_bias->is_array())
        {
            const int n_vocab = model_n_vocab(ctx);
            for (const auto &el : *logit_bias)
            {
                if (el.is_array() && el.size() == 2)
                {
                    float bias;
                    if (el[1].is_number())
                    {
                        bias = el[1].get<float>();
                    }
                    else if (el[1].is_boolean() && !el[1].get<bool>())
                    {
                        bias = -INFINITY;
                    }
                    else
                    {
                        continue;
                    }

                    if (el[0].is_number_integer())
                    {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab)
                        {
                            slot->sparams.logit_bias[tok] = bias;
                        }
                    }
                    else if (el[0].is_string())
                    {
                        auto toks = llama_tokenize(ctx, el[0].get<std::string>(), false, false);
                        for (auto tok : toks)
                        {
                            slot->sparams.logit_bias[tok] = bias;
                        }
                    }
                }
            }
        }

        slot->params.antiprompt.clear();

        const auto &stop = data.find("stop");
        if (stop != data.end() && stop->is_array())
        {
            for (const auto &word : *stop)
            {
                if (!word.empty())
                {
                    slot->params.antiprompt.push_back(word);
                }
            }
        }

        const auto &samplers_sequence = data.find("samplers");
        if (samplers_sequence != data.end() && samplers_sequence->is_array())
        {
            std::vector<std::string> sampler_names;
            for (const auto &sampler_name : *samplers_sequence)
            {
                if (sampler_name.is_string())
                {
                    sampler_names.emplace_back(sampler_name);
                }
            }
            slot->sparams.samplers_sequence = sampler_types_from_names(sampler_names, false);
        }
        else
        {
            slot->sparams.samplers_sequence = default_sparams.samplers_sequence;
        }

        if (multimodal)
        {
            return false;
        }

        if (slot->ctx_sampling != nullptr)
        {
            llama_sampling_free(slot->ctx_sampling);
        }
        slot->ctx_sampling = llama_sampling_init(slot->sparams);
        model_set_rng_seed(ctx, slot->params.seed);
        slot->command = LOAD_PROMPT;

        all_slots_are_idle = false;

        LOG_INFO("slot is processing task", {
            {"slot_id", slot->id},
            {"task_id", slot->task_id},
        });

        return true;
    }

    void system_prompt_update() {
        system_tokens.clear();

        if (!system_prompt.empty()) {
            system_tokens = ::llama_tokenize(ctx, system_prompt, add_bos_token, false);

            batch_clear();

            for (int i = 0; i < (int)system_tokens.size(); ++i)
            {
                batch_add(system_tokens[i]);
            }

            for (uint32_t i = 0; i < batch.n_tokens; i += params.n_batch)
            {
                const uint32_t n_tokens = std::min((uint32_t)params.n_batch, (batch.n_tokens - i));
                std::vector<model_input> batch_view = {model_input{
                    batch.tokens   + i,
                    n_tokens,
                    0,
                    batch.n_past   + i,
                    batch.n_total  + i,
                    0, 0, 0, 0,// unused
                }};
                if (model_eval(ctx, batch_view.data(), batch_view.size(), n_thread ) != 0)
                {
                    LOG_ERROR("model_eval() failed on system prompt\n", {});
                    return;
                }
            }

        }

        LOG_INFO("system prompt updated\n", {});
        system_need_update = false;
    }

    void system_prompt_notify() {
        // release all slots
        for (server_slot &slot : slots)
        {
            slot.release();
        }

        system_need_update = true;
    }

    void system_prompt_process(const json &sys_props) {
        system_prompt  = sys_props.value("prompt", "");
        name_user      = sys_props.value("anti_prompt", "");
        name_assistant = sys_props.value("assistant_name", "");


        system_prompt_notify();
    }

    static size_t find_stopping_strings(const std::string &text, const size_t last_token_size,
                                        const stop_type type, server_slot &slot)
    {
        size_t stop_pos = std::string::npos;

        for (const std::string &word : slot.params.antiprompt)
        {
            size_t pos;
            if (type == STOP_FULL)
            {
                const size_t tmp = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
                pos = text.find(word, from_pos);
            }
            else
            {
                pos = find_partial_stop_string(word, text);
            }
            if (pos != std::string::npos &&
                (stop_pos == std::string::npos || pos < stop_pos))
            {
                if (type == STOP_FULL)
                {
                    slot.stopped_word   = true;
                    slot.stopping_word  = word;
                    slot.has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    bool process_token(completion_token_output &result, server_slot &slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = llama_token_to_piece(ctx, result.tok);
        slot.sampled = result.tok;

        // search stop word and delete it
        slot.generated_text += token_str;
        slot.has_next_token = true;

        if (slot.ctx_sampling->params.use_penalty_prompt_tokens && result.tok != -1)
        {
            // we can change penalty_prompt_tokens because it is always created from scratch each request
            slot.ctx_sampling->params.penalty_prompt_tokens.push_back(result.tok);
        }

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = false;
        for (unsigned i = 1; i < 5 && i <= slot.generated_text.size(); ++i)
        {
            unsigned char c = slot.generated_text[slot.generated_text.size() - i];
            if ((c & 0xC0) == 0x80)
            {
                // continuation byte: 10xxxxxx
                continue;
            }
            if ((c & 0xE0) == 0xC0)
            {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            }
            else if ((c & 0xF0) == 0xE0)
            {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            }
            else if ((c & 0xF8) == 0xF0)
            {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }

        if (!incomplete)
        {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());
            const std::string str_test = slot.generated_text.substr(pos);
            bool is_stop_full = false;
            size_t stop_pos = find_stopping_strings(str_test, token_str.size(), STOP_FULL, slot);
            if (stop_pos != std::string::npos)
            {
                is_stop_full = true;
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            }
            else
            {
                is_stop_full = false;
                stop_pos = find_stopping_strings(str_test, token_str.size(), STOP_PARTIAL, slot);
            }

            // check if there is any token to predict
            if (stop_pos == std::string::npos || (!slot.has_next_token && !is_stop_full && stop_pos > 0))
            {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            }
            slot.add_token_string(result);
            if (slot.params.stream)
            {
                send_partial_response(slot, result);
            }
        }

        if (incomplete)
        {
            slot.has_next_token = true;
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params))
        {
            slot.stopped_limit = true;
            slot.has_next_token = false;
        }

        if (!slot.cache_tokens.empty() && result.tok == ctx->vocab.special_eos_id)
        {
            slot.stopped_eos = true;
            slot.has_next_token = false;
            LOG_VERBOSE("eos token found", {});
        }

        LOG_VERBOSE("next token", {
                                      {"token", result.tok},
                                      {"token_text", tokens_to_output_formatted_string(ctx, result.tok)},
                                      {"has_next_token", slot.has_next_token},
                                      {"n_remain", slot.n_remaining},
                                      {"num_tokens_predicted", slot.n_decoded},
                                      {"stopped_eos", slot.stopped_eos},
                                      {"stopped_word", slot.stopped_word},
                                      {"stopped_limit", slot.stopped_limit},
                                      {"stopping_word", slot.stopping_word},
                                  });

        return slot.has_next_token; // continue
    }

    void send_error(task_server& task, const std::string &error)
    {
        //LOG_INFO("task %i - error: %s\n", task.id, error.c_str());
        task_result res;
        res.id = task.id;
        res.multitask_id = task.multitask_id;
        res.stop = false;
        res.error = true;
        res.result_json = { { "content", error } };
        queue_results.send(res);
    }

    json get_formated_generation(server_slot &slot)
    {
        const auto eos_bias = slot.sparams.logit_bias.find(ctx->vocab.special_eos_id);
        const bool ignore_eos = eos_bias != slot.sparams.logit_bias.end() &&
                                eos_bias->second < 0.0f && std::isinf(eos_bias->second);
        std::vector<std::string> samplers_sequence;
        for (const auto &sampler_type : slot.sparams.samplers_sequence)
        {
            samplers_sequence.emplace_back(sampler_type_to_name_string(sampler_type));
        }

        return json {
            {"n_ctx",             slot.n_ctx},
            {"n_predict",         slot.n_predict},
            {"model",             params.model_name},
            {"seed",              slot.params.seed},
            {"temperature",       slot.sparams.temp},
            {"dynatemp_range",    slot.sparams.dynatemp_range},
            {"dynatemp_exponent", slot.sparams.dynatemp_exponent},
            {"top_k",             slot.sparams.top_k},
            {"top_p",             slot.sparams.top_p},
            {"min_p",             slot.sparams.min_p},
            {"tfs_z",             slot.sparams.tfs_z},
            {"typical_p",         slot.sparams.typical_p},
            {"repeat_last_n",     slot.sparams.penalty_last_n},
            {"repeat_penalty",    slot.sparams.penalty_repeat},
            {"presence_penalty",  slot.sparams.penalty_present},
            {"frequency_penalty", slot.sparams.penalty_freq},
            {"penalty_prompt_tokens", slot.sparams.penalty_prompt_tokens},
            {"use_penalty_prompt_tokens", slot.sparams.use_penalty_prompt_tokens},
            {"mirostat",          slot.sparams.mirostat},
            {"mirostat_tau",      slot.sparams.mirostat_tau},
            {"mirostat_eta",      slot.sparams.mirostat_eta},
            {"penalize_nl",       slot.sparams.penalize_nl},
            {"stop",              slot.params.antiprompt},
            {"n_predict",         slot.params.n_predict},
            {"n_keep",            params.n_keep},
            {"ignore_eos",        ignore_eos},
            {"stream",            slot.params.stream},
            {"logit_bias",        slot.sparams.logit_bias},
            {"n_probs",           slot.sparams.n_probs},
            {"min_keep",          slot.sparams.min_keep},
            {"grammar",           slot.sparams.grammar},
            {"samplers",          samplers_sequence}
        };
    }

    void send_partial_response(server_slot &slot, completion_token_output tkn)
    {
        task_result res;
        res.id = slot.task_id;
        res.multitask_id = slot.multitask_id;
        res.error = false;
        res.stop = false;

        res.result_json = json
        {
            {"content",    tkn.text_to_send},
            {"stop",       false},
            {"slot_id",    slot.id},
            {"multimodal", multimodal}
        };

        if (slot.sparams.n_probs > 0)
        {
            std::vector<completion_token_output> probs_output = {};
            const std::vector<llama_token> to_send_toks = llama_tokenize(ctx, tkn.text_to_send, false, false);
            size_t probs_pos      = std::min(slot.n_sent_token_probs,                       slot.generated_token_probs.size());
            size_t probs_stop_pos = std::min(slot.n_sent_token_probs + to_send_toks.size(), slot.generated_token_probs.size());
            if (probs_pos < probs_stop_pos)
            {
                probs_output = std::vector<completion_token_output>(slot.generated_token_probs.begin() + probs_pos, slot.generated_token_probs.begin() + probs_stop_pos);
            }
            slot.n_sent_token_probs = probs_stop_pos;
            res.result_json["completion_probabilities"] = probs_vector_to_json(ctx, probs_output);
        }

        queue_results.send(res);
    }

    void send_final_response(server_slot &slot)
    {
        task_result res;
        res.id = slot.task_id;
        res.multitask_id = slot.multitask_id;
        res.error = false;
        res.stop = true;

        res.result_json = json
        {
            {"content",             !slot.params.stream ? slot.generated_text : ""},
            {"slot_id",             slot.id},
            {"stop",                true},
            {"model",               params.model_name},
            {"tokens_predicted",    slot.n_decoded},
            {"tokens_evaluated",    slot.n_prompt_tokens},
            {"generation_settings", get_formated_generation(slot)},
            {"prompt",              slot.prompt},
            {"truncated",           slot.truncated},
            {"stopped_eos",         slot.stopped_eos},
            {"stopped_word",        slot.stopped_word},
            {"stopped_limit",       slot.stopped_limit},
            {"stopping_word",       slot.stopping_word},
            {"tokens_cached",       slot.n_past},
            {"timings",             slot.get_formated_timings()}
        };

        if (slot.sparams.n_probs > 0)
        {
            std::vector<completion_token_output> probs = {};
            if (!slot.params.stream && slot.stopped_word)
            {
                const std::vector<llama_token> stop_word_toks = llama_tokenize(ctx, slot.stopping_word, false, false);
                probs = std::vector<completion_token_output>(slot.generated_token_probs.begin(), slot.generated_token_probs.end() - stop_word_toks.size());
            }
            else
            {
                probs = std::vector<completion_token_output>(
                                    slot.generated_token_probs.begin(),
                                    slot.generated_token_probs.end());
            }
            res.result_json["completion_probabilities"] = probs_vector_to_json(ctx, probs);
        }

        queue_results.send(res);
    }

    void send_embedding(server_slot & slot, const model_input & batch)
    {
        task_result res;
        res.id = slot.task_id;
        res.multitask_id = slot.multitask_id;
        res.error = false;
        res.stop = true;

        const int n_embd = ctx->model.hparams.n_embd;

        {
            LOG_WARNING("embedding disabled", {{"params.embedding", params.embedding}});
            res.result_json = json
            {
                {"embedding", std::vector<float>(n_embd, 0.0f)},
            };
        }
        queue_results.send(res);
    }

    void request_completion(int task_id, json data, bool infill, bool embedding, int multitask_id)
    {
        task_server task;
        task.id = task_id;
        task.target_id = 0;
        task.data = std::move(data);
        task.infill_mode = infill;
        task.embedding_mode = embedding;
        task.type = TASK_TYPE_COMPLETION;
        task.multitask_id = multitask_id;

        // when a completion task's prompt array is not a singleton, we split it into multiple requests
        // otherwise, it's a single-prompt task, we actually queue it
        // if there's numbers in the prompt array it will be treated as an array of tokens
        if (task.data.count("prompt") != 0 && task.data.at("prompt").size() > 1) {
            bool numbers = false;
            for (const auto& e : task.data.at("prompt")) {
                if (e.is_number()) {
                    numbers = true;
                    break;
                }
            }

            // NOTE: split_multiprompt_task() does not handle a mix of strings and numbers,
            // it will completely stall the server. I don't know where the bug for this is.
            //
            // if there are numbers, it needs to be treated like a single prompt,
            // queue_tasks handles a mix of strings and numbers just fine.
            if (numbers) {
                queue_tasks.post(task);
            } else {
                split_multiprompt_task(task_id, task);
            }
        } else {
            // an empty prompt can make slot become buggy
            if (task.data.contains("prompt") && task.data["prompt"].is_string() && task.data["prompt"].get<std::string>().empty()) {
                task.data["prompt"] = " "; // add a space so that we have one token
            }
            queue_tasks.post(task);
        }
    }

    void request_cancel(int task_id)
    {
        task_server task;
        task.type = TASK_TYPE_CANCEL;
        task.target_id = task_id;
        queue_tasks.post(task);
    }

    void split_multiprompt_task(int multitask_id, task_server& multiprompt_task)
    {
        int prompt_count = multiprompt_task.data.at("prompt").size();
        if (prompt_count <= 1) {
            send_error(multiprompt_task, "error while handling multiple prompts");
            return;
        }

        // generate all the ID for subtask
        std::vector<int> subtask_ids(prompt_count);
        for (int i = 0; i < prompt_count; i++)
        {
            subtask_ids[i] = queue_tasks.get_new_id();
        }

        // queue up the multitask so we can track its subtask progression
        queue_tasks.add_multitask(multitask_id, subtask_ids);

        // add subtasks
        for (int i = 0; i < prompt_count; i++)
        {
            json subtask_data = multiprompt_task.data;
            subtask_data["prompt"] = subtask_data["prompt"][i];

            // subtasks inherit everything else (infill mode, embedding mode, etc.)
            request_completion(subtask_ids[i], subtask_data, multiprompt_task.infill_mode, multiprompt_task.embedding_mode, multitask_id);
        }
    }

    void process_single_task(task_server& task)
    {
        switch (task.type)
        {
            case TASK_TYPE_COMPLETION: {
                server_slot *slot = get_slot(json_value(task.data, "slot_id", -1));
                if (slot == nullptr)
                {
                    // if no slot is available, we defer this task for processing later
                    LOG_VERBOSE("no slot is available", {{"task_id", task.id}});
                    queue_tasks.defer(task);
                    break;
                }

                if (task.data.contains("system_prompt"))
                {
                    if (!all_slots_are_idle) {
                        send_error(task, "system prompt can only be updated when all slots are idle");
                        break;
                    }
                    system_prompt_process(task.data["system_prompt"]);

                    // reset cache_tokens for all slots
                    for (server_slot &slot : slots)
                    {
                        slot.cache_tokens.clear();
                        slot.n_past    = 0;
                    }
                }

                slot->reset();

                slot->infill       = task.infill_mode;
                slot->embedding    = task.embedding_mode;
                slot->task_id      = task.id;
                slot->multitask_id = task.multitask_id;

                if (!launch_slot_with_data(slot, task.data))
                {
                    // send error result
                    send_error(task, "internal_error");
                    break;
                }
            } break;
            case TASK_TYPE_CANCEL: { // release slot linked with the task id
                for (auto & slot : slots)
                {
                    if (slot.task_id == task.target_id)
                    {
                        slot.release();
                        break;
                    }
                }
            } break;
            case TASK_TYPE_NEXT_RESPONSE: {
                // do nothing
            } break;
            case TASK_TYPE_METRICS: {
                json slots_data        = json::array();
                int n_idle_slots       = 0;
                int n_processing_slots = 0;

                for (server_slot &slot: slots) {
                    json slot_data = get_formated_generation(slot);
                    slot_data["id"] = slot.id;
                    slot_data["task_id"] = slot.task_id;
                    slot_data["state"] = slot.state;
                    slot_data["prompt"] = slot.prompt;
                    slot_data["next_token"] = {
                            {"has_next_token",       slot.has_next_token},
                            {"n_remain",             slot.n_remaining},
                            {"num_tokens_predicted", slot.n_decoded},
                            {"stopped_eos",          slot.stopped_eos},
                            {"stopped_word",         slot.stopped_word},
                            {"stopped_limit",        slot.stopped_limit},
                            {"stopping_word",        slot.stopping_word},
                    };
                    if (slot_data["state"] == IDLE) {
                        n_idle_slots++;
                    } else {
                        n_processing_slots++;
                    }
                    slots_data.push_back(slot_data);
                }
                LOG_INFO("slot data", {
                    {"task_id",            task.id},
                    {"n_idle_slots",       n_idle_slots},
                    {"n_processing_slots", n_processing_slots}
                });
                LOG_VERBOSE("slot data", {
                    {"task_id",            task.id},
                    {"n_idle_slots",       n_idle_slots},
                    {"n_processing_slots", n_processing_slots},
                    {"slots",              slots_data}
                });
                task_result res;
                res.id = task.id;
                res.multitask_id = task.multitask_id;
                res.stop = true;
                res.error = false;
                res.result_json = {
                        { "idle",                            n_idle_slots       },
                        { "processing",                      n_processing_slots },
                        { "deferred",                        queue_tasks.queue_tasks_deferred.size() },

                        { "n_prompt_tokens_processed_total", metrics.n_prompt_tokens_processed_total},
                        { "n_tokens_predicted_total",        metrics.n_tokens_predicted_total},

                        { "n_prompt_tokens_processed",       metrics.n_prompt_tokens_processed},
                        { "t_prompt_processing",             metrics.t_prompt_processing},
                        { "n_tokens_predicted",              metrics.n_tokens_predicted},
                        { "t_tokens_generation",             metrics.t_tokens_generation},

                        { "slots",                           slots_data },
                };
                metrics.reset_bucket();
                queue_results.send(res);
            } break;
        }
    }

    void on_finish_multitask(task_multi& multitask)
    {
        // all subtasks done == multitask is done
        task_result result;
        result.id = multitask.id;
        result.stop = true;
        result.error = false;

        // collect json results into one json result
        std::vector<json> result_jsons;
        for (auto& subres : multitask.results)
        {
            result_jsons.push_back(subres.result_json);
            result.error = result.error && subres.error;
        }
        result.result_json = json{ { "results", result_jsons } };
        queue_results.send(result);
    }

    bool update_slots() {
        if (system_need_update)
        {
            LOG_INFO("updating system prompt", {});
            system_prompt_update();
        }

        batch_clear();

        if (all_slots_are_idle)
        {
            return true;
        }

        LOG_VERBOSE("posting NEXT_RESPONSE", {});
        task_server task;
        task.type = TASK_TYPE_NEXT_RESPONSE;
        task.target_id = -1;
        queue_tasks.post(task);

        for (server_slot &slot : slots)
        {
            if (slot.is_processing() && system_tokens.size() + slot.cache_tokens.size() >= (size_t) slot.n_ctx)
            {
                // Shift context
                const int n_keep    = slot.params.n_keep + add_bos_token;
                const int n_left    = (int) system_tokens.size() + slot.n_past - n_keep;
                const int n_discard = n_left / 2;

                LOG_INFO("slot context shift", {
                    {"slot_id",         slot.id},
                    {"task_id",         slot.task_id},
                    {"n_keep",          n_keep},
                    {"n_left",          n_left},
                    {"n_discard",       n_discard},
                    {"n_ctx",           n_ctx},
                    {"n_past",          slot.n_past},
                    {"n_system_tokens", system_tokens.size()},
                    {"n_cache_tokens",  slot.cache_tokens.size()}
                });

                for (size_t i = n_keep + n_discard; i < slot.cache_tokens.size(); i++)
                {
                    slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
                }

                slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);

                slot.n_past -= n_discard;

                slot.truncated = true;
            }
        }

        // decode any currently ongoing sequences
        LOG_VERBOSE("decoding ongoing sequences", {});
        for (auto & slot : slots)
        {
            // release the slot
            if (slot.command == RELEASE)
            {
                slot.state = IDLE;
                slot.command = NONE;
                slot.t_last_used = ne_time_us();

                LOG_INFO("slot released", {
                    {"slot_id",         slot.id},
                    {"task_id",         slot.task_id},
                    {"n_ctx",           n_ctx},
                    {"n_past",          slot.n_past},
                    {"n_system_tokens", system_tokens.size()},
                    {"n_cache_tokens",  slot.cache_tokens.size()},
                    {"truncated",       slot.truncated}
                });
                queue_tasks.notify_slot_changed();

                continue;
            }

            if (slot.state == IDLE)
            {
                continue;
            }

            slot.i_batch = batch.n_tokens;

            const int32_t slot_npast = slot.n_past;

            // TODO: we always have to take into account the "system_tokens"
            //       this is not great and needs to be improved somehow
            batch_add(slot.sampled);
            slot.n_past += 1;
        }

        // process in chunks of params.n_batch
        uint32_t n_batch = params.n_batch;

        // assign workload to the slots
        if (params.cont_batching || batch.n_tokens == 0)
        {
            for (auto & slot : slots)
            {
                const bool has_prompt = slot.prompt.is_array() || (slot.prompt.is_string() && !slot.prompt.get<std::string>().empty());

                // empty prompt passed -> release the slot and send empty response
                // note: infill mode allows empty prompt
                if (slot.state == IDLE && slot.command == LOAD_PROMPT && !has_prompt && !slot.infill)
                {
                    slot.release();
                    slot.print_timings();
                    send_final_response(slot);
                    continue;
                }

                // need process the prompt
                if (slot.state == IDLE && slot.command == LOAD_PROMPT)
                {
                    slot.state = PROCESSING;
                    slot.command = NONE;
                    std::vector<llama_token> prompt_tokens;
                    slot.t_start_process_prompt = ne_time_us();
                    slot.t_start_genereration = 0;

                    if (slot.infill)
                    {
                        bool suff_rm_leading_spc = true;
                        if (params.input_suffix.find_first_of(' ') == 0 && params.input_suffix.size() > 1)
                        {
                            params.input_suffix.erase(0, 1);
                            suff_rm_leading_spc = false;
                        }
                        auto prefix_tokens = tokenize(slot.params.input_prefix, false);
                        auto suffix_tokens = tokenize(slot.params.input_suffix, false);

                        const int space_token = 29871; // TODO: this should not be hardcoded
                        if (suff_rm_leading_spc && !suffix_tokens.empty() && suffix_tokens[0] == space_token) {
                            suffix_tokens.erase(suffix_tokens.begin());
                        }

                        prefix_tokens.insert(prefix_tokens.begin(), ctx->vocab.special_prefix_id);
                        prefix_tokens.insert(prefix_tokens.begin(), ctx->vocab.special_bos_id); // always add BOS
                        prefix_tokens.insert(prefix_tokens.end(),   ctx->vocab.special_suffix_id);
                        prefix_tokens.insert(prefix_tokens.end(),   suffix_tokens.begin(), suffix_tokens.end());
                        prefix_tokens.push_back(ctx->vocab.special_middle_id);
                        prompt_tokens = prefix_tokens;
                    }
                    else
                    {
                        prompt_tokens = tokenize(slot.prompt, system_prompt.empty() && add_bos_token);  // add BOS if there isn't system prompt
                    }

                    slot.n_prompt_tokens = prompt_tokens.size();

                    if (slot.params.n_keep < 0)
                    {
                        slot.params.n_keep = slot.n_prompt_tokens;
                    }
                    slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                    // if input prompt is too big, truncate it, if group attention self-extend is disabled
                    if (slot.n_prompt_tokens >= slot.n_ctx)
                    {
                        const int n_left = slot.n_ctx - slot.params.n_keep;
                        const int n_block_size = n_left / 2;
                        const int erased_blocks = (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                        std::vector<llama_token> new_tokens(
                            prompt_tokens.begin(),
                            prompt_tokens.begin() + slot.params.n_keep);
                        new_tokens.insert(
                            new_tokens.end(),
                            prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size,
                            prompt_tokens.end());

                        //LOG_VERBOSE("input truncated", {
                        //    {"n_ctx",      slot.n_ctx},
                        //    {"n_keep",     slot.params.n_keep},
                        //    {"n_left",     n_left},
                        //    {"new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend())},
                        //});
                        slot.truncated = true;
                        prompt_tokens = new_tokens;

                        slot.n_prompt_tokens = prompt_tokens.size();
                        NE_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
                    }

                    if (!slot.params.cache_prompt)
                    {
                        llama_sampling_reset(slot.ctx_sampling);

                        slot.n_past    = 0;
                        slot.n_prompt_tokens_processed = slot.n_prompt_tokens;
                    }
                    else
                    {
                        // push the prompt into the sampling context (do not apply grammar)
                        for (auto &token : prompt_tokens)
                        {
                            llama_sampling_accept(slot.ctx_sampling, ctx, token, false);
                        }

                        slot.n_past = common_part(slot.cache_tokens, prompt_tokens);

                        // the last token of the cache is not in the KV cache until the next call to llama_decode
                        // (it was sampled, pushed into the "cache_tokens", but not yet put in the context)
                        if (slot.n_past > 0 && slot.n_past == (int32_t) slot.cache_tokens.size())
                        {
                            slot.n_past -= 1;
                        }

                        slot.n_prompt_tokens_processed = slot.n_prompt_tokens - slot.n_past;

                        LOG_INFO("slot progression", {
                            { "slot_id",    slot.id },
                            { "task_id",    slot.task_id },
                            { "n_past",     slot.n_past },
                            { "n_prompt_tokens_processed", slot.n_prompt_tokens_processed }
                        });
                    }

                    slot.cache_tokens = prompt_tokens;

                    if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0)
                    {
                        // we have to evaluate at least 1 token to generate logits.
                        LOG_INFO("we have to evaluate at least 1 token to generate logits", {
                            { "slot_id", slot.id },
                            { "task_id", slot.task_id }
                        });
                        slot.n_past--;
                    }

                    int p0 = (int) system_tokens.size() + slot.n_past;
                    LOG_INFO("kv cache rm [p0, end)", {
                        { "slot_id", slot.id },
                        { "task_id", slot.task_id },
                        { "p0",      p0 }
                    });

                    //LOG_VERBOSE("prompt ingested", {
                    //                                {"n_past",  slot.n_past},
                    //                                {"cached",  tokens_to_str(ctx, slot.cache_tokens.cbegin(), slot.cache_tokens.cbegin() + slot.n_past)},
                    //                                {"to_eval", tokens_to_str(ctx, slot.cache_tokens.cbegin() + slot.n_past, slot.cache_tokens.cend())},
                    //                            });

                    std::vector<llama_token> prefix_tokens = prompt_tokens;

                    int32_t slot_npast = slot.n_past;

                    for (; slot.n_past < (int) prefix_tokens.size(); ++slot.n_past)
                    {
                        batch_add(prefix_tokens[slot.n_past]);
                        slot_npast++;
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0)
                    {
                        ctx->logits[batch.n_tokens - 1] = true;
                    }

                    slot.n_decoded = 0;
                    slot.i_batch   = batch.n_tokens - 1;
                }
            }
        }

        if (batch.n_tokens == 0)
        {
            all_slots_are_idle = true;
            return true;
        }

        for (uint32_t i = 0; i < batch.n_tokens; i += n_batch)
        {
            const uint32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            std::vector<model_input> batch_view = {model_input{
                batch.tokens   + i,
                n_tokens,
                0,
                batch.n_past   + i,
                batch.n_total  + i,
                0, 0, 0, 0,// unused
            }};

            const int ret = model_eval(ctx, batch_view.data(), batch_view.size(), n_thread);

            if (ret != 0)
            {
                if (n_batch == 1 || ret < 0)
                {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    //LOG_INFO("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return false;
                }

                //LOG_INFO("%s : failed to find free space in the KV cache, retrying with smaller n_batch = %d\n", __func__, n_batch / 2);

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;
                continue;
            }

            for (auto & slot : slots)
            {
                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens))
                {
                    continue;
                }

                // prompt evaluated for embedding
                if (slot.embedding)
                {
                    send_embedding(slot, batch_view.at(0));
                    slot.release();
                    slot.i_batch = -1;
                    continue;
                }

                completion_token_output result;
                const llama_token id = llama_sampling_sample(slot.ctx_sampling, ctx, NULL, slot.i_batch - i);

                llama_sampling_accept(slot.ctx_sampling, ctx, id, true);

                slot.n_decoded += 1;
                if (slot.n_decoded == 1)
                {
                    slot.t_start_genereration = ne_time_us();
                    slot.t_prompt_processing = (slot.t_start_genereration - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                llama_token_data_array cur_p = { slot.ctx_sampling->cur.data(), slot.ctx_sampling->cur.size(), false };
                result.tok = id;

                const int32_t n_probs = slot.sparams.n_probs;
                if (slot.sparams.temp <= 0 && n_probs > 0)
                {
                    // for llama_sample_token_greedy we need to sort candidates
                    llama_sample_softmax(ctx, &cur_p);
                }

                for (size_t i = 0; i < std::min(cur_p.size, (size_t)n_probs); ++i)
                {
                    result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
                }

                if (!process_token(result, slot))
                {
                    slot.release();
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                }

                slot.i_batch = -1;
            }
        }

        LOG_VERBOSE("slots updated", {});
        return true;
    }

};

static json format_tokenizer_response(const std::vector<llama_token> &tokens)
{
    return json {
        {"tokens", tokens}
    };
}

static json format_detokenized_response(std::string content)
{
    return json {
        {"content", content}
    };
}
