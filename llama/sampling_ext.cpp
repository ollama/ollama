// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#include "sampling.h"
#include "sampling_ext.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-grammar.h"
#include "nlohmann/json.hpp"

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
