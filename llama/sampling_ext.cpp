// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#include "sampling.h"
#include "sampling_ext.h"
#include "json-schema-to-grammar.h"
#include "grammar.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-model-loader.h"

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
        sparams.mirostat = params->mirostat;
        sparams.mirostat_tau = params->mirostat_tau;
        sparams.mirostat_eta = params->mirostat_eta;
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
        llama_model_loader ml(std::string(fname), splits, false, false, nullptr, nullptr);
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

struct grammar *grammar_init(char* grammar) {
    if (grammar == nullptr) {
        LLAMA_LOG_ERROR("%s: null grammar input\n", __func__);
        return nullptr;
    }
    
    
    // Create vocab object
    ollama_vocab *vocab = nullptr;
    try {
        vocab = new ollama_vocab();
        if (vocab == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate vocab object\n", __func__);
            return nullptr;
        }
        
        
        // Initialize grammar with the vocab
        struct grammar *g = grammar_init_impl(vocab, grammar, "root", false, nullptr, 0, nullptr, 0);
        if (g == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize grammar\n", __func__);
            delete vocab;
            return nullptr;
        }
        
        return g;
    } catch (const std::exception& e) {
        LLAMA_LOG_ERROR("%s: exception during initialization: %s\n", __func__, e.what());
        delete vocab;
        return nullptr;
    }
}

void grammar_free(struct grammar *g) {
    if (g != nullptr) {
        if (g->vocab != nullptr) {
            delete g->vocab;
        }
        grammar_free_impl(g);
    }
}

void grammar_apply(struct grammar *g, struct llama_token_data_array *tokens) {
    grammar_apply_impl(*g, tokens);
}

void grammar_accept(struct grammar *g, llama_token id) {
    grammar_accept_impl(*g, id);
}

void grammar_add_symbol_id(struct grammar *g, const char *symbol, uint32_t id) {
    g->vocab->add_symbol_id(symbol, id);
}

void grammar_add_token_piece(struct grammar *g, uint32_t token, const char *piece) {
    g->vocab->add_token_piece(token, piece);
}

void grammar_set_eog_token(struct grammar *g, uint32_t token) {
    g->vocab->set_eog_token(token);
}