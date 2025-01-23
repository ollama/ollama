// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#include "sampling.h"
#include "sampling_ext.h"
#include "json-schema-to-grammar.h"

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
