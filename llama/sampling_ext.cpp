// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#include "sampling.h"
#include "sampling_ext.h"

struct gpt_sampler *gpt_sampler_cinit(
    const struct llama_model *model, struct gpt_sampler_cparams *params)
{
    gpt_sampler_params sparams;
    sparams.top_k = params->top_k;
    sparams.top_p = params->top_p;
    sparams.min_p = params->min_p;
    sparams.tfs_z = params->tfs_z;
    sparams.typ_p = params->typical_p;
    sparams.temp = params->temp;
    sparams.penalty_last_n = params->penalty_last_n;
    sparams.penalty_repeat = params->penalty_repeat;
    sparams.penalty_freq = params->penalty_freq;
    sparams.penalty_present = params->penalty_present;
    sparams.mirostat = params->mirostat;
    sparams.mirostat_tau = params->mirostat_tau;
    sparams.mirostat_eta = params->mirostat_eta;
    sparams.penalize_nl = params->penalize_nl;
    sparams.seed = params->seed;
    sparams.grammar = params->grammar;
    return gpt_sampler_init(model, sparams);
}

void gpt_sampler_cfree(struct gpt_sampler *sampler)
{
    gpt_sampler_free(sampler);
}

void gpt_sampler_creset(struct gpt_sampler *sampler)
{
    gpt_sampler_reset(sampler);
}

llama_token gpt_sampler_csample(
    struct gpt_sampler *sampler,
    struct llama_context *ctx_main,
    int idx)
{
    return gpt_sampler_sample(sampler, ctx_main, idx);
}

void gpt_sampler_caccept(
    struct gpt_sampler *sampler,
    llama_token id,
    bool apply_grammar)
{
    gpt_sampler_accept(sampler, id, apply_grammar);
}
