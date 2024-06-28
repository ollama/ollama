// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#include "sampling.h"
#include "sampling_ext.h"

struct llama_sampling_context *llama_sampling_cinit(struct llama_sampling_cparams *params)
{
    llama_sampling_params sparams;
    sparams.top_k = params->top_k;
    sparams.top_p = params->top_p;
    sparams.tfs_z = params->tfs_z;
    sparams.typical_p = params->typical_p;
    sparams.temp = params->temp;
    sparams.penalty_repeat = params->penalty_repeat;
    sparams.penalty_freq = params->penalty_freq;
    sparams.penalty_present = params->penalty_present;
    sparams.mirostat = params->mirostat;
    sparams.mirostat_tau = params->mirostat_tau;
    sparams.mirostat_eta = params->mirostat_eta;
    sparams.penalize_nl = params->penalize_nl;
    sparams.seed = params->seed;
    sparams.grammar = params->grammar;
    return llama_sampling_init(sparams);
}

void llama_sampling_cfree(struct llama_sampling_context *ctx)
{
    llama_sampling_free(ctx);
}

void llama_sampling_creset(struct llama_sampling_context *ctx)
{
    llama_sampling_reset(ctx);
}

llama_token llama_sampling_csample(
    struct llama_sampling_context *ctx_sampling,
    struct llama_context *ctx_main,
    struct llama_context *ctx_cfg,
    int idx)
{
    return llama_sampling_sample(ctx_sampling, ctx_main, ctx_cfg, idx);
}

void llama_sampling_caccept(
    struct llama_sampling_context *ctx_sampling,
    struct llama_context *ctx_main,
    llama_token id,
    bool apply_grammar)
{
    llama_sampling_accept(ctx_sampling, ctx_main, id, apply_grammar);
}
