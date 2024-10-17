// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#ifndef gpt_sampler_EXT_H
#define gpt_sampler_EXT_H

#include "llama.h"

#ifdef __cplusplus
extern "C"
{
#endif

    struct gpt_sampler_cparams
    {
        int32_t top_k;
        float top_p;
        float min_p;
        float tfs_z;
        float typical_p;
        float temp;
        int32_t penalty_last_n;
        float penalty_repeat;
        float penalty_freq;
        float penalty_present;
        int32_t mirostat;
        float mirostat_tau;
        float mirostat_eta;
        bool penalize_nl;
        uint32_t seed;
        char *grammar;
    };

    struct llama_sampler *gpt_sampler_cinit(
        const struct llama_model *model,
        struct gpt_sampler_cparams *params);
    void gpt_sampler_cfree(struct llama_sampler *sampler);
    void gpt_sampler_creset(struct llama_sampler *sampler);

    llama_token gpt_sampler_csample(
        struct llama_sampler *sampler,
        struct llama_context *ctx_main,
        int idx);

    void gpt_sampler_caccept(
        struct llama_sampler *sampler,
        llama_token id,
        bool apply_grammar);

#ifdef __cplusplus
}
#endif

#endif // gpt_sampler_EXT_H
