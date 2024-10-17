// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#ifndef GPT_SAMPLER_EXT_H
#define GPT_SAMPLER_EXT_H

#ifdef __cplusplus
extern "C"
{
#endif

    // Forward declaration to avoid include of "sampling.h" which has c++
    // includes
    struct gpt_sampler;

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

    struct gpt_sampler *gpt_sampler_cinit(
        const struct llama_model *model,
        struct gpt_sampler_cparams *params);
    void gpt_sampler_cfree(struct gpt_sampler *sampler);
    void gpt_sampler_creset(struct gpt_sampler *sampler);

    llama_token gpt_sampler_csample(
        struct gpt_sampler *sampler,
        struct llama_context *ctx_main,
        int idx);

    void gpt_sampler_caccept(
        struct gpt_sampler *sampler,
        llama_token id,
        bool apply_grammar);

#ifdef __cplusplus
}
#endif

#endif // GPT_SAMPLER_EXT_H
