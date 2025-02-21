// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#ifndef SAMPLING_EXT_H
#define SAMPLING_EXT_H

#ifdef __cplusplus
extern "C"
{
#endif

    // Forward declaration to avoid include of "sampling.h" which has c++
    // includes
    struct common_sampler;
    struct common_sampler_cparams {
        int32_t top_k;
        float top_p;
        float min_p;
        float typical_p;
        float temp;
        int32_t penalty_last_n;
        float penalty_repeat;
        float penalty_freq;
        float penalty_present;
        int32_t mirostat;
        float mirostat_tau;
        float mirostat_eta;
        uint32_t seed;
        char *grammar;
    };

    struct common_sampler *common_sampler_cinit(const struct llama_model *model, struct common_sampler_cparams *params);
    void common_sampler_cfree(struct common_sampler *sampler);
    void common_sampler_creset(struct common_sampler *sampler);
    void common_sampler_caccept(struct common_sampler *sampler, llama_token id, bool apply_grammar);
    llama_token common_sampler_csample(struct common_sampler *sampler, struct llama_context *ctx, int idx);

    int schema_to_grammar(const char *json_schema, char *grammar, size_t max_len);

#ifdef __cplusplus
}
#endif

#endif // SAMPLING_EXT_H
