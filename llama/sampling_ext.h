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
        uint32_t seed;
        char *grammar;
    };

    struct common_sampler *common_sampler_cinit(const struct llama_model *model, struct common_sampler_cparams *params);
    void common_sampler_cfree(struct common_sampler *sampler);
    void common_sampler_creset(struct common_sampler *sampler);
    void common_sampler_caccept(struct common_sampler *sampler, llama_token id, bool apply_grammar);
    llama_token common_sampler_csample(struct common_sampler *sampler, struct llama_context *ctx, int idx);

    int schema_to_grammar(const char *json_schema, char *grammar, size_t max_len);


    struct llama_grammar *grammar_init(char* grammar, uint32_t* tokens, size_t n_tokens, const char** pieces, uint32_t* eog_tokens, size_t n_eog_tokens);
    void grammar_free(struct llama_grammar *g);
    void grammar_apply(struct llama_grammar *g, struct llama_token_data_array *tokens);
    void grammar_accept(struct llama_grammar *g, llama_token id);


#ifdef __cplusplus
}
#endif

#endif // SAMPLING_EXT_H
