// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#ifndef SPECULATIVE_EXT_H
#define SPECULATIVE_EXT_H

#ifdef __cplusplus
extern "C"
{
#endif

struct common_speculative_cparams {
    int n_draft; //= 16;  // max drafted tokens
    int n_reuse; // = 256;

    float p_min; // = 0.9f; // min probabiliy required to accept a token in the draft
};

struct common_speculative * common_speculative_cinit(struct llama_context * ctx_dft);
void common_speculative_cfree(struct common_speculative * spec);
bool common_speculative_c_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);
int common_speculative_cgen_draft(
        struct common_speculative * spec,
        struct common_speculative_cparams cparams,
        const llama_token *prompt,
        size_t prompt_size,
        llama_token id_last,
        llama_token *result,
        size_t max_result_size);

#ifdef __cplusplus
}
#endif

#endif // SAMPLING_EXT_H
