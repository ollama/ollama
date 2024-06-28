// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#ifndef LLAMA_SAMPLING_EXT_H
#define LLAMA_SAMPLING_EXT_H

#include "llama.h"

#ifdef __cplusplus
extern "C"
{
#endif

    struct llama_sampling_cparams
    {
        int32_t top_k;
        float top_p;
        float tfs_z;
        float typical_p;
        float temp;
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

    struct llama_sampling_context *llama_sampling_cinit(struct llama_sampling_cparams *params);
    void llama_sampling_cfree(struct llama_sampling_context *ctx);
    void llama_sampling_creset(struct llama_sampling_context *ctx);

    llama_token llama_sampling_csample(
        struct llama_sampling_context *ctx_sampling,
        struct llama_context *ctx_main,
        struct llama_context *ctx_cfg,
        int idx);

    void llama_sampling_caccept(
        struct llama_sampling_context *ctx_sampling,
        struct llama_context *ctx_main,
        llama_token id,
        bool apply_grammar);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SAMPLING_EXT_H
