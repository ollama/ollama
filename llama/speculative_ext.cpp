// TODO: this is a temporary wrapper to allow calling C++ code from CGo
#include <vector>
#include "speculative.h"
#include "speculative_ext.h"


struct common_speculative * common_speculative_cinit(struct llama_context * ctx_dft) {
    return common_speculative_init(ctx_dft);
}

void common_speculative_cfree(struct common_speculative * spec) {
    common_speculative_cfree(spec);
}

bool common_speculative_c_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft) {
    return common_speculative_are_compatible(ctx_tgt, ctx_dft);
}

// sample up to n_draft tokens and add them to the batch using the draft model
int common_speculative_cgen_draft(
        struct common_speculative * spec,
        struct common_speculative_cparams cparams,
        const llama_token *prompt,
        size_t prompt_size,
        llama_token id_last,
        llama_token *result,
        size_t max_result_size) {
    try {
        struct common_speculative_params params;
        std::vector<llama_token> prompt_vec(prompt, prompt+prompt_size);

        params.n_draft = cparams.n_draft;
        params.n_reuse = cparams.n_reuse;
        params.p_min = cparams.p_min;

        std::vector<llama_token> result_vec = common_speculative_gen_draft(spec, params, prompt_vec, id_last);

        size_t num_tokens_to_copy = std::min(result_vec.size(), max_result_size);

        for (size_t i = 0; i < num_tokens_to_copy; ++i) {
            result[i] = result_vec[i];
        }

        return static_cast<int>(num_tokens_to_copy);
    } catch (const std::exception &e) {
        return -1;
    }
}
