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
    struct llama_grammar *grammar_init_lazy(
        char* grammar,
        uint32_t* tokens, size_t n_tokens,
        const char** pieces,
        uint32_t* eog_tokens, size_t n_eog_tokens,
        const char** trigger_patterns, size_t n_trigger_patterns
    );
    void grammar_free(struct llama_grammar *g);
    void grammar_apply(struct llama_grammar *g, struct llama_token_data_array *tokens);
    void grammar_accept(struct llama_grammar *g, llama_token id);

    // =========================================================================
    // Tool Call Grammar Builder
    // =========================================================================
    //
    // Builds a GBNF grammar from a JSON array of tool definitions for the
    // Qwen 3.5 / Qwen3-Coder XML tool call format:
    //
    //   <tool_call>
    //   <function=name>
    //   <parameter=param>
    //   value
    //   </parameter>
    //   </function>
    //   </tool_call>
    //
    // The grammar constrains generation so the model can ONLY produce:
    //   - Function names that are one of the declared tool names
    //   - Parameter names that are declared for the matched function
    //   - Well-formed XML nesting
    //   - One or more parallel tool calls
    //
    // Parameter values are free text (no JSON schema type constraints).
    // Type mismatches are rare for well-trained models and are a tolerated
    // tradeoff — see Section 3.2 of grammar_constrained_tool_calls_plan.md.
    //
    // Expected JSON format (matches Ollama api.Tool serialization):
    //   [
    //     {
    //       "function": {
    //         "name": "get_weather",
    //         "parameters": {
    //           "type": "object",
    //           "required": ["location"],
    //           "properties": {
    //             "location": {"type": "string"},
    //             "unit": {"type": "string"}
    //           }
    //         }
    //       }
    //     }
    //   ]

    // Error codes — all negative, TOOL_GRAMMAR_OK (0) on success.
    #define TOOL_GRAMMAR_OK                   0
    #define TOOL_GRAMMAR_ERR_NULL_INPUT      -1
    #define TOOL_GRAMMAR_ERR_NULL_OUTPUT     -2
    #define TOOL_GRAMMAR_ERR_ZERO_LENGTH     -3
    #define TOOL_GRAMMAR_ERR_INVALID_JSON    -4
    #define TOOL_GRAMMAR_ERR_NOT_ARRAY       -5
    #define TOOL_GRAMMAR_ERR_EMPTY_TOOLS     -6
    #define TOOL_GRAMMAR_ERR_INVALID_TOOL    -7
    #define TOOL_GRAMMAR_ERR_TRUNCATED       -8
    #define TOOL_GRAMMAR_ERR_GRAMMAR_BUILD   -9
    #define TOOL_GRAMMAR_ERR_DUPLICATE_NAME -10

    // Build a GBNF grammar string from a JSON array of tool definitions.
    //
    // On success: writes the null-terminated GBNF string to grammar_out,
    //             returns TOOL_GRAMMAR_OK (0).
    // On failure: writes a diagnostic to error_out (if non-NULL),
    //             returns a negative error code.
    //
    // The caller owns both buffers. Neither may alias the other.
    // error_out and error_max_len may be NULL/0 to suppress diagnostics.
    int tool_call_grammar_from_json(
        const char  * tools_json,
        char        * grammar_out,
        size_t        grammar_max_len,
        char        * error_out,
        size_t        error_max_len
    );

#ifdef __cplusplus
}
#endif

#endif // SAMPLING_EXT_H
