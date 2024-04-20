#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

#define LLAMA_MAX_RNG_STATE (64*1024)

#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LLAMA_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 5

#define LLAMA_STATE_SEQ_MAGIC   LLAMA_FILE_MAGIC_GGSQ
#define LLAMA_STATE_SEQ_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_model;
    struct llama_context;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;

    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0, // For models without vocab
        LLAMA_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
    };

    // note: these values should be synchronized with ggml_rope
    // TODO: maybe move this enum to ggml.h (ggml_rope_type)
    enum llama_rope_type {
        LLAMA_ROPE_TYPE_NONE = -1,
        LLAMA_ROPE_TYPE_NORM =  0,
        LLAMA_ROPE_TYPE_NEOX =  2,
        LLAMA_ROPE_TYPE_GLM  =  4,
    };

    enum llama_token_type {
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMA_TOKEN_TYPE_NORMAL       = 1,
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMA_TOKEN_TYPE_CONTROL      = 3,
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMA_TOKEN_TYPE_UNUSED       = 5,
        LLAMA_TOKEN_TYPE_BYTE         = 6,
    };

    // model file types
    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_YARN,
    };

    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS  = 2,
    };

    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE    = 0, // single GPU
        LLAMA_SPLIT_MODE_LAYER   = 1, // split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW     = 2, // split rows across GPUs
    };

    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;

    typedef struct llama_token_data_array {
        llama_token_data * data;
        size_t size;
        bool sorted;
    } llama_token_data_array;

    typedef bool (*llama_progress_callback)(float progress, void *ctx);

    // Input data for llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    // - seq_id : the sequence to which the respective token belongs
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits; // TODO: rename this to "output"

        // NOTE: helpers for smooth API transition - can be deprecated in the future
        //       for future-proof code, use the above fields instead and ignore everything below
        //
        // pos[i] = all_pos_0 + i*all_pos_1
        //
        llama_pos    all_pos_0;  // used if pos == NULL
        llama_pos    all_pos_1;  // used if pos == NULL
        llama_seq_id all_seq_id; // used if seq_id == NULL
    } llama_batch;

    enum llama_model_kv_override_type {
        LLAMA_KV_OVERRIDE_TYPE_INT,
        LLAMA_KV_OVERRIDE_TYPE_FLOAT,
        LLAMA_KV_OVERRIDE_TYPE_BOOL,
    };

    struct llama_model_kv_override {
        char key[128];
        enum llama_model_kv_override_type tag;
        union {
            int64_t int_value;
            double float_value;
            bool bool_value;
        };
    };

    struct llama_model_params {
        int32_t n_gpu_layers; // number of layers to store in VRAM
        enum llama_split_mode split_mode; // how to split the model across multiple GPUs

        // main_gpu interpretation depends on split_mode:
        // LLAMA_SPLIT_NONE: the GPU that is used for the entire model
        // LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results
        // LLAMA_SPLIT_LAYER: ignored
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        llama_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct llama_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only; // only load the vocabulary, no weights
        bool use_mmap;   // use mmap if possible
        bool use_mlock;  // force system to keep model in RAM
    };

    struct llama_context_params {
        uint32_t seed;              // RNG seed, -1 for random
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        uint32_t n_threads;         // number of threads to use for generation
        uint32_t n_threads_batch;   // number of threads to use for batch processing

        enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
        enum llama_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
                                                        // (ignored if no pooling layer)

        // ref: https://github.com/ggerganov/llama.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // defragment the KV cache if holes/size > thold, < 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache
        enum ggml_type type_v; // data type for V cache

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool logits_all;  // the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU

        // Abort callback
        // if it returns true, execution of llama_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;
    };

    // model quantization parameters
    typedef struct llama_model_quantize_params {
        int32_t nthread;                     // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llama_ftype ftype;              // quantize to this llama_ftype
        enum ggml_type output_tensor_type;   // output tensor type
        enum ggml_type token_embedding_type; // itoken embeddings tensor type
        bool allow_requantize;               // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;         // quantize output.weight
        bool only_copy;                      // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                           // quantize all tensors to the default type
        void * imatrix;                      // pointer to importance matrix data
        void * kv_overrides;                 // pointer to vector containing overrides
    } llama_model_quantize_params;

    // grammar types
    struct llama_grammar;

    // grammar element type
    enum llama_gretype {
        // end of rule definition
        LLAMA_GRETYPE_END            = 0,

        // start of alternate definition for rule
        LLAMA_GRETYPE_ALT            = 1,

        // non-terminal element: reference to rule
        LLAMA_GRETYPE_RULE_REF       = 2,

        // terminal element: character (code point)
        LLAMA_GRETYPE_CHAR           = 3,

        // inverse char(s) ([^a], [^a-b] [^abc])
        LLAMA_GRETYPE_CHAR_NOT       = 4,

        // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
        // be an inclusive range ([a-z])
        LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,

        // modifies a preceding LLAMA_GRETYPE_CHAR or
        // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
        LLAMA_GRETYPE_CHAR_ALT       = 6,
    };

    typedef struct llama_grammar_element {
        enum llama_gretype type;
        uint32_t           value; // Unicode code point or rule ID
    } llama_grammar_element;

    // performance timing information
    struct llama_timings {
        double t_start_ms;
        double t_end_ms;
        double t_load_ms;
        double t_sample_ms;
        double t_p_eval_ms;
        double t_eval_ms;

        int32_t n_sample;
        int32_t n_p_eval;
        int32_t n_eval;
    };

    // used in chat template
    typedef struct llama_chat_message {
        const char * role;
        const char * content;
    } llama_chat_message;

    // Helpers for getting default parameters
    LLAMA_API struct llama_model_params llama_model_default_params(void);
    LLAMA_API struct llama_context_params llama_context_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);

    // Initialize the llama + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    LLAMA_API void llama_backend_init(void);

    //optional:
    LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);

    // Call once at the end of the program - currently only used for MPI
    LLAMA_API void llama_backend_free(void);

    LLAMA_API struct llama_model * llama_load_model_from_file(
                             const char * path_model,
            struct llama_model_params     params);

    LLAMA_API void llama_free_model(struct llama_model * model);

    LLAMA_API struct llama_context * llama_new_context_with_model(
                     struct llama_model * model,
            struct llama_context_params   params);

    // Frees all allocated memory
    LLAMA_API void llama_free(struct llama_context * ctx);

    LLAMA_API int64_t llama_time_us(void);

    LLAMA_API size_t llama_max_devices(void);

    LLAMA_API bool llama_supports_mmap       (void);
    LLAMA_API bool llama_supports_mlock      (void);
    LLAMA_API bool llama_supports_gpu_offload(void);

    LLAMA_API const struct llama_model * llama_get_model(const struct llama_context * ctx);

    LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);

    LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_model * model);
    LLAMA_API enum llama_rope_type  llama_rope_type (const struct llama_model * model);

    LLAMA_API int32_t llama_n_vocab    (const struct llama_model * model);
    LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model);
    LLAMA_API int32_t llama_n_embd     (const struct llama_model * model);
    LLAMA_API int32_t llama_n_layer    (const struct llama_model * model);

    // Get the model's RoPE frequency scaling factor
    LLAMA_API float llama_rope_freq_scale_train(const struct llama_model * model);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);

    // Get metadata key name by index
    LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    LLAMA_API uint64_t llama_model_size(const struct llama_model * model);

    // Returns the total number of parameters in the model
    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);

    // Get a llama model tensor
    LLAMA_API struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name);

    // Returns 0 on success
    LLAMA_API uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params);

    // Apply a LoRA adapter to a loaded model
    // path_base_model is the path to a higher quality model to use as a base for
    // the layers modified by the adapter. Can be NULL to use the current loaded model.
    // The model needs to be reloaded before applying a new adapter, otherwise the adapter
    // will be applied on top of the previous one
    // Returns 0 on success
    LLAMA_API int32_t llama_model_apply_lora_from_file(
            const struct llama_model * model,
                          const char * path_lora,
                               float   scale,
                          const char * path_base_model,
                             int32_t   n_threads);

    // Apply a loaded control vector to a llama_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See llama_control_vector_load in common to load a control vector.
    LLAMA_API int32_t llama_control_vector_apply(
            struct llama_context * lctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // KV cache
    //

    // Information associated with an individual cell in the KV cache view.
    struct llama_kv_cache_view_cell {
        // The position for this cell. Takes KV cache shifts into account.
        // May be negative if the cell is not populated.
        llama_pos pos;
    };

    // An updateable view of the KV cache.
    struct llama_kv_cache_view {
        // Number of KV cache cells. This will be the same as the context size.
        int32_t n_cells;

        // Maximum number of sequences that can exist in a cell. It's not an error
        // if there are more sequences in a cell than this value, however they will
        // not be visible in the view cells_sequences.
        int32_t n_seq_max;

        // Number of tokens in the cache. For example, if there are two populated
        // cells, the first with 1 sequence id in it and the second with 2 sequence
        // ids then you'll have 3 tokens.
        int32_t token_count;

        // Number of populated cache cells.
        int32_t used_cells;

        // Maximum contiguous empty slots in the cache.
        int32_t max_contiguous;

        // Index to the start of the max_contiguous slot range. Can be negative
        // when cache is full.
        int32_t max_contiguous_idx;

        // Information for an individual cell.
        struct llama_kv_cache_view_cell * cells;

        // The sequences for each cell. There will be n_seq_max items per cell.
        llama_seq_id * cells_sequences;
    };

    // Create an empty KV cache view. (use only for debugging purposes)
    LLAMA_API struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_seq_max);

    // Free a KV cache view. (use only for debugging purposes)
    LLAMA_API void llama_kv_cache_view_free(struct llama_kv_cache_view * view);

    // Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
    LLAMA_API void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view);

    // Returns the number of tokens in the KV cache (slow, use only for debug)
    // If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    LLAMA_API int32_t llama_get_kv_cache_token_count(const struct llama_context * ctx);

    // Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
    LLAMA_API int32_t llama_get_kv_cache_used_cells(const struct llama_context * ctx);

    // Clear the KV cache
    LLAMA_API void llama_kv_cache_clear(
            struct llama_context * ctx);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLAMA_API bool llama_kv_cache_seq_rm(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_cp(
            struct llama_context * ctx,
                    llama_seq_id   seq_id_src,
                    llama_seq_id   seq_id_dst,
                       llama_pos   p0,
                       llama_pos   p1);

    // Removes all tokens that do not belong to the specified sequence
    LLAMA_API void llama_kv_cache_seq_keep(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_cache_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_add(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                       llama_pos   delta);

    // Integer division of the positions by factor of `d > 1`
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_cache_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_div(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                             int   d);

    // Returns the largest position present in the KV cache for the specified sequence
    LLAMA_API llama_pos llama_kv_cache_seq_pos_max(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Defragment the KV cache
    // This will be applied:
    //   - lazily on next llama_decode()
    //   - explicitly with llama_kv_cache_update()
    LLAMA_API void llama_kv_cache_defrag(struct llama_context * ctx);

    // Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
    LLAMA_API void llama_kv_cache_update(struct llama_context * ctx);

    //
    // State / sessions
    //

    // Returns the maximum size in bytes of the state (rng, logits, embedding
    // and kv_cache) - will often be smaller after compacting tokens
    LLAMA_API size_t llama_state_get_size(const struct llama_context * ctx);
    LLAMA_API DEPRECATED(size_t llama_get_state_size(const struct llama_context * ctx),
        "use llama_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLAMA_API size_t llama_state_get_data(
            struct llama_context * ctx,
                         uint8_t * dst);
    LLAMA_API DEPRECATED(size_t llama_copy_state_data(
            struct llama_context * ctx,
                         uint8_t * dst),
        "use llama_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLAMA_API size_t llama_state_set_data(
            struct llama_context * ctx,
                   const uint8_t * src);
    LLAMA_API DEPRECATED(size_t llama_set_state_data(
            struct llama_context * ctx,
                   const uint8_t * src),
        "use llama_state_set_data instead");

    // Save/load session file
    LLAMA_API bool llama_state_load_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LLAMA_API DEPRECATED(bool llama_load_session_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use llama_state_load_file instead");

    LLAMA_API bool llama_state_save_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count);
    LLAMA_API DEPRECATED(bool llama_save_session_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count),
        "use llama_state_save_file instead");

    // Get the exact size needed to copy the KV cache of a single sequence
    LLAMA_API size_t llama_state_seq_get_size(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Copy the KV cache of a single sequence into the specified buffer
    LLAMA_API size_t llama_state_seq_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                    llama_seq_id   seq_id);

    // Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    LLAMA_API size_t llama_state_seq_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                    llama_seq_id   dest_seq_id);

    LLAMA_API size_t llama_state_seq_save_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count);

    LLAMA_API size_t llama_state_seq_load_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens starting at pos_0
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    LLAMA_API struct llama_batch llama_batch_get_one(
                  llama_token * tokens,
                      int32_t   n_tokens,
                    llama_pos   pos_0,
                 llama_seq_id   seq_id);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with llama_batch_free()
    // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    // The rest of the llama_batch members are allocated with size n_tokens
    // All members are left uninitialized
    LLAMA_API struct llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with llama_batch_init()
    LLAMA_API void llama_batch_free(struct llama_batch batch);

    // Positive return values does not mean a fatal error, but rather a warning.
    //   0 - success
    //   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    // < 0 - error
    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);

    // Set abort callback
    LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    LLAMA_API void llama_synchronize(struct llama_context * ctx);

    // Token logits obtained from the last call to llama_decode()
    // The logits for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which llama_batch.logits[i] != 0
    // Cols: n_vocab
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    // shape: [n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

    //
    // Vocab
    //

    LLAMA_API const char * llama_token_get_text(const struct llama_model * model, llama_token token);

    LLAMA_API float llama_token_get_score(const struct llama_model * model, llama_token token);

    LLAMA_API enum llama_token_type llama_token_get_type(const struct llama_model * model, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_token_bos(const struct llama_model * model); // beginning-of-sentence
    LLAMA_API llama_token llama_token_eos(const struct llama_model * model); // end-of-sentence
    LLAMA_API llama_token llama_token_cls(const struct llama_model * model); // classification
    LLAMA_API llama_token llama_token_sep(const struct llama_model * model); // sentence separator
    LLAMA_API llama_token llama_token_nl (const struct llama_model * model); // next-line

    // Returns -1 if unknown, 1 for true or 0 for false.
    LLAMA_API int32_t         llama_add_bos_token(const struct llama_model * model);

    // Returns -1 if unknown, 1 for true or 0 for false.
    LLAMA_API int32_t         llama_add_eos_token(const struct llama_model * model);

    // codellama infill tokens
    LLAMA_API llama_token llama_token_prefix(const struct llama_model * model); // Beginning of infill prefix
    LLAMA_API llama_token llama_token_middle(const struct llama_model * model); // Beginning of infill middle
    LLAMA_API llama_token llama_token_suffix(const struct llama_model * model); // Beginning of infill suffix
    LLAMA_API llama_token llama_token_eot   (const struct llama_model * model); // End of infill middle

    //
    // Tokenization
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    LLAMA_API int32_t llama_tokenize(
        const struct llama_model * model,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.
    LLAMA_API int32_t llama_token_to_piece(
              const struct llama_model * model,
                           llama_token   token,
                                  char * buf,
                               int32_t   length);

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    /// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
    /// @param chat Pointer to a list of multiple llama_chat_message
    /// @param n_msg Number of llama_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LLAMA_API int32_t llama_chat_apply_template(
              const struct llama_model * model,
                            const char * tmpl,
       const struct llama_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    //
    // Grammar
    //

    LLAMA_API struct llama_grammar * llama_grammar_init(
            const llama_grammar_element ** rules,
                                 size_t    n_rules,
                                 size_t    start_rule_index);

    LLAMA_API void llama_grammar_free(struct llama_grammar * grammar);

    LLAMA_API struct llama_grammar * llama_grammar_copy(const struct llama_grammar * grammar);

    //
    // Sampling functions
    //

    // Sets the current rng seed.
    LLAMA_API void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed);

    /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    LLAMA_API void llama_sample_repetition_penalties(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
               const llama_token * last_tokens,
                          size_t   penalty_last_n,
                           float   penalty_repeat,
                           float   penalty_freq,
                           float   penalty_present);

    /// @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
    /// @param logits Logits extracted from the original generation context.
    /// @param logits_guidance Logits extracted from a separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
    /// @param scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
    LLAMA_API void llama_sample_apply_guidance(
              struct llama_context * ctx,
                             float * logits,
                             float * logits_guidance,
                             float   scale);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    LLAMA_API void llama_sample_softmax(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_k(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                         int32_t   k,
                          size_t   min_keep);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_p(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   p,
                          size_t   min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
    LLAMA_API void llama_sample_min_p(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   p,
                          size_t   min_keep);

    /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    LLAMA_API void llama_sample_tail_free(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   z,
                          size_t   min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API void llama_sample_typical(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   p,
                          size_t   min_keep);

    /// @details Dynamic temperature implementation described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API void llama_sample_entropy(
            struct llama_context * ctx,
          llama_token_data_array * candidates_p,
                           float   min_temp,
                           float   max_temp,
                           float   exponent_val);

    LLAMA_API void llama_sample_temp(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   temp);

    /// @details Apply constraints from grammar
    LLAMA_API void llama_sample_grammar(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
      const struct llama_grammar * grammar);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API llama_token llama_sample_token_mirostat(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   tau,
                           float   eta,
                         int32_t   m,
                           float * mu);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API llama_token llama_sample_token_mirostat_v2(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   tau,
                           float   eta,
                           float * mu);

    /// @details Selects the token with the highest probability.
    ///          Does not compute the token probabilities. Use llama_sample_softmax() instead.
    LLAMA_API llama_token llama_sample_token_greedy(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    /// @details Randomly selects a token from the candidates based on their probabilities.
    LLAMA_API llama_token llama_sample_token(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    /// @details Accepts the sampled token into the grammar
    LLAMA_API void llama_grammar_accept_token(
            struct llama_context * ctx,
            struct llama_grammar * grammar,
                     llama_token   token);

    //
    // Beam search
    //

    struct llama_beam_view {
        const llama_token * tokens;

        size_t n_tokens;
        float  p;        // Cumulative beam probability (renormalized relative to all beams)
        bool   eob;      // Callback should set this to true when a beam is at end-of-beam.
    };

    // Passed to beam_search_callback function.
    // Whenever 0 < common_prefix_length, this number of tokens should be copied from any of the beams
    // (e.g. beams[0]) as they will be removed (shifted) from all beams in all subsequent callbacks.
    // These pointers are valid only during the synchronous callback, so should not be saved.
    struct llama_beams_state {
        struct llama_beam_view * beam_views;

        size_t n_beams;               // Number of elements in beam_views[].
        size_t common_prefix_length;  // Current max length of prefix tokens shared by all beams.
        bool   last_call;             // True iff this is the last callback invocation.
    };

    // Type of pointer to the beam_search_callback function.
    // void* callback_data is any custom data passed to llama_beam_search, that is subsequently
    // passed back to beam_search_callback. This avoids having to use global variables in the callback.
    typedef void (*llama_beam_search_callback_fn_t)(void * callback_data, struct llama_beams_state);

    /// @details Deterministically returns entire sentence constructed by a beam search.
    /// @param ctx Pointer to the llama_context.
    /// @param callback Invoked for each iteration of the beam_search loop, passing in beams_state.
    /// @param callback_data A pointer that is simply passed back to callback.
    /// @param n_beams Number of beams to use.
    /// @param n_past Number of tokens already evaluated.
    /// @param n_predict Maximum number of tokens to predict. EOS may occur earlier.
    LLAMA_API void llama_beam_search(
                   struct llama_context * ctx,
        llama_beam_search_callback_fn_t   callback,
                                   void * callback_data,
                                 size_t   n_beams,
                                int32_t   n_past,
                                int32_t   n_predict);

    /// @details Build a split GGUF final path for this chunk.
    ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LLAMA_API int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LLAMA_API int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);

    // Performance information
    LLAMA_API struct llama_timings llama_get_timings(struct llama_context * ctx);

    LLAMA_API void llama_print_timings(struct llama_context * ctx);
    LLAMA_API void llama_reset_timings(struct llama_context * ctx);

    // Print system information
    LLAMA_API const char * llama_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);

    LLAMA_API void llama_dump_timing_info_yaml(FILE * stream, const struct llama_context * ctx);

#ifdef __cplusplus
}
#endif

// Internal API to be implemented by llama.cpp and used by tests/benchmarks only
#ifdef LLAMA_API_INTERNAL

#include <vector>
#include <string>

struct ggml_tensor;

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar {
    const std::vector<std::vector<llama_grammar_element>>   rules;
    std::vector<std::vector<const llama_grammar_element *>> stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8                                      partial_utf8;
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
};

const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
);

void llama_grammar_accept(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const uint32_t                                                  chr,
        std::vector<std::vector<const llama_grammar_element *>>       & new_stacks);

std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
        const std::string & src,
        llama_partial_utf8   partial_start);

#endif // LLAMA_API_INTERNAL

#endif // LLAMA_H
