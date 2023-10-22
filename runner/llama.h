/**
 * llama.cpp - git 465219b9143ac01db0990bbcb0a081ef72ec2008
 *
 * MIT License
 *
 * Copyright (c) 2023 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#define LLAMA_MAX_DEVICES GGML_CUDA_MAX_DEVICES
#else
#define LLAMA_MAX_DEVICES 1
#endif // GGML_USE_CUBLAS
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

#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'

#define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 2

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST) || defined(GGML_USE_METAL)
// Defined when llama.cpp is compiled with support for offloading model layers to GPU.
#define LLAMA_SUPPORTS_GPU_OFFLOAD
#endif

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
        LLAMA_VOCAB_TYPE_SPM = 0, // SentencePiece
        LLAMA_VOCAB_TYPE_BPE = 1, // Byte Pair Encoding
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

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
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

    typedef void (*llama_progress_callback)(float progress, void *ctx);

    // Input data for llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    // - seq_id : the sequence to which the respective token belongs
    // - logits : if zero, the logits for the respective token will not be output
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;

        // NOTE: helpers for smooth API transition - can be deprecated in the future
        //       for future-proof code, use the above fields instead and ignore everything below
        //
        // pos[i] = all_pos_0 + i*all_pos_1
        //
        llama_pos    all_pos_0;  // used if pos == NULL
        llama_pos    all_pos_1;  // used if pos == NULL
        llama_seq_id all_seq_id; // used if seq_id == NULL
    } llama_batch;

    struct llama_model_params {
        int32_t n_gpu_layers; // number of layers to store in VRAM
        int32_t main_gpu;     // the GPU that is used for scratch and small tensors
        const float * tensor_split; // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)

        // called with a progress value between 0 and 1, pass NULL to disable
        llama_progress_callback progress_callback;
        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only; // only load the vocabulary, no weights
        bool use_mmap;   // use mmap if possible
        bool use_mlock;  // force system to keep model in RAM
    };

    struct llama_context_params {
        uint32_t seed;            // RNG seed, -1 for random
        uint32_t n_ctx;           // text context, 0 = from model
        uint32_t n_batch;         // prompt processing maximum batch size
        uint32_t n_threads;       // number of threads to use for generation
        uint32_t n_threads_batch; // number of threads to use for batch processing

        // ref: https://github.com/ggerganov/llama.cpp/pull/2054
        float rope_freq_base;  // RoPE base frequency, 0 = from model
        float rope_freq_scale; // RoPE frequency scaling factor, 0 = from model

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool mul_mat_q;  // if true, use experimental mul_mat_q kernels
        bool f16_kv;     // use fp16 for KV cache, fp32 otherwise
        bool logits_all; // the llama_eval() call computes all logits, not just the last one
        bool embedding;  // embedding mode only
    };

    // model quantization parameters
    typedef struct llama_model_quantize_params {
        int nthread;                 // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llama_ftype ftype;      // quantize to this llama_ftype
        bool allow_requantize;       // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor; // quantize output.weight
        bool only_copy;              // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
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

    // Helpers for getting default parameters
    LLAMA_API struct llama_model_params llama_model_default_params(void);
    LLAMA_API struct llama_context_params llama_context_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);

    // Initialize the llama + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    LLAMA_API void llama_backend_init(bool numa);

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

    LLAMA_API int  llama_max_devices    (void);
    LLAMA_API bool llama_mmap_supported (void);
    LLAMA_API bool llama_mlock_supported(void);

    LLAMA_API const struct llama_model * llama_get_model(const struct llama_context * ctx);

    LLAMA_API int llama_n_ctx      (const struct llama_context * ctx);

    LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_model * model);

    LLAMA_API int llama_n_vocab    (const struct llama_model * model);
    LLAMA_API int llama_n_ctx_train(const struct llama_model * model);
    LLAMA_API int llama_n_embd     (const struct llama_model * model);

    // Get the model's RoPE frequency scaling factor
    LLAMA_API float llama_rope_freq_scale_train(const struct llama_model * model);

    // Get a string describing the model type
    LLAMA_API int llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    LLAMA_API uint64_t llama_model_size(const struct llama_model * model);

    // Returns the total number of parameters in the model
    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);

    // Get a llama model tensor
    LLAMA_API struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name);

    // Returns 0 on success
    LLAMA_API int llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params);

    // Apply a LoRA adapter to a loaded model
    // path_base_model is the path to a higher quality model to use as a base for
    // the layers modified by the adapter. Can be NULL to use the current loaded model.
    // The model needs to be reloaded before applying a new adapter, otherwise the adapter
    // will be applied on top of the previous one
    // Returns 0 on success
    LLAMA_API DEPRECATED(int llama_apply_lora_from_file(
            struct llama_context * ctx,
                      const char * path_lora,
                           float   scale,
                      const char * path_base_model,
                             int   n_threads),
            "use llama_model_apply_lora_from_file instead");

    LLAMA_API int llama_model_apply_lora_from_file(
            const struct llama_model * model,
                      const char * path_lora,
                           float   scale,
                      const char * path_base_model,
                             int   n_threads);

    //
    // KV cache
    //

    // Returns the number of tokens in the KV cache
    LLAMA_API DEPRECATED(int llama_get_kv_cache_token_count(const struct llama_context * ctx),
            "avoid using this, it will be removed in the future, instead - count the tokens in user code");

    // Remove all tokens data of cells in [c0, c1)
    // c0 < 0 : [0,  c1]
    // c1 < 0 : [c0, inf)
    LLAMA_API void llama_kv_cache_tokens_rm(
            struct llama_context * ctx,
                         int32_t   c0,
                         int32_t   c1);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_rm(
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
    // If the KV cache is RoPEd, the KV data is updated accordingly
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_kv_cache_seq_shift(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                       llama_pos   delta);

    //
    // State / sessions
    //

    // Returns the maximum size in bytes of the state (rng, logits, embedding
    // and kv_cache) - will often be smaller after compacting tokens
    LLAMA_API size_t llama_get_state_size(const struct llama_context * ctx);

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLAMA_API size_t llama_copy_state_data(
            struct llama_context * ctx,
                         uint8_t * dst);

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLAMA_API size_t llama_set_state_data(
            struct llama_context * ctx,
                         uint8_t * src);

    // Save/load session file
    LLAMA_API bool llama_load_session_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

    LLAMA_API bool llama_save_session_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count);

    //
    // Decoding
    //

    // Run the llama inference to obtain the logits and probabilities for the next token(s).
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    // DEPRECATED: use llama_decode() instead
    LLAMA_API DEPRECATED(int llama_eval(
            struct llama_context * ctx,
                     llama_token * tokens,
                         int32_t   n_tokens,
                             int   n_past),
            "use llama_decode() instead");

    // Same as llama_eval, but use float matrix input directly.
    // DEPRECATED: use llama_decode() instead
    LLAMA_API DEPRECATED(int llama_eval_embd(
            struct llama_context * ctx,
                           float * embd,
                         int32_t   n_tokens,
                             int   n_past),
            "use llama_decode() instead");

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
    LLAMA_API int llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch);

    // Token logits obtained from the last call to llama_eval()
    // The logits for the last token are stored in the last row
    // Logits for which llama_batch.logits[i] == 0 are undefined
    // Rows: n_tokens provided with llama_batch
    // Cols: n_vocab
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Logits for the ith token. Equivalent to:
    // llama_get_logits(ctx) + i*n_vocab
    LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    //
    // Vocab
    //

    LLAMA_API const char * llama_token_get_text(const struct llama_context * ctx, llama_token token);

    LLAMA_API float llama_token_get_score(const struct llama_context * ctx, llama_token token);

    LLAMA_API enum llama_token_type llama_token_get_type(const struct llama_context * ctx, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_token_bos(const struct llama_context * ctx);  // beginning-of-sentence
    LLAMA_API llama_token llama_token_eos(const struct llama_context * ctx);  // end-of-sentence
    LLAMA_API llama_token llama_token_nl (const struct llama_context * ctx);  // next-line
    // codellama infill tokens
    LLAMA_API llama_token llama_token_prefix(const struct llama_context * ctx); // Beginning of infill prefix
    LLAMA_API llama_token llama_token_middle(const struct llama_context * ctx); // Beginning of infill middle
    LLAMA_API llama_token llama_token_suffix(const struct llama_context * ctx); // Beginning of infill suffix
    LLAMA_API llama_token llama_token_eot   (const struct llama_context * ctx); // End of infill middle

    //
    // Tokenization
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_max_tokens
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @param special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext.
    ///                Does not insert a leading space.
    LLAMA_API int llama_tokenize(
        const struct llama_model * model,
                      const char * text,
                             int   text_len,
                     llama_token * tokens,
                             int   n_max_tokens,
                            bool   add_bos,
                            bool   special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.
    LLAMA_API int llama_token_to_piece(
              const struct llama_model * model,
                           llama_token   token,
                                  char * buf,
                                  int    length);

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
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
    /// @params guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
    /// @params scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
    LLAMA_API void llama_sample_classifier_free_guidance(
              struct llama_context * ctx,
            llama_token_data_array * candidates,
              struct llama_context * guidance_ctx,
                             float   scale);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    LLAMA_API void llama_sample_softmax(
            struct llama_context * ctx,
          llama_token_data_array * candidates);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_k(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                             int   k,
                          size_t   min_keep);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API void llama_sample_top_p(
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

    LLAMA_API void llama_sample_temp(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
                           float   temp);

    LLAMA_API DEPRECATED(void llama_sample_temperature(
                struct llama_context * ctx,
              llama_token_data_array * candidates,
                               float   temp),
            "use llama_sample_temp instead");

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
                             int   m,
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
                                    int   n_past,
                                    int   n_predict);

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

const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
);

#endif // LLAMA_API_INTERNAL

#endif // LLAMA_H
