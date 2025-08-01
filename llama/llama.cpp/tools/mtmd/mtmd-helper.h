#ifndef MTMD_HELPER_H
#define MTMD_HELPER_H

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// libmtmd helper functions
//
// Please note that these helpers are not guaranteed to be stable.
// BREAKING CHANGES are expected.
//

// helper function to construct a mtmd_bitmap from a file
// it calls mtmd_helper_bitmap_init_from_buf() internally
// returns nullptr on failure
// this function is thread-safe
MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname);

// helper function to construct a mtmd_bitmap from a buffer containing a file
// supported formats:
//     image: formats supported by stb_image: jpg, png, bmp, gif, etc.
//     audio: formats supported by miniaudio: wav, mp3, flac
// note: audio files will be auto-detected based on magic bytes
// returns nullptr on failure
// this function is thread-safe
MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len);

// helper to count the total number of tokens from a list of chunks, useful to keep track of KV cache
MTMD_API size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks);

// helper to count the total position of tokens from a list of chunks, useful to keep track of n_past
// normally, n_pos is equal to n_tokens, but for M-RoPE it is different
MTMD_API llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks);

// helper function that automatically:
// 1. run llama_decode() on text chunks
// 2. run mtmd_encode() on image chunks, then mtmd_get_output_embd() and then llama_decode()
// if any of the mtmd_encode() or llama_decode() calls return non-zero, stop and forward the error
// otherwise, returns 0 on success
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                         struct llama_context * lctx,
                                         const mtmd_input_chunks * chunks,
                                         llama_pos n_past,
                                         llama_seq_id seq_id,
                                         int32_t n_batch,
                                         bool logits_last,
                                         llama_pos * new_n_past);

// works like mtmd_helper_eval_chunks(), but only for a single chunk
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
                                               struct llama_context * lctx,
                                               const mtmd_input_chunk * chunk,
                                               llama_pos n_past,
                                               llama_seq_id seq_id,
                                               int32_t n_batch,
                                               bool logits_last,
                                               llama_pos * new_n_past);

// helper function to decode an image whose embeddings have already been calculated
// this helper will handle batching and pre/post decoding setup (for ex. gemma 3 requires non-causal attention)
// ret 0 on success, -1 on chunk not being a valid image chunk, 1 on decode failure
MTMD_API int32_t mtmd_helper_decode_image_chunk(mtmd_context * ctx,
                                                struct llama_context * lctx,
                                                const mtmd_input_chunk * chunk,
                                                float * encoded_embd,
                                                llama_pos n_past,
                                                llama_seq_id seq_id,
                                                int32_t n_batch,
                                                llama_pos * new_n_past);

#ifdef __cplusplus
} // extern "C"
#endif

//
// C++ wrappers
//

#endif
