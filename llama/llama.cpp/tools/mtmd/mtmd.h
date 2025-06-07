#ifndef MTMD_H
#define MTMD_H

#include "ggml.h"
#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <string>
#include <vector>
#include <cinttypes>
#include <memory>
#endif

/**
 * libmtmd: A library for multimodal support in llama.cpp.
 *
 * WARNING: This API is experimental and subject to many BREAKING CHANGES.
 *          Issues related to API usage may receive lower priority support.
 *
 * For the usage, see an example in mtmd-cli.cpp
 */

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MTMD_API __declspec(dllexport)
#        else
#            define MTMD_API __declspec(dllimport)
#        endif
#    else
#        define MTMD_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MTMD_API
#endif

// deprecated marker, use mtmd_default_marker() instead
#define MTMD_DEFAULT_IMAGE_MARKER "<__image__>"

#ifdef __cplusplus
extern "C" {
#endif

enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
    MTMD_INPUT_CHUNK_TYPE_AUDIO,
};

// opaque types
struct mtmd_context;
struct mtmd_bitmap;
struct mtmd_image_tokens;
struct mtmd_input_chunk;
struct mtmd_input_chunks;

struct mtmd_input_text {
    const char * text;
    bool add_special;
    bool parse_special;
};

//
// C API
//

typedef struct mtmd_context      mtmd_context;
typedef struct mtmd_bitmap       mtmd_bitmap;
typedef struct mtmd_image_tokens mtmd_image_tokens;
typedef struct mtmd_input_chunk  mtmd_input_chunk;
typedef struct mtmd_input_chunks mtmd_input_chunks;
typedef struct mtmd_input_text   mtmd_input_text;

struct mtmd_context_params {
    bool use_gpu;
    bool print_timings;
    int n_threads;
    enum ggml_log_level verbosity;
    const char * image_marker; // deprecated, use media_marker instead
    const char * media_marker;
};

MTMD_API const char * mtmd_default_marker(void);

MTMD_API struct mtmd_context_params mtmd_context_params_default(void);

// initialize the mtmd context
// return nullptr on failure
MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
                                            const struct llama_model * text_model,
                                            const struct mtmd_context_params ctx_params);

MTMD_API void mtmd_free(mtmd_context * ctx);

// whether we need to set non-causal mask before llama_decode
MTMD_API bool mtmd_decode_use_non_causal(mtmd_context * ctx);

// whether the current model use M-RoPE for llama_decode
MTMD_API bool mtmd_decode_use_mrope(mtmd_context * ctx);

// whether the current model supports vision input
MTMD_API bool mtmd_support_vision(mtmd_context * ctx);

// whether the current model supports audio input
MTMD_API bool mtmd_support_audio(mtmd_context * ctx);

// get audio bitrate in Hz, for example 16000 for Whisper
// return -1 if audio is not supported
MTMD_API int mtmd_get_audio_bitrate(mtmd_context * ctx);

// mtmd_bitmap
//
// if bitmap is image:
//     length of data must be nx * ny * 3
//     the data is in RGBRGBRGB... format
// if bitmap is audio:
//     length of data must be n_samples * sizeof(float)
//     the data is in float format (PCM F32)
MTMD_API mtmd_bitmap *         mtmd_bitmap_init           (uint32_t nx, uint32_t ny, const unsigned char * data);
MTMD_API mtmd_bitmap *         mtmd_bitmap_init_from_audio(size_t n_samples,         const float         * data);
MTMD_API uint32_t              mtmd_bitmap_get_nx     (const mtmd_bitmap * bitmap);
MTMD_API uint32_t              mtmd_bitmap_get_ny     (const mtmd_bitmap * bitmap);
MTMD_API const unsigned char * mtmd_bitmap_get_data   (const mtmd_bitmap * bitmap);
MTMD_API size_t                mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap);
MTMD_API bool                  mtmd_bitmap_is_audio   (const mtmd_bitmap * bitmap);
MTMD_API void                  mtmd_bitmap_free       (mtmd_bitmap * bitmap);
// bitmap ID is optional, but useful for KV cache tracking
// these getters/setters are dedicated functions, so you can for example calculate the hash of the image based on mtmd_bitmap_get_data()
MTMD_API const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap);
MTMD_API void         mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id);


// mtmd_input_chunks
//
// this is simply a list of mtmd_input_chunk
// the elements can only be populated via mtmd_tokenize()
MTMD_API mtmd_input_chunks *      mtmd_input_chunks_init(void);
MTMD_API size_t                   mtmd_input_chunks_size(const mtmd_input_chunks * chunks);
MTMD_API const mtmd_input_chunk * mtmd_input_chunks_get (const mtmd_input_chunks * chunks, size_t idx);
MTMD_API void                     mtmd_input_chunks_free(mtmd_input_chunks * chunks);

// mtmd_input_chunk
//
// the instance will be constructed via mtmd_tokenize()
// it will be freed along with mtmd_input_chunks
MTMD_API enum mtmd_input_chunk_type mtmd_input_chunk_get_type        (const mtmd_input_chunk * chunk);
MTMD_API const llama_token *        mtmd_input_chunk_get_tokens_text (const mtmd_input_chunk * chunk, size_t * n_tokens_output);
MTMD_API const mtmd_image_tokens *  mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk);
MTMD_API size_t                     mtmd_input_chunk_get_n_tokens    (const mtmd_input_chunk * chunk);
// returns nullptr for ID on text chunk
MTMD_API const char *               mtmd_input_chunk_get_id          (const mtmd_input_chunk * chunk);
// number of temporal positions (always 1 for M-RoPE, n_tokens otherwise)
MTMD_API llama_pos                  mtmd_input_chunk_get_n_pos       (const mtmd_input_chunk * chunk);

// in case you want to use custom logic to handle the chunk (i.e. KV cache management)
// you can move the chunk ownership to your own code by copying it
// remember to free the chunk when you are done with it
MTMD_API mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk);
MTMD_API void               mtmd_input_chunk_free(mtmd_input_chunk * chunk);


// mtmd_image_tokens
//
// the instance will be constructed via mtmd_tokenize()
// it will be freed along with mtmd_input_chunk
MTMD_API size_t       mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens); // TODO: deprecate
MTMD_API size_t       mtmd_image_tokens_get_nx      (const mtmd_image_tokens * image_tokens);
MTMD_API size_t       mtmd_image_tokens_get_ny      (const mtmd_image_tokens * image_tokens);
MTMD_API const char * mtmd_image_tokens_get_id      (const mtmd_image_tokens * image_tokens); // TODO: deprecate
// number of temporal positions (always 1 for M-RoPE, n_tokens otherwise)
MTMD_API llama_pos    mtmd_image_tokens_get_n_pos   (const mtmd_image_tokens * image_tokens); // TODO: deprecate

// tokenize an input text prompt and a list of bitmaps (images/audio)
// the prompt must have the input image marker (default: "<__media__>") in it
// the default marker is defined by mtmd_default_marker()
// the marker will be replaced with the image/audio chunk
// for example:
//   "here is an image: <__media__>\ndescribe it in detail."
//   this will gives 3 chunks:
//   1. "here is an image: <start_of_image>"
//   2. (image/audio tokens)
//   3. "<end_of_image>\ndescribe it in detail."
// number of bitmaps must be equal to the number of markers in the prompt
// this function is thread-safe (shared ctx)
// return values:
//   0 on success
//   1 on number of bitmaps not matching the number of markers
//   2 on image preprocessing error
MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
                               mtmd_input_chunks * output,
                               const mtmd_input_text * text,
                               const mtmd_bitmap ** bitmaps,
                               size_t n_bitmaps);

// returns 0 on success
// TODO: deprecate
MTMD_API int32_t mtmd_encode(mtmd_context * ctx,
                             const mtmd_image_tokens * image_tokens);

// returns 0 on success
MTMD_API int32_t mtmd_encode_chunk(mtmd_context * ctx,
                                   const mtmd_input_chunk * chunk);

// get output embeddings from the last encode pass
// the reading size (in bytes) is equal to:
// llama_model_n_embd(model) * mtmd_input_chunk_get_n_tokens(chunk) * sizeof(float)
MTMD_API float * mtmd_get_output_embd(mtmd_context * ctx);

/////////////////////////////////////////

// test function, to be used in test-mtmd-c-api.c
MTMD_API mtmd_input_chunks * mtmd_test_create_input_chunks(void);

#ifdef __cplusplus
} // extern "C"
#endif

//
// C++ wrappers
//

#ifdef __cplusplus

namespace mtmd {

struct mtmd_context_deleter {
    void operator()(mtmd_context * val) { mtmd_free(val); }
};
using context_ptr = std::unique_ptr<mtmd_context, mtmd_context_deleter>;

struct mtmd_bitmap_deleter {
    void operator()(mtmd_bitmap * val) { mtmd_bitmap_free(val); }
};
using bitmap_ptr = std::unique_ptr<mtmd_bitmap, mtmd_bitmap_deleter>;

struct mtmd_input_chunks_deleter {
    void operator()(mtmd_input_chunks * val) { mtmd_input_chunks_free(val); }
};
using input_chunks_ptr = std::unique_ptr<mtmd_input_chunks, mtmd_input_chunks_deleter>;

struct mtmd_input_chunk_deleter {
    void operator()(mtmd_input_chunk * val) { mtmd_input_chunk_free(val); }
};
using input_chunk_ptr = std::unique_ptr<mtmd_input_chunk, mtmd_input_chunk_deleter>;

struct bitmap {
    bitmap_ptr ptr;
    bitmap() : ptr(nullptr) {}
    bitmap(mtmd_bitmap * bitmap) : ptr(bitmap) {}
    bitmap(bitmap && other) noexcept : ptr(std::move(other.ptr)) {}
    bitmap(uint32_t nx, uint32_t ny, const unsigned char * data) {
        ptr.reset(mtmd_bitmap_init(nx, ny, data));
    }
    ~bitmap() = default;
    uint32_t nx() { return mtmd_bitmap_get_nx(ptr.get()); }
    uint32_t ny() { return mtmd_bitmap_get_ny(ptr.get()); }
    const unsigned char * data() { return mtmd_bitmap_get_data(ptr.get()); }
    size_t n_bytes() { return mtmd_bitmap_get_n_bytes(ptr.get()); }
    std::string id() { return mtmd_bitmap_get_id(ptr.get()); }
    void set_id(const char * id) { mtmd_bitmap_set_id(ptr.get(), id); }
};

struct bitmaps {
    std::vector<bitmap> entries;
    ~bitmaps() = default;
    // return list of pointers to mtmd_bitmap
    // example:
    //   auto bitmaps_c_ptr = bitmaps.c_ptr();
    //   int32_t res = mtmd_tokenize(... bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    std::vector<const mtmd_bitmap *> c_ptr() {
        std::vector<const mtmd_bitmap *> res(entries.size());
        for (size_t i = 0; i < entries.size(); i++) {
            res[i] = entries[i].ptr.get();
        }
        return res;
    }
};

struct input_chunks {
    input_chunks_ptr ptr;
    input_chunks() = default;
    input_chunks(mtmd_input_chunks * chunks) : ptr(chunks) {}
    ~input_chunks() = default;
    size_t size() { return mtmd_input_chunks_size(ptr.get()); }
    const mtmd_input_chunk * operator[](size_t idx) {
        return mtmd_input_chunks_get(ptr.get(), idx);
    }
};

} // namespace mtmd

#endif

#endif
