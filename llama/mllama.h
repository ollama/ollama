#ifndef MLLAMA_H
#define MLLAMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef LLAMA_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef LLAMA_BUILD
#define MLLAMA_API __declspec(dllexport)
#else
#define MLLAMA_API __declspec(dllimport)
#endif
#else
#define MLLAMA_API __attribute__((visibility("default")))
#endif
#else
#define MLLAMA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct mllama_ctx;

struct mllama_image_batch {
    struct mllama_image *data;
    size_t size;
};

MLLAMA_API struct mllama_ctx *mllama_model_load(const char *fname, int verbosity);
MLLAMA_API struct mllama_ctx *mllama_model_load_cpu(const char *fname, int verbosity);

MLLAMA_API void mllama_free(struct mllama_ctx *ctx);

MLLAMA_API int32_t mllama_image_size(const struct mllama_ctx *ctx);
MLLAMA_API int32_t mllama_patch_size(const struct mllama_ctx *ctx);
MLLAMA_API int32_t mllama_hidden_size(const struct mllama_ctx *ctx);

MLLAMA_API int mllama_n_patches(const struct mllama_ctx *ctx);
MLLAMA_API int mllama_n_positions(const struct mllama_ctx *ctx);
MLLAMA_API int mllama_n_tiles(const struct mllama_ctx *ctx);
MLLAMA_API int mllama_n_embd(const struct mllama_ctx *ctx);
MLLAMA_API size_t mllama_n_embd_bytes(const struct mllama_ctx *ctx);

MLLAMA_API struct mllama_image *mllama_image_init();

MLLAMA_API void mllama_image_free(struct mllama_image *img);
MLLAMA_API void mllama_image_batch_free(struct mllama_image_batch *batch);

MLLAMA_API bool mllama_image_load_from_data(const void *data, const int n, const int nx, const int ny, const int nc, const int nt, const int aspect_ratio_id, struct mllama_image *img);

MLLAMA_API bool mllama_image_encode(struct mllama_ctx *ctx, int n_threads, struct mllama_image *img, float *vec);
MLLAMA_API bool mllama_image_batch_encode(struct mllama_ctx *ctx, int n_threads, const struct mllama_image_batch *imgs, float *vec);

#ifdef __cplusplus
}
#endif

#endif // MLLAMA_H
