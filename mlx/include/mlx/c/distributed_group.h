/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_DISTRIBUTED_GROUP_H
#define MLX_DISTRIBUTED_GROUP_H

#include <stdbool.h>
#include <stddef.h>

#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_distributed_group MLX distributed
 */
/**@{*/

/**
 * A MLX distributed group object.
 */
typedef struct mlx_distributed_group_ {
  void* ctx;
} mlx_distributed_group;

/**
 * Create an empty group.
 */
mlx_distributed_group mlx_distributed_group_new(void);

/**
 * Free the group.
 */
int mlx_distributed_group_free(mlx_distributed_group group);

/**
 * Initialize distributed.
 */
int mlx_distributed_init(
    mlx_distributed_group* res,
    bool strict,
    const char* bk /* may be null */);

/**
 * Byte-level all-gather callback used by the JACCL side channel.
 *
 * \param rank The rank of the process.
 * \param size The size of the group.
 * \param src Pointer to this rank's data of size n_bytes.
 * \param dst Pointer to an output buffer of size size * n_bytes. After the
 * call, dst[rank * n_bytes, (rank+1) * n_bytes] contains the data from rank.
 * \param n_bytes The number of bytes contributed by each rank.
 * \param payload Opaque user data passed to the callback.
 * \return Zero on success, non-zero on failure.
 */
typedef int (*mlx_distributed_all_gather_fn)(
    int rank,
    int size,
    const char* src,
    char* dst,
    size_t n_bytes,
    void* payload);

/**
 * Initialize JACCL with a custom metadata all-gather callback.
 *
 * `bk` must be `jaccl`. The synchronous callback reads `n_bytes` from `src`
 * and fills `size * n_bytes` in `dst`, ordered by `rank`. Buffers are borrowed.
 * The callback returns zero on success, nonzero on failure.
 *
 * If a `jaccl` group is already cached, it is returned and `all_gather` is
 * never called: the callback is dropped and, with `dtor`, `payload` is
 * destroyed before this function returns. Call
 * `mlx_distributed_clear_backends()` first to initialize with a new callback.
 *
 * With `dtor`, ownership of `payload` transfers on entry, including failure.
 * Destruction occurs once after all references are released. Cached or
 * live groups may delay it. Without `dtor`, `payload` is borrowed and must
 * remain valid while the callback is retained.
 *
 * Returns zero on success, nonzero on error. `res` is unchanged on failure.
 */
int mlx_distributed_init_with_all_gather(
    mlx_distributed_group* res,
    bool strict,
    const char* bk,
    mlx_distributed_all_gather_fn all_gather,
    void* payload,
    void (*dtor)(void*));

/**
 * Clear the distributed backend cache.
 */
int mlx_distributed_clear_backends(void);

/**
 * Get the rank.
 */
int mlx_distributed_group_get_rank(int* res, mlx_distributed_group group);

/**
 * Get the group size.
 */
int mlx_distributed_group_get_size(int* res, mlx_distributed_group group);

/**
 * Split the group.
 */
int mlx_distributed_group_split(
    mlx_distributed_group* res,
    mlx_distributed_group group,
    int color,
    int key);

/**
 * Check if distributed is available.
 */
bool mlx_distributed_is_available(const char* bk /* may be null */);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
