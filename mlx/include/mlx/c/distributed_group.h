/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_DISTRIBUTED_GROUP_H
#define MLX_DISTRIBUTED_GROUP_H

#include <stdbool.h>

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
 * Get the rank.
 */
int mlx_distributed_group_rank(mlx_distributed_group group);

/**
 * Get the group size.
 */
int mlx_distributed_group_size(mlx_distributed_group group);

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
