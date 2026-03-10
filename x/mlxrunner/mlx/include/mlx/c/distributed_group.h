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
mlx_distributed_group
mlx_distributed_group_split(mlx_distributed_group group, int color, int key);

/**
 * Check if distributed is available.
 */
bool mlx_distributed_is_available(void);

/**
 * Initialize distributed.
 */
mlx_distributed_group mlx_distributed_init(bool strict);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
