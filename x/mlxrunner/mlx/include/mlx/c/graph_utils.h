/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_GRAPH_UTILS_H
#define MLX_GRAPH_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup graph_utils Graph Utils
 */
/**@{*/

typedef struct mlx_node_namer_ {
  void* ctx;
} mlx_node_namer;

mlx_node_namer mlx_node_namer_new();
int mlx_node_namer_free(mlx_node_namer namer);
int mlx_node_namer_set_name(
    mlx_node_namer namer,
    const mlx_array arr,
    const char* name);
int mlx_node_namer_get_name(
    const char** name,
    mlx_node_namer namer,
    const mlx_array arr);

int mlx_export_to_dot(
    FILE* os,
    const mlx_node_namer namer,
    const mlx_vector_array outputs);
int mlx_print_graph(
    FILE* os,
    const mlx_node_namer namer,
    const mlx_vector_array outputs);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
