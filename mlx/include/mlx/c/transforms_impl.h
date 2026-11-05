/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_TRANSFORMS_IMPL_H
#define MLX_TRANSFORMS_IMPL_H

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
 * \defgroup transforms_impl Implementation detail operations
 */
/**@{*/

int mlx_detail_vmap_replace(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_vector_array s_inputs,
    const mlx_vector_array s_outputs,
    const int* in_axes,
    size_t in_axes_num,
    const int* out_axes,
    size_t out_axes_num);
int mlx_detail_vmap_trace(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array inputs,
    const int* in_axes,
    size_t in_axes_num);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
