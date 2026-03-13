/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_LINALG_H
#define MLX_LINALG_H

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
 * \defgroup linalg Linear algebra operations
 */
/**@{*/

int mlx_linalg_cholesky(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s);
int mlx_linalg_cholesky_inv(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s);
int mlx_linalg_cross(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s);
int mlx_linalg_eig(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s);
int mlx_linalg_eigh(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s);
int mlx_linalg_eigvals(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_linalg_eigvalsh(
    mlx_array* res,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s);
int mlx_linalg_inv(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_linalg_lu(mlx_vector_array* res, const mlx_array a, const mlx_stream s);
int mlx_linalg_lu_factor(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s);
int mlx_linalg_norm(
    mlx_array* res,
    const mlx_array a,
    double ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s);
int mlx_linalg_norm_matrix(
    mlx_array* res,
    const mlx_array a,
    const char* ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s);
int mlx_linalg_norm_l2(
    mlx_array* res,
    const mlx_array a,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s);
int mlx_linalg_pinv(mlx_array* res, const mlx_array a, const mlx_stream s);
int mlx_linalg_qr(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s);
int mlx_linalg_solve(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s);
int mlx_linalg_solve_triangular(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool upper,
    const mlx_stream s);
int mlx_linalg_svd(
    mlx_vector_array* res,
    const mlx_array a,
    bool compute_uv,
    const mlx_stream s);
int mlx_linalg_tri_inv(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
