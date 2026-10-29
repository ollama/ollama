/* Copyright © 2023-2025 Apple Inc.                   */

#ifndef MLX_EXPORT_H
#define MLX_EXPORT_H

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
 * \defgroup export Function serialization
 */
/**@{*/
int mlx_export_function(
    const char* file,
    const mlx_closure fun,
    const mlx_vector_array args,
    bool shapeless);
int mlx_export_function_kwargs(
    const char* file,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless);

typedef struct mlx_function_exporter_ {
  void* ctx;
} mlx_function_exporter;
mlx_function_exporter mlx_function_exporter_new(
    const char* file,
    const mlx_closure fun,
    bool shapeless);
int mlx_function_exporter_free(mlx_function_exporter xfunc);
int mlx_function_exporter_apply(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args);
int mlx_function_exporter_apply_kwargs(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs);

typedef struct mlx_imported_function_ {
  void* ctx;
} mlx_imported_function;
mlx_imported_function mlx_imported_function_new(const char* file);
int mlx_imported_function_free(mlx_imported_function xfunc);
int mlx_imported_function_apply(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args);
int mlx_imported_function_apply_kwargs(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif
