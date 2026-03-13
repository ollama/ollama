/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_VECTOR_H
#define MLX_VECTOR_H

#include "mlx/c/array.h"
#include "mlx/c/string.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_vector Vectors
 * MLX vector objects.
 */
/**@{*/

/**
 * A vector of array.
 */
typedef struct mlx_vector_array_ {
  void* ctx;
} mlx_vector_array;
mlx_vector_array mlx_vector_array_new(void);
int mlx_vector_array_set(mlx_vector_array* vec, const mlx_vector_array src);
int mlx_vector_array_free(mlx_vector_array vec);
mlx_vector_array mlx_vector_array_new_data(const mlx_array* data, size_t size);
mlx_vector_array mlx_vector_array_new_value(const mlx_array val);
int mlx_vector_array_set_data(
    mlx_vector_array* vec,
    const mlx_array* data,
    size_t size);
int mlx_vector_array_set_value(mlx_vector_array* vec, const mlx_array val);
int mlx_vector_array_append_data(
    mlx_vector_array vec,
    const mlx_array* data,
    size_t size);
int mlx_vector_array_append_value(mlx_vector_array vec, const mlx_array val);
size_t mlx_vector_array_size(mlx_vector_array vec);
int mlx_vector_array_get(
    mlx_array* res,
    const mlx_vector_array vec,
    size_t idx);

/**
 * A vector of vector_array.
 */
typedef struct mlx_vector_vector_array_ {
  void* ctx;
} mlx_vector_vector_array;
mlx_vector_vector_array mlx_vector_vector_array_new(void);
int mlx_vector_vector_array_set(
    mlx_vector_vector_array* vec,
    const mlx_vector_vector_array src);
int mlx_vector_vector_array_free(mlx_vector_vector_array vec);
mlx_vector_vector_array mlx_vector_vector_array_new_data(
    const mlx_vector_array* data,
    size_t size);
mlx_vector_vector_array mlx_vector_vector_array_new_value(
    const mlx_vector_array val);
int mlx_vector_vector_array_set_data(
    mlx_vector_vector_array* vec,
    const mlx_vector_array* data,
    size_t size);
int mlx_vector_vector_array_set_value(
    mlx_vector_vector_array* vec,
    const mlx_vector_array val);
int mlx_vector_vector_array_append_data(
    mlx_vector_vector_array vec,
    const mlx_vector_array* data,
    size_t size);
int mlx_vector_vector_array_append_value(
    mlx_vector_vector_array vec,
    const mlx_vector_array val);
size_t mlx_vector_vector_array_size(mlx_vector_vector_array vec);
int mlx_vector_vector_array_get(
    mlx_vector_array* res,
    const mlx_vector_vector_array vec,
    size_t idx);

/**
 * A vector of int.
 */
typedef struct mlx_vector_int_ {
  void* ctx;
} mlx_vector_int;
mlx_vector_int mlx_vector_int_new(void);
int mlx_vector_int_set(mlx_vector_int* vec, const mlx_vector_int src);
int mlx_vector_int_free(mlx_vector_int vec);
mlx_vector_int mlx_vector_int_new_data(int* data, size_t size);
mlx_vector_int mlx_vector_int_new_value(int val);
int mlx_vector_int_set_data(mlx_vector_int* vec, int* data, size_t size);
int mlx_vector_int_set_value(mlx_vector_int* vec, int val);
int mlx_vector_int_append_data(mlx_vector_int vec, int* data, size_t size);
int mlx_vector_int_append_value(mlx_vector_int vec, int val);
size_t mlx_vector_int_size(mlx_vector_int vec);
int mlx_vector_int_get(int* res, const mlx_vector_int vec, size_t idx);

/**
 * A vector of string.
 */
typedef struct mlx_vector_string_ {
  void* ctx;
} mlx_vector_string;
mlx_vector_string mlx_vector_string_new(void);
int mlx_vector_string_set(mlx_vector_string* vec, const mlx_vector_string src);
int mlx_vector_string_free(mlx_vector_string vec);
mlx_vector_string mlx_vector_string_new_data(const char** data, size_t size);
mlx_vector_string mlx_vector_string_new_value(const char* val);
int mlx_vector_string_set_data(
    mlx_vector_string* vec,
    const char** data,
    size_t size);
int mlx_vector_string_set_value(mlx_vector_string* vec, const char* val);
int mlx_vector_string_append_data(
    mlx_vector_string vec,
    const char** data,
    size_t size);
int mlx_vector_string_append_value(mlx_vector_string vec, const char* val);
size_t mlx_vector_string_size(mlx_vector_string vec);
int mlx_vector_string_get(char** res, const mlx_vector_string vec, size_t idx);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
