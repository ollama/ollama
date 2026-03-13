/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_FAST_H
#define MLX_FAST_H

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
 * \defgroup fast Fast custom operations
 */
/**@{*/

typedef struct mlx_fast_cuda_kernel_config_ {
  void* ctx;
} mlx_fast_cuda_kernel_config;
mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void);
void mlx_fast_cuda_kernel_config_free(mlx_fast_cuda_kernel_config cls);

int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value);
int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose);
int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_cuda_kernel_ {
  void* ctx;
} mlx_fast_cuda_kernel;

mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory);

void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls);

int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream);

int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s);

typedef struct mlx_fast_metal_kernel_config_ {
  void* ctx;
} mlx_fast_metal_kernel_config;
mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void);
void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config cls);

int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value);
int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose);
int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_metal_kernel_ {
  void* ctx;
} mlx_fast_metal_kernel;

mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs);

void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls);

int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream);

int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s);
int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
