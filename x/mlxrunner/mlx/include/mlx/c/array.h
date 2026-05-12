/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_ARRAY_H
#define MLX_ARRAY_H

#include "mlx/c/string.h"

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// Complex number support
#ifdef _MSC_VER
#define _CRT_USE_C_COMPLEX_H
#include <complex.h>
typedef _Fcomplex mlx_complex64_t;
#else
#include <complex.h>
typedef float _Complex mlx_complex64_t;
#endif

#include "half.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_array Array
 * MLX N-dimensional array object.
 */
/**@{*/

/**
 * A N-dimensional array object.
 */
typedef struct mlx_array_ {
  void* ctx;
} mlx_array;

static mlx_array mlx_array_empty;

/**
 * Array element type.
 */
typedef enum mlx_dtype_ {
  MLX_BOOL,
  MLX_UINT8,
  MLX_UINT16,
  MLX_UINT32,
  MLX_UINT64,
  MLX_INT8,
  MLX_INT16,
  MLX_INT32,
  MLX_INT64,
  MLX_FLOAT16,
  MLX_FLOAT32,
  MLX_FLOAT64,
  MLX_BFLOAT16,
  MLX_COMPLEX64,
} mlx_dtype;

/**
 * Size of given mlx_dtype datatype in bytes.
 */
size_t mlx_dtype_size(mlx_dtype dtype);

/**
 * Get array description.
 */
int mlx_array_tostring(mlx_string* str, const mlx_array arr);

/**
 * New empty array.
 */
mlx_array mlx_array_new(void);

/**
 * Free an array.
 */
int mlx_array_free(mlx_array arr);

/**
 * New array from a bool scalar.
 */
mlx_array mlx_array_new_bool(bool val);
/**
 * New array from a int scalar.
 */
mlx_array mlx_array_new_int(int val);
/**
 * New array from a float32 scalar.
 */
mlx_array mlx_array_new_float32(float val);
/**
 * New array from a float scalar.
 * Same as float32.
 */
mlx_array mlx_array_new_float(float val);
/**
 * New array from a float64 scalar.
 */
mlx_array mlx_array_new_float64(double val);
/**
 * New array from a double scalar.
 * Same as float64.
 */
mlx_array mlx_array_new_double(double val);
/**
 * New array from a complex scalar.
 */
mlx_array mlx_array_new_complex(float real_val, float imag_val);
/**
 * New array from existing buffer.
 * @param data A buffer which will be copied.
 * @param shape Shape of the array.
 * @param dim Number of dimensions (size of `shape`).
 * @param dtype Type of array elements.
 */
mlx_array mlx_array_new_data(
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype);
/**
 * New array from existing buffer.
 * @param data A buffer which will be copied.
 * @param shape Shape of the array.
 * @param dim Number of dimensions (size of `shape`).
 * @param dtype Type of array elements.
 * @param dtor Callback for when the buffer is no longer needed.
 */
mlx_array mlx_array_new_data_managed(
    void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    void (*dtor)(void*));
/**
 * New array from existing buffer.
 * @param data A buffer which will be copied.
 * @param shape Shape of the array.
 * @param dim Number of dimensions (size of `shape`).
 * @param dtype Type of array elements.
 * @param payload Payload pointer passed to the `dtor` callback instead of
 * `data`.
 * @param dtor Callback for when the buffer is no longer needed.
 */
mlx_array mlx_array_new_data_managed_payload(
    void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    void* payload,
    void (*dtor)(void*));
/**
 * Set array to provided src array.
 */
int mlx_array_set(mlx_array* arr, const mlx_array src);
/**
 * Set array to a bool scalar.
 */
int mlx_array_set_bool(mlx_array* arr, bool val);
/**
 * Set array to a int scalar.
 */
int mlx_array_set_int(mlx_array* arr, int val);
/**
 * Set array to a float32 scalar.
 */
int mlx_array_set_float32(mlx_array* arr, float val);
/**
 * Set array to a float scalar.
 */
int mlx_array_set_float(mlx_array* arr, float val);
/**
 * Set array to a float64 scalar.
 */
int mlx_array_set_float64(mlx_array* arr, double val);
/**
 * Set array to a double scalar.
 */
int mlx_array_set_double(mlx_array* arr, double val);
/**
 * Set array to a complex scalar.
 */
int mlx_array_set_complex(mlx_array* arr, float real_val, float imag_val);
/**
 * Set array to specified data and shape.
 * @param arr Destination array.
 * @param data A buffer which will be copied.
 * @param shape Shape of the array.
 * @param dim Number of dimensions (size of `shape`).
 * @param dtype Type of array elements.
 */
int mlx_array_set_data(
    mlx_array* arr,
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype);

/**
 * The size of the array's datatype in bytes.
 */
size_t mlx_array_itemsize(const mlx_array arr);
/**
 * Number of elements in the array.
 */
size_t mlx_array_size(const mlx_array arr);
/**
 * The number of bytes in the array.
 */
size_t mlx_array_nbytes(const mlx_array arr);
/**
 * The array's dimension.
 */
size_t mlx_array_ndim(const mlx_array arr);
/**
 * The shape of the array.
 * Returns: a pointer to the sizes of each dimension.
 */
const int* mlx_array_shape(const mlx_array arr);
/**
 * The strides of the array.
 * Returns: a pointer to the sizes of each dimension.
 */
const size_t* mlx_array_strides(const mlx_array arr);
/**
 * The shape of the array in a particular dimension.
 */
int mlx_array_dim(const mlx_array arr, int dim);
/**
 * The array element type.
 */
mlx_dtype mlx_array_dtype(const mlx_array arr);

/**
 * Evaluate the array.
 */
int mlx_array_eval(mlx_array arr);

/**
 * Access the value of a scalar array.
 */
int mlx_array_item_bool(bool* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_uint8(uint8_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_uint16(uint16_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_uint32(uint32_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_uint64(uint64_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_int8(int8_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_int16(int16_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_int32(int32_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_int64(int64_t* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_float32(float* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_float64(double* res, const mlx_array arr);
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_complex64(mlx_complex64_t* res, const mlx_array arr);

#ifdef HAS_FLOAT16
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_float16(float16_t* res, const mlx_array arr);
#endif

#ifdef HAS_BFLOAT16
/**
 * Access the value of a scalar array.
 */
int mlx_array_item_bfloat16(bfloat16_t* res, const mlx_array arr);
#endif

/**
 * Returns a pointer to the array data, cast to `bool*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const bool* mlx_array_data_bool(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `uint8_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const uint8_t* mlx_array_data_uint8(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `uint16_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const uint16_t* mlx_array_data_uint16(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `uint32_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const uint32_t* mlx_array_data_uint32(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `uint64_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const uint64_t* mlx_array_data_uint64(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `int8_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const int8_t* mlx_array_data_int8(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `int16_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const int16_t* mlx_array_data_int16(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `int32_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const int32_t* mlx_array_data_int32(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `int64_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const int64_t* mlx_array_data_int64(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `float32*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const float* mlx_array_data_float32(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `float64*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const double* mlx_array_data_float64(const mlx_array arr);
/**
 * Returns a pointer to the array data, cast to `_Complex*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const mlx_complex64_t* mlx_array_data_complex64(const mlx_array arr);

#ifdef HAS_FLOAT16
/**
 * Returns a pointer to the array data, cast to `float16_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const float16_t* mlx_array_data_float16(const mlx_array arr);
#endif

#ifdef HAS_BFLOAT16
/**
 * Returns a pointer to the array data, cast to `bfloat16_t*`.
 * Array must be evaluated, otherwise returns NULL.
 */
const bfloat16_t* mlx_array_data_bfloat16(const mlx_array arr);
#endif

/**
 * Check if the array is available.
 * Internal function: use at your own risk.
 */
int _mlx_array_is_available(bool* res, const mlx_array arr);

/**
 * Wait on the array to be available. After this `_mlx_array_is_available`
 * returns `true`. Internal function: use at your own risk.
 */
int _mlx_array_wait(const mlx_array arr);

/**
 * Whether the array is contiguous in memory.
 * Internal function: use at your own risk.
 */
int _mlx_array_is_contiguous(bool* res, const mlx_array arr);

/**
 * Whether the array's rows are contiguous in memory.
 * Internal function: use at your own risk.
 */
int _mlx_array_is_row_contiguous(bool* res, const mlx_array arr);

/**
 * Whether the array's columns are contiguous in memory.
 * Internal function: use at your own risk.
 */
int _mlx_array_is_col_contiguous(bool* res, const mlx_array arr);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
