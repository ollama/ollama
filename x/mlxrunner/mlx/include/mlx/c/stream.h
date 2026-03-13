/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_STREAM_H
#define MLX_STREAM_H

#include <stdbool.h>

#include "mlx/c/device.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_stream Stream
 * MLX stream object.
 */
/**@{*/

/**
 * A MLX stream object.
 */
typedef struct mlx_stream_ {
  void* ctx;
} mlx_stream;

/**
 * Returns a new empty stream.
 */
mlx_stream mlx_stream_new(void);

/**
 * Returns a new stream on a device.
 */
mlx_stream mlx_stream_new_device(mlx_device dev);
/**
 * Set stream to provided src stream.
 */
int mlx_stream_set(mlx_stream* stream, const mlx_stream src);
/**
 * Free a stream.
 */
int mlx_stream_free(mlx_stream stream);
/**
 * Get stream description.
 */
int mlx_stream_tostring(mlx_string* str, mlx_stream stream);
/**
 * Check if streams are the same.
 */
bool mlx_stream_equal(mlx_stream lhs, mlx_stream rhs);
/**
 * Return the device of the stream.
 */
int mlx_stream_get_device(mlx_device* dev, mlx_stream stream);
/**
 * Return the index of the stream.
 */
int mlx_stream_get_index(int* index, mlx_stream stream);
/**
 * Synchronize with the provided stream.
 */
int mlx_synchronize(mlx_stream stream);
/**
 * Returns the default stream on the given device.
 */
int mlx_get_default_stream(mlx_stream* stream, mlx_device dev);
/**
 * Set default stream.
 */
int mlx_set_default_stream(mlx_stream stream);
/**
 * Returns the current default CPU stream.
 */
mlx_stream mlx_default_cpu_stream_new(void);

/**
 * Returns the current default GPU stream.
 */
mlx_stream mlx_default_gpu_stream_new(void);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
