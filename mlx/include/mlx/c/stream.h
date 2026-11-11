/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_STREAM_H
#define MLX_STREAM_H

#include <stdbool.h>
#include <stddef.h>

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
 * A per-thread stream object.
 *
 * The same thread-local stream can be used from any number of threads; the
 * stream's GPU command encoder is registered once per thread.
 */
typedef struct mlx_stream_thread_local_ {
  void* ctx;
} mlx_stream_thread_local;

/**
 * A vector of mlx_stream.
 * Forward declaration to avoid circular references.
 */
typedef struct mlx_vector_stream_ mlx_vector_stream;

/**
 * Returns a new empty stream.
 */
mlx_stream mlx_stream_new(void);

/**
 * Returns a new stream on a device.
 */
mlx_stream mlx_stream_new_device(mlx_device dev);
/**
 * Returns a new stream on a device that can be used from any thread.
 *
 * Streams are otherwise thread affine: a stream's GPU command encoder is
 * registered per thread, so evaluating on a stream from a thread other than
 * the one that created it fails. Streams returned here are registered
 * globally instead.
 *
 * MLX applies no synchronization to these streams -- it is the caller's
 * responsibility to ensure there are no data races on them.
 */
mlx_stream mlx_stream_new_thread_unsafe(mlx_device dev);

/**
 * Returns a new per-thread stream on a device.
 *
 * Streams created this way are unique per thread: evaluating the same
 * `mlx_stream_thread_local` from different threads yields distinct streams.
 */
mlx_stream_thread_local mlx_stream_thread_local_new(mlx_device dev);
/**
 * Set a per-thread stream to the provided src stream.
 */
int mlx_stream_thread_local_set(
    mlx_stream_thread_local* tls,
    const mlx_stream_thread_local src);
/**
 * Free a per-thread stream.
 */
int mlx_stream_thread_local_free(mlx_stream_thread_local tls);
/**
 * Returns the concrete stream backing a per-thread stream on the current
 * thread.
 */
int mlx_stream_from_thread_local(
    mlx_stream* res,
    const mlx_stream_thread_local tls);

/**
 * Get the vector of available streams.
 */
int mlx_get_streams(mlx_vector_stream* res);
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
 * Synchronize with the default stream.
 */
int mlx_synchronize_default(void);
/**
 * Synchronize with the stream corresponding to the current thread.
 */
int mlx_synchronize_thread_local(mlx_stream_thread_local tls);
/**
 * Destroy all streams created in the current thread.
 */
int mlx_clear_streams(void);
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
