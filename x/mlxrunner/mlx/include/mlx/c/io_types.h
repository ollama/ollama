/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_IO_TYPES_H
#define MLX_IO_TYPES_H

#include <stdbool.h>

#include "mlx/c/string.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_io_types IO Types
 * MLX IO type objects.
 */
/**@{*/

/**
 * A MLX IO reader object.
 */
typedef struct mlx_io_reader_ {
  void* ctx;
} mlx_io_reader;
/**
 * A MLX IO writer object.
 */
typedef struct mlx_io_writer_ {
  void* ctx;
} mlx_io_writer;

/**
 * Virtual table for custom IO reader and writer objects.
 */
typedef struct mlx_io_vtable_ {
  bool (*is_open)(void*);
  bool (*good)(void*);
  size_t (*tell)(void*);
  void (*seek)(void*, int64_t off, int whence);
  void (*read)(void*, char* data, size_t n);
  void (*read_at_offset)(void*, char* data, size_t n, size_t off);
  void (*write)(void*, const char* data, size_t n);
  const char* (*label)(void*);
  void (*free)(void*);
} mlx_io_vtable;

/**
 * Returns a new custom IO reader.
 * `vtable` operates on user descriptor `desc`.
 */
mlx_io_reader mlx_io_reader_new(void* desc, mlx_io_vtable vtable);

/**
 * Get IO reader user descriptor.
 */
int mlx_io_reader_descriptor(void** desc_, mlx_io_reader io);

/**
 * Get IO reader description.
 */
int mlx_io_reader_tostring(mlx_string* str_, mlx_io_reader io);

/**
 * Free IO reader.
 *
 * Note that MLX arrays are lazily evaluated, so the underlying object may
 * be not freed right away. The ``free()`` callback from ``mlx_io_vtable``
 * will be called when the underlying object is actually freed.
 */
int mlx_io_reader_free(mlx_io_reader io);

/**
 * Returns a new custom IO writer.
 * `vtable` operates on user descriptor `desc`.
 */
mlx_io_writer mlx_io_writer_new(void* desc, mlx_io_vtable vtable);

/**
 * Get IO writer user descriptor.
 */
int mlx_io_writer_descriptor(void** desc_, mlx_io_writer io);

/**
 * Get IO writer description.
 */
int mlx_io_writer_tostring(mlx_string* str_, mlx_io_writer io);

/**
 * Free IO writer.
 *
 * Note that MLX arrays are lazily evaluated, so the underlying object may
 * be not freed right away. The ``free()`` callback from ``mlx_io_vtable``
 * will be called when the underlying object is actually freed.
 */
int mlx_io_writer_free(mlx_io_writer io);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
