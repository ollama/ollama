/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_ERROR_H
#define MLX_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_error Error management
 */
/**@{*/

typedef void (*mlx_error_handler_func)(const char* msg, void* data);

/**
 * Set the error handler.
 */
void mlx_set_error_handler(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*));

/**
 * Throw an error.
 */
void _mlx_error(const char* file, const int line, const char* fmt, ...);

/**
 * Throw an error. Macro which passes file name and line number to _mlx_error().
 */
#define mlx_error(...) _mlx_error(__FILE__, __LINE__, __VA_ARGS__)

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
