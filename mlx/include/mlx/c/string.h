/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_STRING_H
#define MLX_STRING_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_string String
 * MLX string object.
 */
/**@{*/

/**
 * A MLX string object.
 */
typedef struct mlx_string_ {
  void* ctx;
} mlx_string;

/**
 * Returns a new empty string.
 */
mlx_string mlx_string_new(void);

/**
 * Returns a new string, copying contents from `str`, which must end with `\0`.
 */
mlx_string mlx_string_new_data(const char* str);

/**
 * Set string to src string.
 */
int mlx_string_set(mlx_string* str, const mlx_string src);

/**
 * Returns a pointer to the string contents.
 * The pointer is valid for the life duration of the string.
 */
const char* mlx_string_data(mlx_string str);

/**
 * Free string.
 */
int mlx_string_free(mlx_string str);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
