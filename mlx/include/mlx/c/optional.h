/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_OPTIONAL_H
#define MLX_OPTIONAL_H

#include <stdbool.h>

#include "mlx/c/array.h"
#include "mlx/c/string.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_optional Optionals
 * MLX optional scalars.
 */
/**@{*/

/**
 * A int optional.
 */
typedef struct mlx_optional_int_ {
  int value;
  bool has_value;
} mlx_optional_int;

/**
 * A float optional.
 */
typedef struct mlx_optional_float_ {
  float value;
  bool has_value;
} mlx_optional_float;

/**
 * A dtype optional.
 */
typedef struct mlx_optional_dtype_ {
  mlx_dtype value;
  bool has_value;
} mlx_optional_dtype;

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
