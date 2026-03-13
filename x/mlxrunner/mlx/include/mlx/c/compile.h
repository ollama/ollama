/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_COMPILE_H
#define MLX_COMPILE_H

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
 * \defgroup compile Compilation operations
 */
/**@{*/

typedef enum mlx_compile_mode_ {
  MLX_COMPILE_MODE_DISABLED,
  MLX_COMPILE_MODE_NO_SIMPLIFY,
  MLX_COMPILE_MODE_NO_FUSE,
  MLX_COMPILE_MODE_ENABLED
} mlx_compile_mode;
int mlx_compile(mlx_closure* res, const mlx_closure fun, bool shapeless);
int mlx_detail_compile(
    mlx_closure* res,
    const mlx_closure fun,
    uintptr_t fun_id,
    bool shapeless,
    const uint64_t* constants,
    size_t constants_num);
int mlx_detail_compile_clear_cache(void);
int mlx_detail_compile_erase(uintptr_t fun_id);
int mlx_disable_compile(void);
int mlx_enable_compile(void);
int mlx_set_compile_mode(mlx_compile_mode mode);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
