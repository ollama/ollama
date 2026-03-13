/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_FFT_H
#define MLX_FFT_H

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
 * \defgroup fft FFT operations
 */
/**@{*/

int mlx_fft_fft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
int mlx_fft_fft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_fftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_fftshift(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_ifft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
int mlx_fft_ifft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_ifftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_ifftshift(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_irfft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
int mlx_fft_irfft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_irfftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_rfft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s);
int mlx_fft_rfft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);
int mlx_fft_rfftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
