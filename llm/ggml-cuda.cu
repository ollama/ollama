/**
 * llama.cpp - git 8183159cf3def112f6d1fe94815fce70e1bffa12
 *
 * MIT License
 *
 * Copyright (c) 2023 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <atomic>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "ggml-cuda.h"
#include "ggml.h"

#define MIN_CC_DP4A 610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,   \
                cudaGetErrorString(err_));                                              \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#if CUDART_VERSION >= 12000
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d: %s\n",                         \
                    err_, __FILE__, __LINE__, cublasGetStatusString(err_));             \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#else
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#endif // CUDART_VERSION >= 11

#ifdef GGML_CUDA_F16
typedef half dfloat; // dequantize float
typedef half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
#endif //GGML_CUDA_F16

static __device__ __forceinline__ int get_int_from_int8(const int8_t * x8, const int & i32) {
    const uint16_t * x16 = (uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t * x8, const int & i32) {
    return *((int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);
typedef void (*to_fp32_cuda_t)(const void * __restrict__ x, float * __restrict__ y, int k, cudaStream_t stream);
typedef void (*dot_kernel_k_t)(const void * __restrict__ vx, const int ib, const int iqs, const float * __restrict__ y, float & v);
typedef void (*cpy_kernel_t)(const char * cx, char * cdst);
typedef void (*ggml_cuda_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
typedef void (*ggml_cuda_op_t)(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i, float * src0_ddf_i,
    float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main);

// QK = number of values after dequantization
// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
    half2   dm;             // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(ggml_fp16_t) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
    half2 dm;               // dm.x = delta, dm.y = min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
    half    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct {
    half2   ds;             // ds.x = delta, ds.y = sum
    int8_t  qs[QK8_0];      // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_fp16_t) + QK8_0, "wrong q8_1 block size/padding");

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);
typedef void (*allocate_tiles_cuda_t)(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc);
typedef void (*load_tiles_cuda_t)(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row);
typedef float (*vec_dot_q_mul_mat_cuda_t)(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ms, const int & i, const int & j, const int & k);

//================================= k-quants

#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K))
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half2 dm;                // super-block scale for quantized scales/mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
#ifdef GGML_QKK_64
    uint8_t scales[2]; // scales, quantized with 8 bits
#else
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
#endif
    half d;             // super-block scale
} block_q3_K;
//static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + K_SCALE_SIZE, "wrong q3_K block size/padding");

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
#ifdef GGML_QKK_64
typedef struct {
    half    d[2];              // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + QK_K/2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {
    half2 dm;                  // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2, "wrong q4_K block size/padding");
#endif

#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
#ifdef GGML_QKK_64
typedef struct {
    half d;                  // super-block scale
    int8_t scales[QK_K/16];  // block scales
    uint8_t qh[QK_K/8];      // quants, high bit
    uint8_t qs[QK_K/2];      // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == sizeof(ggml_fp16_t) + QK_K/2 + QK_K/8 + QK_K/16, "wrong q5_K block size/padding");
#else
typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");
#endif

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half    d;         // delta
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + 13*QK_K/16, "wrong q6_K block size/padding");

#define WARP_SIZE 32
#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#define CUDA_ADD_BLOCK_SIZE 256
#define CUDA_MUL_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_CPY_BLOCK_SIZE 32
#define CUDA_SCALE_BLOCK_SIZE 256
#define CUDA_ROPE_BLOCK_SIZE 256
#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

#ifndef GGML_CUDA_MMQ_Y
#define GGML_CUDA_MMQ_Y 64
#endif // GGML_CUDA_MMQ_Y

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_CUDA_DMMV_X
#define GGML_CUDA_DMMV_X 32
#endif
#ifndef GGML_CUDA_MMV_Y
#define GGML_CUDA_MMV_Y 1
#endif

#ifndef K_QUANTS_PER_ITERATION
#define K_QUANTS_PER_ITERATION 2
#else
static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#endif

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES]; // events for synchronizing multiple GPUs
};

static __global__ void add_f32(const float * x, const float * y, float * dst, const int kx, const int ky) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= kx) {
        return;
    }
    dst[i] = x[i] + y[i%ky];
}

static __global__ void add_f16_f32_f16(const half * x, const float * y, half * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = __hadd(x[i], __float2half(y[i]));
}

static __global__ void mul_f32(const float * x, const float * y, float * dst, const int kx, const int ky) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= kx) {
        return;
    }
    dst[i] = x[i] * y[i%ky];
}

static __global__ void gelu_f32(const float * x, float * dst, const int k) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    float xi = x[i];
    dst[i] = 0.5f*xi*(1.0f + tanhf(SQRT_2_OVER_PI*xi*(1.0f + GELU_COEF_A*xi*xi)));
}

static __global__ void silu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + expf(-x[i]));
}

static __global__ void norm_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const float eps = 1e-5f;

    float mean = 0.0f;
    float var = 0.0f;

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        const float xi = x[row*ncols + col];
        mean += xi;
        var += xi * xi;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        mean += __shfl_xor_sync(0xffffffff, mean, mask, 32);
        var += __shfl_xor_sync(0xffffffff, var, mask, 32);
    }

    mean /= ncols;
    var = var / ncols - mean * mean;
    const float inv_var = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[row*ncols + col] = (x[row*ncols + col] - mean) * inv_var;
    }
}

static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[row*ncols + col] = scale * x[row*ncols + col];
    }
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = x[ib].dm.x;
    const dfloat m = x[ib].dm.y;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {16.0f, 16.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = x[ib].dm.x;
    const dfloat m = x[ib].dm.y;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

//================================== k-quants

static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, float * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int tid = threadIdx.x;
#if QK_K == 256
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    float * y = yy + i*QK_K + 128*n;

    float dall = x[i].dm.x;
    float dmin = x[i].dm.y;
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#else
    const int is = tid/16;  // 0 or 1
    const int il = tid%16;  // 0...15
    const uint8_t q = x[i].qs[il] >> (2*is);
    float * y = yy + i*QK_K + 16*is + il;
    float dall = x[i].dm.x;
    float dmin = x[i].dm.y;
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, float * __restrict__ yy) {

    const int i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

#if QK_K == 256
    const int r = threadIdx.x/4;
    const int tid = r/2;
    const int is0 = r%2;
    const int l0 = 16*is0 + 4*(threadIdx.x%4);
    const int n = tid / 4;
    const int j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    float * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#else
    const int tid = threadIdx.x;
    const int is  = tid/16;  // 0 or 1
    const int il  = tid%16;  // 0...15
    const int im  = il/8;    // 0...1
    const int in  = il%8;    // 0...7

    float * y = yy + i*QK_K + 16*is + il;

    const uint8_t q = x[i].qs[il] >> (2*is);
    const uint8_t h = x[i].hmask[in] >> (2*is + im);
    const float   d = (float)x[i].d;

    if (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } else {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

#if QK_K == 256
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, float * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int i = blockIdx.x;

#if QK_K == 256
    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    float * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = x[i].dm.x;
    const float dmin = x[i].dm.y;

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
#else
    const int tid = threadIdx.x;
    const uint8_t * q = x[i].qs;
    float * y = yy + i*QK_K;
    const float d = (float)x[i].d[0];
    const float m = (float)x[i].d[1];
    y[tid+ 0] = d * (x[i].scales[0] & 0xF) * (q[tid] & 0xF) - m * (x[i].scales[0] >> 4);
    y[tid+32] = d * (x[i].scales[1] & 0xF) * (q[tid] >>  4) - m * (x[i].scales[1] >> 4);
#endif
}

static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, float * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int i = blockIdx.x;

#if QK_K == 256
    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    float * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = x[i].dm.x;
    const float dmin = x[i].dm.y;

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#else
    const int tid = threadIdx.x;
    const uint8_t q = x[i].qs[tid];
    const int im = tid/8;  // 0...3
    const int in = tid%8;  // 0...7
    const int is = tid/16; // 0 or 1
    const uint8_t h = x[i].qh[in] >> im;
    const float d = x[i].d;
    float * y = yy + i*QK_K + tid;
    y[ 0] = d * x[i].scales[is+0] * ((q & 0xF) - ((h >> 0) & 1 ? 0 : 16));
    y[32] = d * x[i].scales[is+2] * ((q >>  4) - ((h >> 4) & 1 ? 0 : 16));
#endif
}

static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, float * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int i = blockIdx.x;
#if QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int ip  = tid/32;   // ip is 0 or 1
    const int il  = tid - 32*ip; // 0...32
    const int is  = 8*ip + il/16;

    float * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#else

    // assume 32 threads
    const int tid = threadIdx.x;
    const int ip  = tid/16;         // 0 or 1
    const int il  = tid - 16*ip;    // 0...15

    float * y = yy + i*QK_K + 16*ip + il;

    const float d = x[i].d;

    const uint8_t   ql = x[i].ql[16*ip + il];
    const uint8_t   qh = x[i].qh[il] >> (2*ip);
    const int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}

static __global__ void dequantize_mul_mat_vec_q2_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...15
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2
    const int q_offset = 32*im + l0;
    const int s_offset = 8*im;
    const int y_offset = 128*im + l0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;

        const float dall = x[i].dm.x;
        const float dmin = x[i].dm.y;

        const uint32_t * a = (const uint32_t *)(x[i].scales + s_offset);
        aux[0] = a[0] & 0x0f0f0f0f;
        aux[1] = a[1] & 0x0f0f0f0f;
        aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
        aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                  + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                  + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                  + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                  + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                  + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                  + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                  +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
            sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                  + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

        }
        tmp += dall * sum1 - dmin * sum2;

    }
#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    const uint8_t * d = (const uint8_t *)uaux;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint32_t * s = (const uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        const float2 dall = __half22float2(x[i].dm);

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x * sum1 - dall.y * sum2;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

static __global__ void dequantize_mul_mat_vec_q3_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0....15 or 0...7

    const uint8_t m = 1 << (4*im);

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2
    const int q_offset =  32*im + l0;
    const int y_offset = 128*im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4*im;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;
        const uint8_t * h = x[i].hmask + l0;

        const uint16_t * a = (const uint16_t *)x[i].scales;
        utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
        utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
        utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
        utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < n; ++l) {
            sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                 + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                 + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                 + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
            sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                 + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                 + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
        }
        tmp += d * sum;

    }
#else

    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    const int in = offset/8;                                 // 0 or 1
    const int im = offset%8;                                 // 0...7

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint8_t * s = x[i].scales;

        const float dall = (float)x[i].d;

        float sum = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t hl = x[i].hmask[im+l] >> in;
            const uint8_t ql = q[l];
            sum += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

static __global__ void dequantize_mul_mat_vec_q4_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row > nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il  = tid/step;                            // 0...3
    const int ir  = tid - step*il;                       // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y1 = yy + i*QK_K + y_offset;
        const float   * y2 = y1 + 128;

        const float dall = x[i].dm.x;
        const float dmin = x[i].dm.y;

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
        const uint32_t * q1 = (const uint32_t *)(x[i].qs + q_offset);
        const uint32_t * q2 = q1 + 16;

        q32[0] = q1[0] & 0x0f0f0f0f;
        q32[1] = q1[0] & 0xf0f0f0f0;
        q32[2] = q2[0] & 0x0f0f0f0f;
        q32[3] = q2[0] & 0xf0f0f0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 4; ++l) {
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+ 4];
            s.z += y2[l] * q4[l+8]; s.w += y2[l+32] * q4[l+12];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
#else
        const uint16_t * q1 = (const uint16_t *)(x[i].qs + q_offset);
        const uint16_t * q2 = q1 + 32;

        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[0] & 0xf0f0;
        q16[2] = q2[0] & 0x0f0f;
        q16[3] = q2[0] & 0xf0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 2; ++l) {
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+2];
            s.z += y2[l] * q4[l+4]; s.w += y2[l+32] * q4[l+6];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
#endif

    }
#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const float d = (float)x[i].d[0];
        const float m = (float)x[i].d[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static __global__ void dequantize_mul_mat_vec_q5_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols) {

    const int row = blockIdx.x;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/2;  // 0...15
    const int ix  = threadIdx.x%2;

    const int il  = tid/4;     // 0...3
    const int ir  = tid - 4*il;// 0...3
    const int n   = 2;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1  = 1 << (2*im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * ql1 = x[i].qs + q_offset;
        const uint8_t * qh  = x[i].qh + l0;
        const float   * y1  = yy + i*QK_K + y_offset;
        const float   * y2  = y1 + 128;

        const float dall = x[i].dm.x;
        const float dmin = x[i].dm.y;

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        float4 sum = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        const uint16_t * q1 = (const uint16_t *)ql1;
        const uint16_t * q2 = q1 + 32;
        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[8] & 0x0f0f;
        q16[2] = (q1[0] >> 4) & 0x0f0f;
        q16[3] = (q1[8] >> 4) & 0x0f0f;
        q16[4] = q2[0] & 0x0f0f;
        q16[5] = q2[8] & 0x0f0f;
        q16[6] = (q2[0] >> 4) & 0x0f0f;
        q16[7] = (q2[8] >> 4) & 0x0f0f;
        for (int l = 0; l < n; ++l) {
            sum.x += y1[l+ 0] * (q4[l +0] + (qh[l+ 0] & (hm1 << 0) ? 16 : 0))
                   + y1[l+16] * (q4[l +2] + (qh[l+16] & (hm1 << 0) ? 16 : 0));
            sum.y += y1[l+32] * (q4[l +4] + (qh[l+ 0] & (hm1 << 1) ? 16 : 0))
                   + y1[l+48] * (q4[l +6] + (qh[l+16] & (hm1 << 1) ? 16 : 0));
            sum.z += y2[l+ 0] * (q4[l +8] + (qh[l+ 0] & (hm2 << 0) ? 16 : 0))
                   + y2[l+16] * (q4[l+10] + (qh[l+16] & (hm2 << 0) ? 16 : 0));
            sum.w += y2[l+32] * (q4[l+12] + (qh[l+ 0] & (hm2 << 1) ? 16 : 0))
                   + y2[l+48] * (q4[l+14] + (qh[l+16] & (hm2 << 1) ? 16 : 0));
            smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                  + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
        }
        tmp += dall * (sum.x * sc[0] + sum.y * sc[1] + sum.z * sc[4] + sum.w * sc[5]) - dmin * smin;
    }

#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);
    const int step = tid * K_QUANTS_PER_ITERATION;
    const int im = step/8;
    const int in = step%8;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const int8_t  * s = x[i].scales;
        const float   * y = yy + i*QK_K + step;
        const float     d = x[i].d;
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            const uint8_t h = x[i].qh[in+j] >> im;
            sum += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

static __global__ void dequantize_mul_mat_vec_q6_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

#if QK_K == 256

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif
    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = x[i].d;

#if K_QUANTS_PER_ITERATION == 1
        float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                  + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                  + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                  + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                  + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                  + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                  + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                  +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
        tmp += sum;
#else
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                 + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                 + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                 + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        tmp += sum;
#endif

    }

#else

    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = x[i].ql + step;
        const uint8_t * qh = x[i].qh + step;
        const int8_t  * s  = x[i].scales;

        const float d = x[i+0].d;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static __device__ void convert_f16(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const half * x = (const half *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x = x[ib + iqs + 0];
    v.y = x[ib + iqs + 1];
}

static __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    const int ix = blockDim.x*blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int iy = blockDim.y*blockIdx.y + threadIdx.y;

    const int i_padded = iy*kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib = i_padded / QK8_1; // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds.x = d;
    y[ib].ds.y = sum;
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_block(const void * __restrict__ vx, float * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + 2*threadIdx.x;

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = __dp4a(vi0, u[2*i+0], sumi);
        sumi = __dp4a(vi1, u[2*i+1], sumi);
    }

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * __half2float(ds8.x) - (8*vdr/QI4_0) * __half2float(ds8.y));
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = __dp4a(vi0, u[2*i+0], sumi);
        sumi = __dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef GGML_CUDA_F16
    const half2 tmp = __hmul2(dm4, ds8);
    const float d4d8 = __half2float(tmp.x);
    const float m4s8 = __half2float(tmp.y);
#else
    const float d4d8 = __half2float(dm4.x) * __half2float(ds8.x);
    const float m4s8 = __half2float(dm4.y) * __half2float(ds8.y);
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = __dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = __dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi*__half2float(ds8.x) - (16*vdr/QI5_0) * __half2float(ds8.y));
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = __dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = __dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_CUDA_F16
    const half2 tmp = __hmul2(dm5, ds8);
    const float d5d8 = __half2float(tmp.x);
    const float m5s8 = __half2float(tmp.y);
#else
    const float d5d8 = __half2float(dm5.x) * __half2float(ds8.x);
    const float m5s8 = __half2float(dm5.y) * __half2float(ds8.y);
#endif // GGML_CUDA_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);

#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const half2 & ds8_1) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = __dp4a(v[i], u[i], sumi);
    }

    return sumi * d8_0 * __half2float(ds8_1.x);
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = __dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_CUDA_F16
    const half2 tmp = __hmul2(dm8, ds8);
    const float d8d8 = __half2float(tmp.x);
    const float m8s8 = __half2float(tmp.y);
#else
    const float d8d8 = __half2float(dm8.x) * __half2float(ds8.x);
    const float m8s8 = __half2float(dm8.y) * __half2float(ds8.y);
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

static __device__ __forceinline__ void allocate_tiles_q4_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int  tile_x_qs[GGML_CUDA_MMQ_Y * (WARP_SIZE)       + GGML_CUDA_MMQ_Y];
    __shared__ float tile_x_d[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI4_0) + GGML_CUDA_MMQ_Y/QI4_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (block_q4_0 *) vx;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

//     const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
//     const int kbxd = k % blocks_per_tile_x_row;

// #pragma unroll
//     for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI4_0) {
//         FIXME out-of-bounds
//         const int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

//         if (i >= GGML_CUDA_MMQ_Y) {
//             return;
//         }

//         const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

//         x_dm[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd].x = bxi->d;
//     }
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * (2*WARP_SIZE) + kyqs + l];
        u[2*l+1] = y_qs[j * (2*WARP_SIZE) + kyqs + l + QI4_0];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (2*WARP_SIZE/QI8_1) + 2*k/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ void allocate_tiles_q4_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_qs[GGML_CUDA_MMQ_Y * (WARP_SIZE) +     + GGML_CUDA_MMQ_Y];
    __shared__ half2 tile_x_dm[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI4_1) + GGML_CUDA_MMQ_Y/QI4_1];

    *x_ql = tile_x_qs;
    *x_dm = tile_x_dm;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI4_1) {
        int i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * (2*WARP_SIZE) + kyqs + l];
        u[2*l+1] = y_qs[j * (2*WARP_SIZE) + kyqs + l + QI4_1];
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (2*WARP_SIZE/QI8_1) + 2*k/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

static __device__ __forceinline__ void allocate_tiles_q5_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int  tile_x_ql[GGML_CUDA_MMQ_Y * (2*WARP_SIZE)     + GGML_CUDA_MMQ_Y];
    __shared__ float tile_x_d[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI5_0) + GGML_CUDA_MMQ_Y/QI5_0];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    const float * x_dmf = (float *) x_dm;

    int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * (2*WARP_SIZE) + kyqs + l];
        u[2*l+1] = y_qs[j * (2*WARP_SIZE) + kyqs + l + QI5_0];
    }

    return vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_ds[j * (2*WARP_SIZE/QI8_1) + 2*k/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ void allocate_tiles_q5_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[GGML_CUDA_MMQ_Y * (2*WARP_SIZE)     + GGML_CUDA_MMQ_Y];
    __shared__ half2 tile_x_dm[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI5_1) + GGML_CUDA_MMQ_Y/QI5_1];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI5_1) {
        int i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * (2*WARP_SIZE) + kyqs + l];
        u[2*l+1] = y_qs[j * (2*WARP_SIZE) + kyqs + l + QI5_1];
    }

    return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (2*WARP_SIZE/QI8_1) + 2*k/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, bq8_1->ds);
}

static __device__ __forceinline__ void allocate_tiles_q8_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int  tile_x_qs[GGML_CUDA_MMQ_Y * (WARP_SIZE)       + GGML_CUDA_MMQ_Y];
    __shared__ float tile_x_d[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI8_0) + GGML_CUDA_MMQ_Y/QI8_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbx] = bxi->d;
    }

//     const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;
//     const int kbxd = k % blocks_per_tile_x_row;

// #pragma unroll
//     for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI8_0) {
//         FIXME out-of-bounds
//         const int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

// #if GGML_CUDA_MMQ_Y < 64
//         if (i >= GGML_CUDA_MMQ_Y) {
//             return;
//         }
// #endif // GGML_CUDA_MMQ_Y < 64

//         const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

//         x_dm[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd].x = bxi->d;
//     }
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const float * x_dmf = (float *) x_dm;

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_ds[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

#define VDR_q2_K_q8_1 1

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        int sc_high = sc >> 4;
        sc_high |= sc_high <<  8;
        sc_high |= sc_high << 16;
        sumf_m += d8[i] * __dp4a(sc_high, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dmf = __half22float2(dm);

    return dmf.x*sumf_d - dmf.y*sumf_m;
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int u[QR2_K];
    float d8[QR2_K];

    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds.x;
    }

    return vec_dot_q2_K_q8_1_impl(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ void allocate_tiles_q2_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[GGML_CUDA_MMQ_Y * (WARP_SIZE)       + GGML_CUDA_MMQ_Y];
    __shared__ half2 tile_x_dm[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI2_K) + GGML_CUDA_MMQ_Y/QI2_K];
    __shared__ int   tile_x_sc[GGML_CUDA_MMQ_Y * (WARP_SIZE/4)     + GGML_CUDA_MMQ_Y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI2_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % GGML_CUDA_MMQ_Y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const int bq8_offset = QR2_K * (kqsx / QI8_1);
    const int scale_offset = kqsx - kqsx % QI8_1 + (kqsx % QI8_1) / (QI8_1/2);

    const uint8_t * scales = ((uint8_t *) (x_sc + i * (WARP_SIZE/4) + i / 4)) + kbx*16 + scale_offset;

    int u[QR2_K];
    float d8[QR2_K];

    for (int l = 0; l < QR2_K; ++ l) {
        const int y_qs_index = j * (QR2_K*WARP_SIZE) + kbx * (QR2_K*QI2_K) + (bq8_offset + l)*QI8_1 + kqsx % QI8_1;
        u[l]  = y_qs[y_qs_index];
        d8[l] = y_ds[y_qs_index / QI8_1].x;
    }

    return vec_dot_q2_K_q8_1_impl(x_ql[i * (WARP_SIZE + 1) + k], u, scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], d8);
}

#define VDR_q3_K_q8_1 1

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf = 0.0f;

    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int u[QR3_K];
    float d8[QR3_K];

    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds.x;
    }

    return vec_dot_q3_K_q8_1_impl(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ void allocate_tiles_q3_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[GGML_CUDA_MMQ_Y * (WARP_SIZE)       + GGML_CUDA_MMQ_Y];
    __shared__ half2 tile_x_dm[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI3_K) + GGML_CUDA_MMQ_Y/QI3_K];
    __shared__ int   tile_x_qh[GGML_CUDA_MMQ_Y * (WARP_SIZE/2)     + GGML_CUDA_MMQ_Y/2];
    __shared__ int   tile_x_sc[GGML_CUDA_MMQ_Y * (WARP_SIZE/4)     + GGML_CUDA_MMQ_Y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % GGML_CUDA_MMQ_Y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd].x = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8(bxi->scales, k % (QI3_K/4));
    }
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const int bq8_offset = QR3_K * (kqsx / (QI3_K/2));
    const int scale_offset = kqsx - kqsx % QI8_1 + (kqsx % QI8_1) / (QI8_1/2);

    const uint8_t * scales = ((uint8_t *) (x_sc + i * (WARP_SIZE/4) + i / 4)) + kbx*16;

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + kqsx % (QI3_K/2)] >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

    for (int l = 0; l < QR3_K; ++ l) {
        const int y_qs_index = j * (QR3_K*WARP_SIZE) + kbx * (QR3_K*QI3_K) + (bq8_offset + l)*QI8_1 + kqsx % QI8_1;
        u[l]  = y_qs[y_qs_index];
        d8[l] = y_ds[y_qs_index / QI8_1].x;
    }

    return vec_dot_q3_K_q8_1_impl(x_ql[i * (WARP_SIZE + 1) + k], vh, u, scales, scale_offset,
                                  x_dm[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx].x, d8);
}

#define VDR_q4_K_q8_1 2

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    return __half2float(dm4.x)*sumf_d - __half2float(dm4.y)*sumf_m;

#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#ifndef GGML_QKK_64
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds.x;

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl(v, u, sc, m, bq4_K->dm, d8);

#else

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    const uint16_t * a = (const uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    const float dall = bq4_K->d[0];
    const float dmin = bq4_K->d[1];

    const float d8_1 = bq8_1[0].ds.x;
    const float d8_2 = bq8_1[1].ds.x;

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * q4 = (const int *)bq4_K->qs + (iqs/2);
    const int v1 = q4[0];
    const int v2 = q4[4];

    const int dot1 = __dp4a(ui2, v2 & 0x0f0f0f0f, __dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    const int dot2 = __dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, __dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    const int dot3 = __dp4a(0x01010101, ui2, __dp4a(0x01010101, ui1, 0));
    const int dot4 = __dp4a(0x01010101, ui4, __dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    return dall * sumf_d - dmin * sumf_m;

#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A

#endif
}

static __device__ __forceinline__ void allocate_tiles_q4_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[GGML_CUDA_MMQ_Y * (WARP_SIZE)       + GGML_CUDA_MMQ_Y];
    __shared__ half2 tile_x_dm[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI4_K) + GGML_CUDA_MMQ_Y/QI4_K];
    __shared__ int   tile_x_sc[GGML_CUDA_MMQ_Y * (WARP_SIZE/8)     + GGML_CUDA_MMQ_Y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;           // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % GGML_CUDA_MMQ_Y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % GGML_CUDA_MMQ_Y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_uint8_aligned(bxi->scales, k % (QI4_K/8));
    }
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // kqsx is in 0,2...30. bq8_offset = 2 * (kqsx/4) -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((kqsx/2) / (QI8_1/2));

    v[0] = x_ql[i * (WARP_SIZE + 1) + 4 * bq8_offset + (kqsx/2) % 4 + 0];
    v[1] = x_ql[i * (WARP_SIZE + 1) + 4 * bq8_offset + (kqsx/2) % 4 + 4];

    const uint16_t * scales = (const uint16_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + kbx * 4];
    uint16_t aux[2];
    const int l = bq8_offset/2;
    if (l < 2) {
        aux[0] = scales[l+0] & 0x3f3f;
        aux[1] = scales[l+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[l+2] >> 0) & 0x0f0f) | ((scales[l-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[l+2] >> 4) & 0x0f0f) | ((scales[l-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int l = 0; l < QR4_K; ++l) {
        const int kqsy = j * (QR4_K*WARP_SIZE) + kbx * (QR4_K*QI4_K) + (bq8_offset + l) * QI8_1 + (kqsx/2) % (QI8_1/2);
        u[2*l+0] = y_qs[kqsy + 0*(QI8_1/2)];
        u[2*l+1] = y_qs[kqsy + 1*(QI8_1/2)];
        d8[l] = y_ds[kqsy / QI8_1].x;
    }

    return vec_dot_q4_K_q8_1_impl(v, u, sc, m, x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K + kbx], d8);
}

#define VDR_q5_K_q8_1 2

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = __dp4a(v0i, u[2*i+0], __dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+0], __dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    return __half2float(dm5.x)*sumf_d - __half2float(dm5.y)*sumf_m;

#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#ifndef GGML_QKK_64
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds.x;

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl(vl, vh, u, sc, m, bq5_K->dm, d8);

#else

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    const int8_t * s = bq5_K->scales;

    const float d = bq5_K->d;

    const float d8_1 = bq8_1[0].ds.x;
    const float d8_2 = bq8_1[1].ds.x;

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * ql = (const int *)bq5_K->qs + (iqs/2);
    const int vl1 = ql[0];
    const int vl2 = ql[4];

    const int step = 4 * (iqs/2); // 0, 4, 8, 12
    const int im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    const int in = step%8; // 0, 4, 0, 4
    const int vh = (*((const int *)(bq5_K->qh + in))) >> im;

    const int v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    const int v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    const int v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    const int v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    const float sumf_d = d8_1 * (__dp4a(ui1, v1, 0) * s[0] + __dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (__dp4a(ui3, v3, 0) * s[2] + __dp4a(ui4, v4, 0) * s[3]);

    return d * sumf_d;

#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A

#endif
}

static __device__ __forceinline__ void allocate_tiles_q5_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[GGML_CUDA_MMQ_Y * (WARP_SIZE)       + GGML_CUDA_MMQ_Y];
    __shared__ half2 tile_x_dm[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI5_K) + GGML_CUDA_MMQ_Y/QI5_K];
    __shared__ int   tile_x_qh[GGML_CUDA_MMQ_Y * (WARP_SIZE/4)     + GGML_CUDA_MMQ_Y/4];
    __shared__ int   tile_x_sc[GGML_CUDA_MMQ_Y * (WARP_SIZE/8)     + GGML_CUDA_MMQ_Y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;           // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % GGML_CUDA_MMQ_Y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI5_K/4);

        x_qh[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8(bxi->qh, k % (QI5_K/4));
    }

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % GGML_CUDA_MMQ_Y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_uint8_aligned(bxi->scales, k % (QI5_K/8));
    }
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    int   vl[2];
    int   vh[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR5_K * ((kqsx/2) / (QI8_1/2));

    vl[0] = x_ql[i * (WARP_SIZE + 1) + 4 * bq8_offset + (kqsx/2) % 4 + 0];
    vl[1] = x_ql[i * (WARP_SIZE + 1) + 4 * bq8_offset + (kqsx/2) % 4 + 4];

    vh[0] = x_qh[i * (WARP_SIZE/4) + i/4 + (kqsx/2) % 4 + 0] >> bq8_offset;
    vh[1] = x_qh[i * (WARP_SIZE/4) + i/4 + (kqsx/2) % 4 + 4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + kbx * 4];
    uint16_t aux[2];
    const int l = bq8_offset/2;
    if (l < 2) {
        aux[0] = scales[l+0] & 0x3f3f;
        aux[1] = scales[l+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[l+2] >> 0) & 0x0f0f) | ((scales[l-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[l+2] >> 4) & 0x0f0f) | ((scales[l-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int l = 0; l < QR5_K; ++l) {
        const int kqsy = j * (QR5_K*WARP_SIZE) + kbx * (QR5_K*QI5_K) + (bq8_offset + l) * QI8_1 + (kqsx/2) % (QI8_1/2);
        u[2*l+0] = y_qs[kqsy + 0*(QI8_1/2)];
        u[2*l+1] = y_qs[kqsy + 1*(QI8_1/2)];
        d8[l] = y_ds[kqsy / QI8_1].x;
    }

    return vec_dot_q5_K_q8_1_impl(vl, vh, u, sc, m, x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K + kbx], d8);
}

#define VDR_q6_K_q8_1 1

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf = 0.0f;

    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + 2*i].ds.x;
    }

    return vec_dot_q6_K_q8_1_impl(vl, vh, u, scales, bq6_K->d, d8);
}

static __device__ __forceinline__ void allocate_tiles_q6_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[GGML_CUDA_MMQ_Y * (WARP_SIZE)       + GGML_CUDA_MMQ_Y];
    __shared__ half2 tile_x_dm[GGML_CUDA_MMQ_Y * (WARP_SIZE/QI6_K) + GGML_CUDA_MMQ_Y/QI6_K];
    __shared__ int   tile_x_qh[GGML_CUDA_MMQ_Y * (WARP_SIZE/2)     + GGML_CUDA_MMQ_Y/2];
    __shared__ int   tile_x_sc[GGML_CUDA_MMQ_Y * (WARP_SIZE/8)     + GGML_CUDA_MMQ_Y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    __builtin_assume(i_offset >= 0);
    __builtin_assume(i_offset <  8);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->ql, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;           // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % GGML_CUDA_MMQ_Y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd].x = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI6_K/2);

        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = get_int_from_uint8(bxi->qh, k % (QI6_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < GGML_CUDA_MMQ_Y; i0 += 8 * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % GGML_CUDA_MMQ_Y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    __builtin_assume(i >= 0);
    __builtin_assume(i <  GGML_CUDA_MMQ_Y);
    __builtin_assume(j >= 0);
    __builtin_assume(j <  WARP_SIZE);
    __builtin_assume(k >= 0);
    __builtin_assume(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const int bq8_offset = 2 * QR6_K * (kqsx / (QI6_K/2)) + (kqsx % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (kqsx / (QI6_K/2)) + (kqsx % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((kqsx % (QI6_K/2)) / (QI6_K/4));

    const int vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI6_K/2) + (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4)] >> vh_shift;

    const int x_sc_offset = i * (WARP_SIZE/8) + i/8 + kbx * (QI6_K/8);
    const int8_t * scales = ((int8_t *) (x_sc + x_sc_offset)) + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

    for (int l = 0; l < QR6_K; ++l) {
        const int kqsy = j * (QR6_K*WARP_SIZE) + kbx * (QR6_K*QI6_K) + (bq8_offset + 2*l)*QI8_1 + kqsx % QI8_1;
        u[l]  = y_qs[kqsy];
        d8[l] = y_ds[kqsy / QI8_1].x;
    }

    return vec_dot_q6_K_q8_1_impl(x_ql[i * (WARP_SIZE + 1) + k], vh, u, scales,
                                  x_dm[i * (WARP_SIZE/QI6_K) + i/QI6_K + kbx].x, d8);
}

template <int qk, int qr, int qi, typename block_q_t,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __global__ void mul_mat_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE / qi;

    const int & ncols_dst = ncols_y;

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    const int row_dst_0 = blockIdx.x*GGML_CUDA_MMQ_Y;
    const int & row_x_0 = row_dst_0;
    const int row_dst = row_dst_0 + tid_x;

    const int col_dst_0 = blockIdx.y*WARP_SIZE;
    const int & col_y_0 = col_dst_0;

    int   * tile_x_ql = nullptr;
    half2 * tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    const int blocks_per_tile_y_col = qr*WARP_SIZE/QI8_1;

    __shared__ int    tile_y_qs[(WARP_SIZE) * (qr*WARP_SIZE)];
    __shared__ half2  tile_y_ds[(WARP_SIZE) * blocks_per_tile_y_col];

    float sum[GGML_CUDA_MMQ_Y/WARP_SIZE][4] = {0.0f};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0*blocks_per_row_x + ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   tid_y, nrows_x-row_x_0-1, tid_x, blocks_per_row_x);

        for (int ir = 0; ir < qr; ++ir) {
            const int kqs = ir*WARP_SIZE + tid_x;
            const int kbxd = kqs / QI8_1;

            for (int i = 0; i < WARP_SIZE; i += 8) {
                const int col_y_eff = min(col_y_0 + tid_y + i, ncols_y-1); // to prevent out-of-bounds memory accesses

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                tile_y_qs[(tid_y + i) * (qr*WARP_SIZE) + kqs] = get_int_from_int8_aligned(by0->qs, tid_x % QI8_1);
            }
        }

        for (int ids0 = 0; ids0 < WARP_SIZE; ids0 += 8 * (WARP_SIZE/blocks_per_tile_y_col)) {
            const int ids = (ids0 + tid_y * (WARP_SIZE/blocks_per_tile_y_col) + tid_x / blocks_per_tile_y_col) % WARP_SIZE;
            const int kby = tid_x % blocks_per_tile_y_col;
            const int col_y_eff = min(col_y_0 + ids, ncols_y-1);
            tile_y_ds[ids * (qr*WARP_SIZE/QI8_1) + kby] = y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kby].ds;
        }

        __syncthreads();

#if __CUDA_ARCH__ >= 700 // Unrolling the loop is slower on Pascal
#pragma unroll
#endif // __CUDA_ARCH__ >= 700
        for (int k = 0; k < WARP_SIZE; k += vdr) {
#pragma unroll
            for (int j = 0; j < WARP_SIZE; j += 8) {
#pragma unroll
                for (int i = 0; i < GGML_CUDA_MMQ_Y; i += WARP_SIZE) {
                    sum[i/WARP_SIZE][j/8] += vec_dot(tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                                                     tid_x + i, tid_y + j, k);
                }
            }
        }

        __syncthreads();
    }


    if (row_dst >= nrows_dst) {
        return;
    }

    for (int j = 0; j < WARP_SIZE; j += 8) {
        const int col_dst = col_dst_0 + j + tid_y;

        if (col_dst >= ncols_dst) {
            return;
        }

        for (int i = 0; i < GGML_CUDA_MMQ_Y; i += WARP_SIZE) {
            dst[col_dst*nrows_dst + row_dst + i] = sum[i/WARP_SIZE][j/8];
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst, const int ncols, const int nrows) {
    const int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = 0; i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i + threadIdx.x / (qi/vdr); // x block index

        const int iby = (i + threadIdx.x / (qi/vdr)) * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_mul_mat_vec(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_CUDA_F16
    half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_CUDA_F16

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_CUDA_F16
            tmp += __hmul2(v, {
                y[iybs + iqs + j/qr + 0],
                y[iybs + iqs + j/qr + y_offset]
            });
#else
            tmp += v.x * y[iybs + iqs + j/qr + 0];
            tmp += v.y * y[iybs + iqs + j/qr + y_offset];
#endif // GGML_CUDA_F16
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
#ifdef GGML_CUDA_F16
        dst[row] = tmp.x + tmp.y;
#else
        dst[row] = tmp;
#endif // GGML_CUDA_F16
    }
}

static __global__ void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y) {

    const half * x = (const half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __global__ void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x, const int channel_x_divisor) {

    const half * x = (const half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / channel_x_divisor;

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;

        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __device__ void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

static __device__ void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    half * dsti = (half *) cdsti;

    *dsti = __float2half(*xi);
}

template <cpy_kernel_t cpy_1>
static __global__ void cpy_f32_f16(const char * cx, char * cdst, const int ne,
                                   const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
                                   const int ne10, const int ne11, const int nb10, const int nb11, const int nb12) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int i02 = i / (ne00*ne01);
    const int i01 = (i - i02*ne01*ne00) / ne00;
    const int i00 = i - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02;

    const int i12 = i / (ne10*ne11);
    const int i11 = (i - i12*ne10*ne11) / ne10;
    const int i10 = i - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = i10*nb10 + i11*nb11 + i12*nb12;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

// rope == RoPE == rotary positional embedding
static __global__ void rope_f32(const float * x, float * dst, const int ncols, const float p0,
                                const float p_delta, const int p_delta_rows, const float theta_scale) {
    const int col = 2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;

    const float theta = (p0 + p_delta * (row/p_delta_rows))*powf(theta_scale, col/2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

static __global__ void rope_glm_f32(const float * x, float * dst, const int ncols, const float p, const float block_p, const float theta_scale) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    const int half_n_dims = ncols/4;

    if (col >= half_n_dims) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;

    const float col_theta_scale = powf(theta_scale, col);

    const float theta = p*col_theta_scale;
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + half_n_dims];

    dst[i + 0]           = x0*cos_theta - x1*sin_theta;
    dst[i + half_n_dims] = x0*sin_theta + x1*cos_theta;

    const float block_theta = block_p*col_theta_scale;
    const float sin_block_theta = sinf(block_theta);
    const float cos_block_theta = cosf(block_theta);

    const float x2 = x[i + half_n_dims * 2];
    const float x3 = x[i + half_n_dims * 3];

    dst[i + half_n_dims * 2] = x2*cos_block_theta - x3*sin_block_theta;
    dst[i + half_n_dims * 3] = x2*sin_block_theta + x3*cos_block_theta;
}

static __global__ void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    const int row = blockDim.y*blockIdx.y + threadIdx.y;

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    // dst[i] = col > n_past + row ? -INFINITY : x[i];
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
}

// the CUDA soft max implementation differs from the CPU implementation
// instead of doubles floats are used
// values are also not normalized to the maximum value by subtracting it in the exponential function
// theoretically these changes could cause problems with rounding error and arithmetic overflow but for LLaMa it seems to be fine
static __global__ void soft_max_f32(const float * x, float * dst, const int ncols) {
    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int block_size = blockDim.x;
    const int tid = threadIdx.x;

    float tmp = 0.0;

    for (int block_start = 0; block_start < ncols; block_start += block_size) {
        const int col = block_start + tid;

        if (col >= ncols) {
            break;
        }

        const int i = row*ncols + col;
        const float val = expf(x[i]);
        tmp += val;
        dst[i] = val;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    for (int block_start = 0; block_start < ncols; block_start += block_size) {
        const int col = block_start + tid;

        if (col >= ncols) {
            break;
        }

        const int i = row*ncols + col;
        dst[i] /= tmp;
    }
}

static __global__ void scale_f32(const float * x, float * dst, const float scale, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i];
}

static void add_f32_cuda(const float * x, const float * y, float * dst, const int kx, const int ky, cudaStream_t stream) {
    const int num_blocks = (kx + CUDA_ADD_BLOCK_SIZE - 1) / CUDA_ADD_BLOCK_SIZE;
    add_f32<<<num_blocks, CUDA_ADD_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
}

static void add_f16_f32_f16_cuda(const half * x, const float * y, half * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_ADD_BLOCK_SIZE - 1) / CUDA_ADD_BLOCK_SIZE;
    add_f16_f32_f16<<<num_blocks, CUDA_ADD_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

static void mul_f32_cuda(const float * x, const float * y, float * dst, const int kx, const int ky, cudaStream_t stream) {
    const int num_blocks = (kx + CUDA_MUL_BLOCK_SIZE - 1) / CUDA_MUL_BLOCK_SIZE;
    mul_f32<<<num_blocks, CUDA_MUL_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
}

static void gelu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void silu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    norm_f32<<<nrows, block_dims, 0, stream>>>(x, dst, ncols);
}

static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    rms_norm_f32<<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
}

static void quantize_row_q8_1_cuda(const float * x, void * vy, const int kx, const int ky, const int kx_padded, cudaStream_t stream) {
    const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ky, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}

static void dequantize_row_q4_0_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK4_0, QR4_0, dequantize_q4_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q4_1_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK4_1, QR4_1, dequantize_q4_1><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q5_0_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK5_0, QR5_0, dequantize_q5_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q5_1_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK5_1, QR5_1, dequantize_q5_1><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q8_0_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK8_0, QR8_0, dequantize_q8_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q2_K_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q2_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

static void dequantize_row_q3_K_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q3_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

static void dequantize_row_q4_K_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

static void dequantize_row_q5_K_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q5_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

static void dequantize_row_q6_K_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q6_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

static void dequantize_mul_mat_vec_q4_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q4_1_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q5_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q5_1_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q8_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q2_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q2_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q3_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q3_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q4_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q4_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void dequantize_mul_mat_vec_q5_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const dim3 block_dims(32, 1, 1);
    dequantize_mul_mat_vec_q5_k<<<nrows, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q6_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q6_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static void mul_mat_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q2_K_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI2_K, block_q2_K, VDR_q2_K_q8_1, vec_dot_q2_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI3_K, block_q3_K, VDR_q3_K_q8_1, vec_dot_q3_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI4_K, block_q4_K, VDR_q4_K_q8_1, vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI5_K, block_q5_K, VDR_q5_K_q8_1, vec_dot_q5_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI6_K, block_q6_K, VDR_q6_K_q8_1, vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void convert_fp16_to_fp32_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<1, 1, convert_f16><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void convert_mul_mat_vec_f16_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<1, 1, convert_f16>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_row_q5_0_cuda;
        case GGML_TYPE_Q5_1:
            return dequantize_row_q5_1_cuda;
        case GGML_TYPE_Q8_0:
            return dequantize_row_q8_0_cuda;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_F16:
            return convert_fp16_to_fp32_cuda;
        default:
            return nullptr;
    }
}

static void ggml_mul_mat_q4_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK4_0, QR4_0, QI4_0, block_q4_0, allocate_tiles_q4_0, load_tiles_q4_0<false>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK4_0, QR4_0, QI4_0, block_q4_0, allocate_tiles_q4_0, load_tiles_q4_0<true>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q4_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK4_1, QR4_1, QI4_1, block_q4_1, allocate_tiles_q4_1, load_tiles_q4_1<false>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK4_1, QR4_1, QI4_1, block_q4_1, allocate_tiles_q4_1, load_tiles_q4_1<true>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK5_0, QR5_0, QI5_0, block_q5_0, allocate_tiles_q5_0, load_tiles_q5_0<false>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK5_0, QR5_0, QI5_0, block_q5_0, allocate_tiles_q5_0, load_tiles_q5_0<true>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK5_1, QR5_1, QI5_1, block_q5_1, allocate_tiles_q5_1, load_tiles_q5_1<false>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK5_1, QR5_1, QI5_1, block_q5_1, allocate_tiles_q5_1, load_tiles_q5_1<true>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q8_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK8_0, QR8_0, QI8_0, block_q8_0, allocate_tiles_q8_0, load_tiles_q8_0<false>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK8_0, QR8_0, QI8_0, block_q8_0, allocate_tiles_q8_0, load_tiles_q8_0<true>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q2_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK_K, QR2_K, QI2_K, block_q2_K, allocate_tiles_q2_K, load_tiles_q2_K<false>, VDR_q2_K_q8_1, vec_dot_q2_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK_K, QR2_K, QI2_K, block_q2_K, allocate_tiles_q2_K, load_tiles_q2_K<true>, VDR_q2_K_q8_1, vec_dot_q2_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q3_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK_K, QR3_K, QI3_K, block_q3_K, allocate_tiles_q3_K, load_tiles_q3_K<false>, VDR_q3_K_q8_1, vec_dot_q3_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK_K, QR3_K, QI3_K, block_q3_K, allocate_tiles_q3_K, load_tiles_q3_K<true>, VDR_q3_K_q8_1, vec_dot_q3_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q4_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK_K, QR4_K, QI4_K, block_q4_K, allocate_tiles_q4_K, load_tiles_q4_K<false>, VDR_q4_K_q8_1, vec_dot_q4_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK_K, QR4_K, QI4_K, block_q4_K, allocate_tiles_q4_K, load_tiles_q4_K<true>, VDR_q4_K_q8_1, vec_dot_q4_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK_K, QR5_K, QI5_K, block_q5_K, allocate_tiles_q5_K, load_tiles_q5_K<false>, VDR_q5_K_q8_1, vec_dot_q5_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK_K, QR5_K, QI5_K, block_q5_K, allocate_tiles_q5_K, load_tiles_q5_K<true>, VDR_q5_K_q8_1, vec_dot_q5_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q6_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    const int block_num_x = (nrows_x + GGML_CUDA_MMQ_Y - 1) / GGML_CUDA_MMQ_Y;
    const int block_num_y = (ncols_y + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, WARP_SIZE/4, 1);

    if (nrows_x % GGML_CUDA_MMQ_Y == 0) {
        mul_mat_q<QK_K, QR6_K, QI6_K, block_q6_K, allocate_tiles_q6_K, load_tiles_q6_K<false>, VDR_q6_K_q8_1, vec_dot_q6_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        mul_mat_q<QK_K, QR6_K, QI6_K, block_q6_K, allocate_tiles_q6_K, load_tiles_q6_K<true>, VDR_q6_K_q8_1, vec_dot_q6_K_q8_1_mul_mat>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_p021_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x,
    const int nchannels_x, const int nchannels_y, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_p021_f16_f32<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, nchannels_x, nchannels_y);
}

static void ggml_mul_mat_vec_nc_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int row_stride_x,
    const int nchannels_x, const int nchannels_y, const int channel_stride_x, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_vec_nc_f16_f32<<<block_nums, block_dims, 0, stream>>>
        (vx, y, dst, ncols_x, nrows_x, row_stride_x, channel_stride_x, nchannels_y/nchannels_x);
}

static void ggml_cpy_f32_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
    const int ne10, const int ne11, const int nb10, const int nb11, const int nb12, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12);
}

static void ggml_cpy_f32_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
    const int ne10, const int ne11, const int nb10, const int nb11, const int nb12, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12);
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, k);
}

static void rope_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float p0,
                          const float p_delta, const int p_delta_rows, const float theta_scale, cudaStream_t stream) {
    GGML_ASSERT(nrows % 2 == 0);
    const dim3 block_dims(2*CUDA_ROPE_BLOCK_SIZE, 1, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(num_blocks_x, nrows, 1);
    rope_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, p0, p_delta, p_delta_rows, theta_scale);
}

static void rope_glm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float p, const float block_p, const float theta_scale, cudaStream_t stream) {
    GGML_ASSERT(nrows % 4 == 0);
    const dim3 block_dims(4*CUDA_ROPE_BLOCK_SIZE, 1, 1);
    const int num_blocks_x = (ncols + 4*CUDA_ROPE_BLOCK_SIZE - 1) / (4*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(num_blocks_x, nrows, 1);
    rope_glm_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, p, block_p, theta_scale);
}

static void diag_mask_inf_f32_cuda(const float * x, float * dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream) {
    const dim3 block_dims(CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ncols_x + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, nrows_x, 1);
    diag_mask_inf_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x, rows_per_channel, n_past);
}

static void soft_max_f32_cuda(const float * x, float * dst, const int ncols_x, const int nrows_x, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(1, nrows_x, 1);
    soft_max_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x);
}

// buffer pool for cuda
#define MAX_CUDA_BUFFERS 256

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

struct cuda_buffer {
    void * ptr = nullptr;
    size_t size = 0;
};

static cuda_buffer g_cuda_buffer_pool[GGML_CUDA_MAX_DEVICES][MAX_CUDA_BUFFERS];
static std::atomic_flag g_cuda_pool_lock = ATOMIC_FLAG_INIT;

static void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
#ifdef DEBUG_CUDA_MALLOC
    int nnz = 0;
    size_t max_size = 0, tot_size = 0;
#endif
    size_t best_diff = 1ull << 36;
    int ibest = -1;
    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
            ++nnz;
            tot_size += b.size;
            if (b.size > max_size) max_size = b.size;
#endif
            if (b.size >= size) {
                size_t diff = b.size - size;
                if (diff < best_diff) {
                    best_diff = diff;
                    ibest = i;
                    if (!best_diff) {
                        void * ptr = b.ptr;
                        *actual_size = b.size;
                        b.ptr = nullptr;
                        b.size = 0;
                        return ptr;
                    }
                }
            }
        }
    }
    if (ibest >= 0) {
        cuda_buffer& b = g_cuda_buffer_pool[id][ibest];
        void * ptr = b.ptr;
        *actual_size = b.size;
        b.ptr = nullptr;
        b.size = 0;
        return ptr;
    }
#ifdef DEBUG_CUDA_MALLOC
    fprintf(stderr, "%s: %d buffers, max_size = %u MB, tot_size = %u MB, requested %u MB\n", __func__, nnz,
            (uint32_t)(max_size/1024/1024), (uint32_t)(tot_size/1024/1024), (uint32_t)(size/1024/1024));
#endif
    void * ptr;
    size_t look_ahead_size = (size_t) (1.05 * size);
    look_ahead_size = 256 * ((look_ahead_size + 255)/256);
    CUDA_CHECK(cudaMalloc((void **) &ptr, look_ahead_size));
    *actual_size = look_ahead_size;
    return ptr;
}

static void ggml_cuda_pool_free(void * ptr, size_t size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
    CUDA_CHECK(cudaFree(ptr));
}


static void * g_scratch_buffer = nullptr;
static size_t g_scratch_size = 1024*1024*1024; // 1 GB by default
static size_t g_scratch_offset = 0;

static int g_device_count = -1;
static int g_main_device = 0;
static int g_compute_capabilities[GGML_CUDA_MAX_DEVICES];
static float g_tensor_split[GGML_CUDA_MAX_DEVICES] = {0};
static bool g_mul_mat_q = false;

static cublasHandle_t g_cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

static cudaStream_t g_cudaStreams_main[GGML_CUDA_MAX_DEVICES] = { nullptr };

void ggml_init_cublas() {
    static bool initialized = false;

    if (!initialized) {
        CUDA_CHECK(cudaGetDeviceCount(&g_device_count));
        GGML_ASSERT(g_device_count <= GGML_CUDA_MAX_DEVICES);
        int64_t total_vram = 0;
        fprintf(stderr, "%s: found %d CUDA devices:\n", __func__, g_device_count);
        for (int id = 0; id < g_device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            fprintf(stderr, "  Device %d: %s, compute capability %d.%d\n", id, prop.name, prop.major, prop.minor);

            g_tensor_split[id] = total_vram;
            total_vram += prop.totalGlobalMem;

            g_compute_capabilities[id] = 100*prop.major + 10*prop.minor;
        }
        for (int id = 0; id < g_device_count; ++id) {
            g_tensor_split[id] /= total_vram;
        }

        for (int id = 0; id < g_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));

            // create main stream
            CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams_main[id], cudaStreamNonBlocking));

            // create cublas handle
            CUBLAS_CHECK(cublasCreate(&g_cublas_handles[id]));
            CUBLAS_CHECK(cublasSetMathMode(g_cublas_handles[id], CUBLAS_TF32_TENSOR_OP_MATH));
        }

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

        initialized = true;
    }
}

void ggml_cuda_set_tensor_split(const float * tensor_split) {
    if (tensor_split == nullptr) {
        return;
    }
    bool all_zero = true;
    for (int i = 0; i < g_device_count; ++i) {
        if (tensor_split[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return;
    }
    float split_sum = 0.0f;
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] = split_sum;
        split_sum += tensor_split[i];
    }
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] /= split_sum;
    }
}

void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // The allocation error can be bypassed. A null ptr will assigned out of this function.
        // This can fixed the OOM error in WSL.
        cudaGetLastError();
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory: %s\n",
            size/1024.0/1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

void ggml_cuda_host_free(void * ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    cudaMemcpyKind kind;
    char * src_ptr;
    if (src->backend == GGML_BACKEND_CPU) {
        kind = cudaMemcpyHostToDevice;
        src_ptr = (char *) src->data;
    } else if (src->backend == GGML_BACKEND_GPU) {
        kind = cudaMemcpyDeviceToDevice;
        struct ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        CUDA_CHECK(cudaGetDevice(&id));
        src_ptr = (char *) extra->data_device[id];
    } else {
        GGML_ASSERT(false);
    }
    char * dst_ptr = (char *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, kind, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, kind, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, kind, stream);
            if (r != cudaSuccess) return r;
        }
        return cudaSuccess;
    }
}

inline void ggml_cuda_op_add(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddq_i != nullptr || src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i  != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    // compute
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        add_f32_cuda(src0_ddf_i, src1_ddf_i, dst_ddf_i, ne00*i01_diff, ne10*ne11, cudaStream_main);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        add_f16_f32_f16_cuda((half *) src0_ddq_i, src1_ddf_i, (half *) dst_ddf_i, ne00*i01_diff, cudaStream_main);
    } else {
        GGML_ASSERT(false);
    }

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_mul(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i  != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    mul_f32_cuda(src0_ddf_i, src1_ddf_i, dst_ddf_i, ne00*i01_diff, ne10*ne11, cudaStream_main);

    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_gelu(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    gelu_f32_cuda(src0_ddf_i, dst_ddf_i, ne00*i01_diff, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_silu(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    silu_f32_cuda(src0_ddf_i, dst_ddf_i, ne00*i01_diff, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_norm(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    norm_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_rms_norm(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // compute
    rms_norm_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, eps, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_mul_mat_q(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddq_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t i01_diff = i01_high - i01_low;

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    const int64_t nrows_dst = dst->backend == GGML_BACKEND_GPU && id == g_main_device ? ne0 : i01_diff;

    const int64_t padded_row_size = ne10 % MATRIX_ROW_PADDING == 0 ?
        ne10 : ne10 - ne10 % MATRIX_ROW_PADDING + MATRIX_ROW_PADDING;
    size_t as;
    void * src1_q8_1 = ggml_cuda_pool_malloc(padded_row_size*ne11*sizeof(block_q8_1)/QK8_1, &as);
    quantize_row_q8_1_cuda(src1_ddf_i, src1_q8_1, ne10, ne11, padded_row_size, cudaStream_main);

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        case GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        case GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        case GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        case GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        case GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        case GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        case GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        case GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        case GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, i01_diff, ne11, padded_row_size, nrows_dst, cudaStream_main);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    ggml_cuda_pool_free(src1_q8_1, as);

    (void) src1;
    (void) dst;
    (void) src0_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_mul_mat_vec(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddq_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = i01_high - i01_low;

#ifdef GGML_CUDA_FORCE_DMMV
    const bool use_mul_mat_vec_q = false;
    (void) g_compute_capabilities[0];
#else
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    bool mul_mat_vec_q_implemented =
        src0->type == GGML_TYPE_Q4_0 ||
        src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 ||
        src0->type == GGML_TYPE_Q5_1 ||
        src0->type == GGML_TYPE_Q8_0;
#if QK_K == 256
    mul_mat_vec_q_implemented = mul_mat_vec_q_implemented ||
        src0->type == GGML_TYPE_Q2_K ||
        src0->type == GGML_TYPE_Q3_K ||
        src0->type == GGML_TYPE_Q4_K ||
        src0->type == GGML_TYPE_Q5_K ||
        src0->type == GGML_TYPE_Q6_K;
#endif // QK_K == 256

    const bool use_mul_mat_vec_q = g_compute_capabilities[id] >= MIN_CC_DP4A && mul_mat_vec_q_implemented;
#endif

    if (use_mul_mat_vec_q) {
        const int64_t padded_row_size = ne00 % MATRIX_ROW_PADDING == 0 ?
            ne00 : ne00 - ne00 % MATRIX_ROW_PADDING + MATRIX_ROW_PADDING;
        size_t as;
        void * src1_q8_1 = ggml_cuda_pool_malloc(padded_row_size*sizeof(block_q8_1)/QK8_1, &as);
        quantize_row_q8_1_cuda(src1_ddf_i, src1_q8_1, ne00, 1, padded_row_size, cudaStream_main);

        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                mul_mat_vec_q4_0_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q4_1:
                mul_mat_vec_q4_1_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q5_0:
                mul_mat_vec_q5_0_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q5_1:
                mul_mat_vec_q5_1_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q8_0:
                mul_mat_vec_q8_0_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q2_K:
                mul_mat_vec_q2_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q3_K:
                mul_mat_vec_q3_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q4_K:
                mul_mat_vec_q4_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q5_K:
                mul_mat_vec_q5_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q6_K:
                mul_mat_vec_q6_K_q8_1_cuda(src0_ddq_i, src1_q8_1, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            default:
                GGML_ASSERT(false);
                break;
        }

        ggml_cuda_pool_free(src1_q8_1, as);
    } else {
        // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_CUDA_F16
        size_t ash;
        dfloat * src1_dfloat = nullptr; // dfloat == half

        bool src1_convert_f16 = src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
            src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
            src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_F16;

        if (src1_convert_f16) {
            src1_dfloat = (half *) ggml_cuda_pool_malloc(ne00*sizeof(half), &ash);
            ggml_cpy_f32_f16_cuda((char *) src1_ddf_i, (char *) src1_dfloat, ne00,
                                    ne00, 1, sizeof(float), 0, 0,
                                    ne00, 1, sizeof(half),  0, 0, cudaStream_main);
        }
#else
        dfloat * src1_dfloat = src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_CUDA_F16

        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                dequantize_mul_mat_vec_q4_0_cuda(src0_ddq_i, src1_dfloat, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q4_1:
                dequantize_mul_mat_vec_q4_1_cuda(src0_ddq_i, src1_dfloat, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q5_0:
                dequantize_mul_mat_vec_q5_0_cuda(src0_ddq_i, src1_dfloat, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q5_1:
                dequantize_mul_mat_vec_q5_1_cuda(src0_ddq_i, src1_dfloat, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q8_0:
                dequantize_mul_mat_vec_q8_0_cuda(src0_ddq_i, src1_dfloat, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q2_K:
                dequantize_mul_mat_vec_q2_K_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q3_K:
                dequantize_mul_mat_vec_q3_K_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q4_K:
                dequantize_mul_mat_vec_q4_K_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q5_K:
                dequantize_mul_mat_vec_q5_K_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q6_K:
                dequantize_mul_mat_vec_q6_K_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_F16:
                convert_mul_mat_vec_f16_cuda(src0_ddq_i, src1_dfloat, dst_ddf_i, ne00, nrows, cudaStream_main);
                break;
            default:
                GGML_ASSERT(false);
                break;
        }

#ifdef GGML_CUDA_F16
        if (src1_convert_f16) {
            ggml_cuda_pool_free(src1_dfloat, ash);
        }
#endif // GGML_CUDA_F16
    }

    (void) src1;
    (void) dst;
    (void) src0_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_mul_mat_cublas(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int64_t ne0 = dst->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = dst->backend == GGML_BACKEND_GPU && id == g_main_device ? ne0 : i01_diff;

    CUBLAS_CHECK(cublasSetStream(g_cublas_handles[id], cudaStream_main));
    CUBLAS_CHECK(
        cublasSgemm(g_cublas_handles[id], CUBLAS_OP_T, CUBLAS_OP_N,
                i01_diff, ne11, ne10,
                &alpha, src0_ddf_i, ne00,
                        src1_ddf_i, ne10,
                &beta,  dst_ddf_i,  ldc));

    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_rope(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t i01_diff = i01_high - i01_low;

    const int n_past = ((int32_t *) dst->op_params)[0];
    const int n_dims = ((int32_t *) dst->op_params)[1];
    const int mode   = ((int32_t *) dst->op_params)[2];
    const int n_ctx  = ((int32_t *) dst->op_params)[3];
    // RoPE alteration for extended context

    float freq_base, freq_scale;
    memcpy(&freq_base,  (int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 5, sizeof(float));

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const bool is_glm = mode & 4;

    // compute
    if (is_glm) {
        const float p = (((mode & 1) == 0 ? n_past + i02 : i02)) * freq_scale;
        const float id_p = min(p, n_ctx - 2.f);
        const float block_p = max(p - (n_ctx - 2.f), 0.f);
        rope_glm_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, id_p, block_p, theta_scale, cudaStream_main);
    } else {
        const float p0 = (((mode & 1) == 0 ? n_past : 0)) * freq_scale;
        rope_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, p0, freq_scale, ne01, theta_scale, cudaStream_main);
    }

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i1;
}

inline void ggml_cuda_op_diag_mask_inf(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t i01_diff = i01_high - i01_low;

    const int n_past = ((int32_t *) dst->op_params)[0];

    // compute
    diag_mask_inf_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, ne01, n_past, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_soft_max(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    soft_max_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_scale(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const float scale = ((float *) src1->data)[0];

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    scale_f32_cuda(src0_ddf_i, dst_ddf_i, scale, ne00*i01_diff, cudaStream_main);
    CUDA_CHECK(cudaGetLastError());

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

static void ggml_cuda_op(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                         ggml_cuda_op_t op, bool src0_needs_f32, bool flatten_rows) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t nrows0 = ggml_nrows(src0);

    const bool use_src1 = src1 != nullptr;
    const int64_t ne10 = use_src1 ? src1->ne[0] : 1;
    const int64_t ne11 = use_src1 ? src1->ne[1] : 1;
    const int64_t ne12 = use_src1 ? src1->ne[2] : 1;
    const int64_t ne13 = use_src1 ? src1->ne[3] : 1;
    const int64_t nrows1 = use_src1 ? ggml_nrows(src1) : 1;

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    GGML_ASSERT(dst->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(!use_src1 || src1->backend != GGML_BACKEND_GPU_SPLIT);

    // strides for iteration over dims 3 and 2
    const int64_t num_iters_0 = ne02 >= ne12 ? ne02*ne03 : ne12*ne13;
    const int64_t num_iters = flatten_rows ? 1 : num_iters_0;
    const int64_t stride_mod = flatten_rows ? num_iters_0 : 1;
    const int64_t src0_stride = ne00 * ne01 * stride_mod;
    const int64_t src1_stride = ne10 * ne11 * stride_mod;
    const int64_t dst_stride = ne0 * ne1 * stride_mod;

    const int64_t rows_per_iter = flatten_rows ? nrows0 : ne01;
    const int64_t i03_max = flatten_rows ? 1 : ne03;
    const int64_t i02_max = flatten_rows ? 1 : (ne02 >= ne12 ? ne02 : ne12);
    const int64_t i02_divisor = ne02 >= ne12 ? 1 : ne12 / ne02;
    GGML_ASSERT(!(flatten_rows && ne02 < ne12));

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);

    struct ggml_tensor_extra_gpu * src0_extra =            (ggml_tensor_extra_gpu *) src0->extra;
    struct ggml_tensor_extra_gpu * src1_extra = use_src1 ? (ggml_tensor_extra_gpu *) src1->extra : nullptr;
    struct ggml_tensor_extra_gpu * dst_extra  =            (ggml_tensor_extra_gpu *) dst->extra;

    const bool src0_on_device = src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT;
    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src0_is_f32 = src0->type == GGML_TYPE_F32;

    const bool src1_is_contiguous = use_src1 && ggml_is_contiguous(src1);
    const bool src1_stays_on_host = use_src1 && (
        dst->op == GGML_OP_SCALE || dst->op == GGML_OP_DIAG_MASK_INF || dst->op == GGML_OP_ROPE);

    const bool split = src0->backend == GGML_BACKEND_GPU_SPLIT;
    GGML_ASSERT(!(split && ne02 < ne12));

    const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);

    // dd = data device
    char  * src0_ddq[GGML_CUDA_MAX_DEVICES] = {nullptr}; // quantized
    float * src0_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr}; // float
    float * src1_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr};
    float *  dst_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr};

    // asq = actual size quantized, asf = actual size float
    size_t src0_asq[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src0_asf[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src1_asf[GGML_CUDA_MAX_DEVICES] = {0};
    size_t  dst_asf[GGML_CUDA_MAX_DEVICES] = {0};

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signifies that the main device has finished calculating the input data
    if (split && g_device_count > 1) {
        CUDA_CHECK(cudaSetDevice(g_main_device));
        CUDA_CHECK(cudaEventRecord(src0_extra->events[g_main_device], g_cudaStreams_main[g_main_device]));
    }

    for (int id = 0; id < g_device_count; ++id) {
        if (!split && id != g_main_device) {
            continue;
        }

        const bool src1_on_device = use_src1 && src1->backend == GGML_BACKEND_GPU && id == g_main_device;
        const bool dst_on_device = dst->backend == GGML_BACKEND_GPU && id == g_main_device;

        int64_t row_low, row_high;
        if (split) {
            row_low = id == 0 ? 0 : nrows0*g_tensor_split[id];
            row_low -= row_low % GGML_CUDA_MMQ_Y;

            if (id == g_device_count - 1) {
                row_high = nrows0;
            } else {
                row_high = nrows0*g_tensor_split[id + 1];
                row_high -= row_high % GGML_CUDA_MMQ_Y;
            }
        } else {
            row_low = 0;
            row_high = nrows0*i02_divisor;
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t row_diff = row_high - row_low;

        cudaSetDevice(id);
        cudaStream_t cudaStream_main = g_cudaStreams_main[id];

        // wait for main GPU data if necessary
        if (split && id != g_main_device) {
            CUDA_CHECK(cudaStreamWaitEvent(cudaStream_main, src0_extra->events[g_main_device]));
        }

        if (src0_on_device && src0_is_contiguous) {
            if (src0_is_f32) {
                src0_ddf[id] = (float *) src0_extra->data_device[id];
            } else {
                src0_ddq[id] = (char *) src0_extra->data_device[id];
            }
        } else {
            if (src0_is_f32) {
                src0_ddf[id] = (float *) ggml_cuda_pool_malloc(row_diff*ne00 * sizeof(float), &src0_asf[id]);
            } else {
                src0_ddq[id] = (char *) ggml_cuda_pool_malloc(row_diff*ne00 * src0_ts/src0_bs, &src0_asq[id]);
            }
        }

        if (src0_needs_f32 && !src0_is_f32) {
            src0_ddf[id] = (float *) ggml_cuda_pool_malloc(row_diff*ne00 * sizeof(float), &src0_asf[id]);
        }

        if (use_src1 && !src1_stays_on_host) {
            if (src1_on_device && src1_is_contiguous) {
                src1_ddf[id] = (float *) src1_extra->data_device[id];
            } else {
                src1_ddf[id] = (float *) ggml_cuda_pool_malloc(num_iters*src1_stride * sizeof(float), &src1_asf[id]);
            }
        }
        if (dst_on_device) {
            dst_ddf[id] = (float *) dst_extra->data_device[id];
        } else {
            size_t size_dst_ddf = split ? row_diff*ne1 * sizeof(float) : num_iters*dst_stride * sizeof(float);
            dst_ddf[id] = (float *) ggml_cuda_pool_malloc(size_dst_ddf, &dst_asf[id]);
        }

        for (int64_t i03 = 0; i03 < i03_max; i03++) {
            const int64_t i13 = i03 % ne13;
            for (int64_t i02 = 0; i02 < i02_max; i02++) {
                const int64_t i12 = i02 % ne12;

                const int64_t i0 = i03*i02_max + i02;

                // i0 values that contain the lower/upper rows for a split tensor when using multiple GPUs
                const int64_t i0_offset_low = row_low/rows_per_iter;
                const int64_t i0_offset_high = row_high/rows_per_iter;

                int64_t i01_low = 0;
                int64_t i01_high = rows_per_iter;
                if (split) {
                    if (i0 < i0_offset_low || i0 > i0_offset_high) {
                        continue;
                    }
                    if (i0 == i0_offset_low) {
                        i01_low = row_low % rows_per_iter;
                    }
                    if (i0 == i0_offset_high) {
                        i01_high = row_high % rows_per_iter;
                    }
                }

                // There is possibly a bug in the Windows nvcc compiler regarding instruction reordering or optimizing out local variables.
                // Removing the first assert or changing the order of the arguments causes the second assert to fail.
                // Removing both asserts results in i01_high becoming 0 which in turn results in garbage output.
                // The root cause seems to be a problem with i0_offset_high becoming 0 when it should always be >0 (for single GPU).
                GGML_ASSERT(i01_low == 0 || g_device_count > 1);
                GGML_ASSERT(i01_high == rows_per_iter || g_device_count > 1);

                const int64_t i01_diff = i01_high - i01_low;
                if (i01_diff == 0) {
                    continue;
                }
                const int64_t i11 = i13*ne12 + i12;

                // for split tensors the data begins at i0 == i0_offset_low
                char  * src0_ddq_i = src0_ddq[id] + (i0/i02_divisor - i0_offset_low)*src0_stride*src0_ts/src0_bs;
                float * src0_ddf_i = src0_ddf[id] + (i0/i02_divisor - i0_offset_low)*src0_stride;
                float * src1_ddf_i = src1_ddf[id] + i11*src1_stride;
                float * dst_ddf_i  =  dst_ddf[id] + (i0             - i0_offset_low)*dst_stride;

                // for split tensors the data pointer needs to be rounded down
                // to the bin edge for i03, i02 bins beyond the first
                if (i0 - i0_offset_low > 0) {
                    GGML_ASSERT(!flatten_rows);
                    src0_ddq_i -= (row_low % ne01)*ne00 * src0_ts/src0_bs;
                    src0_ddf_i -= (row_low % ne01)*ne00;
                    dst_ddf_i  -= (row_low % ne0)*ne1;
                }

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (dst->backend == GGML_BACKEND_GPU && id == g_main_device) {
                    dst_ddf_i += i01_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (use_src1 && !src1_stays_on_host) {
                    if (src1->backend == GGML_BACKEND_CPU) {
                        GGML_ASSERT(!flatten_rows || nrows0 == ggml_nrows(src1));
                        int64_t nrows1 = flatten_rows ? nrows0 : ne11;
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, 0, nrows1, cudaStream_main));
                    } else if (src1->backend == GGML_BACKEND_GPU && src1_is_contiguous) {
                        if (id != g_main_device) {
                            GGML_ASSERT(!flatten_rows);
                            float * src1_ddf_i_source = (float *) src1_extra->data_device[g_main_device];
                            src1_ddf_i_source += i11*src1_stride;
                            CUDA_CHECK(cudaMemcpyAsync(src1_ddf_i, src1_ddf_i_source, src1_stride*sizeof(float),
                                                    cudaMemcpyDeviceToDevice, cudaStream_main));
                        }
                    } else if (src1_on_device && !src1_is_contiguous) {
                        GGML_ASSERT(!split);
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, 0, ne11, cudaStream_main));
                    } else {
                        GGML_ASSERT(false);
                    }
                }

                if ((!src0_on_device || !src0_is_contiguous) && i02 % i02_divisor == 0) {
                    if (src0_is_f32) {
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_ddf_i, src0, i03, i02/i02_divisor, i01_low, i01_high, cudaStream_main));
                    } else {
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_ddq_i, src0, i03, i02/i02_divisor, i01_low, i01_high, cudaStream_main));
                    }
                }

                // convert src0 to f32 if it is necessary for the ggml_cuda_op
                if (src0_needs_f32 && !src0_is_f32) {
                    to_fp32_cuda(src0_ddq_i, src0_ddf_i, i01_diff*ne00, cudaStream_main);
                    CUDA_CHECK(cudaGetLastError());
                }

                // do the computation
                op(src0, src1, dst, src0_ddq_i, src0_ddf_i, src1_ddf_i, dst_ddf_i, i02, i01_low, i01_high, i11, cudaStream_main);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device;
                    cudaMemcpyKind kind;
                    if (dst->backend == GGML_BACKEND_CPU) {
                        dst_off_device = dst->data;
                        kind = cudaMemcpyDeviceToHost;
                    } else if (dst->backend == GGML_BACKEND_GPU) {
                        dst_off_device = dst_extra->data_device[g_main_device];
                        kind = cudaMemcpyDeviceToDevice;
                    } else {
                        GGML_ASSERT(false);
                    }
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i01_low*sizeof(float) + i02*nb2 + i03*nb3);
                        CUDA_CHECK(cudaMemcpy2DAsync(dhf_dst_i, ne0*sizeof(float), dst_ddf_i, i01_diff*sizeof(float),
                                                     i01_diff*sizeof(float), ne1, kind, cudaStream_main));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_ddf_i, dst_stride*sizeof(float), kind, cudaStream_main));
                    }
                }

                // signify to main device that other device is done
                if (split && g_device_count > 1 && id != g_main_device) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id], cudaStream_main));
                }
            }
        }
    }

    // wait until each device is finished, then free their buffers
    for (int id = 0; id < g_device_count; ++id) {
        if (src0_asq[id] == 0 && src0_asf[id] == 0 && src1_asf[id] == 0 && dst_asf[id] == 0) {
            continue;
        }

        CUDA_CHECK(cudaSetDevice(id));

        if (src0_asq[id] > 0) {
            ggml_cuda_pool_free(src0_ddq[id], src0_asq[id]);
        }
        if (src0_asf[id] > 0) {
            ggml_cuda_pool_free(src0_ddf[id], src0_asf[id]);
        }
        if (src1_asf[id] > 0) {
            ggml_cuda_pool_free(src1_ddf[id], src1_asf[id]);
        }
        if (dst_asf[id] > 0) {
            ggml_cuda_pool_free(dst_ddf[id], dst_asf[id]);
        }
    }

    // main device waits for all other devices to be finished
    if (split && g_device_count > 1) {
        CUDA_CHECK(cudaSetDevice(g_main_device));
        for (int id = 0; id < g_device_count; ++id) {
            if (id != g_main_device) {
                CUDA_CHECK(cudaStreamWaitEvent(g_cudaStreams_main[g_main_device], src0_extra->events[id]));
            }
        }
    }

    if (dst->backend == GGML_BACKEND_CPU) {
        CUDA_CHECK(cudaSetDevice(g_main_device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void ggml_cuda_add(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // ggml_cuda_add permits f16 dst even though this could in theory cause problems with the pointer arithmetic in ggml_cuda_op.
    // Due to flatten_rows == true this does in practice not make a difference however.
    // Better solution would be nice but right now that would require disproportionate changes.
    GGML_ASSERT(
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
        src1->type == GGML_TYPE_F32 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16));
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_add, false, true);
}

void ggml_cuda_mul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul, true, false); // TODO ggml_cuda_op needs modification for flatten
}

void ggml_cuda_gelu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_gelu, true, true);
}

void ggml_cuda_silu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_silu, true, true);
}

void ggml_cuda_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_norm, true, true);
}

void ggml_cuda_rms_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_rms_norm, true, true);
}

bool ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        (ne0 >= 32 && ne1 >= 32 && ne10 >= 32)) {
        return true;
    }

    return false;
}

void ggml_cuda_mul_mat_vec_p021(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    CUDA_CHECK(cudaSetDevice(g_main_device));
    cudaStream_t cudaStream_main = g_cudaStreams_main[g_main_device];

    struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    void * src0_ddq = src0_extra->data_device[g_main_device];

    struct ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_main_device];

    struct ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_main_device];

    ggml_mul_mat_p021_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, cudaStream_main);
}

void ggml_cuda_mul_mat_vec_nc(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    GGML_ASSERT(!ggml_is_contiguous(src0) && ggml_is_contiguous(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    CUDA_CHECK(cudaSetDevice(g_main_device));
    cudaStream_t cudaStream_main = g_cudaStreams_main[g_main_device];

    struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    void * src0_ddq = src0_extra->data_device[g_main_device];

    struct ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_main_device];

    struct ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_main_device];

    const int row_stride_x = nb01 / sizeof(half);
    const int channel_stride_x = nb02 / sizeof(half);

    ggml_mul_mat_vec_nc_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, cudaStream_main);
}

void ggml_cuda_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    bool all_on_device = (src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT) &&
        src1->backend == GGML_BACKEND_GPU && dst->backend == GGML_BACKEND_GPU;

    if (all_on_device && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_p021(src0, src1, dst);
    } else if (all_on_device && !ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_nc(src0, src1, dst);
    }else if (src0->type == GGML_TYPE_F32) {
        ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul_mat_cublas, true, false);
    } else if (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F16) {
        if (src1->ne[1] == 1 && src0->ne[0] % GGML_CUDA_DMMV_X == 0) {
            ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul_mat_vec, false, false);
        } else {
            int min_compute_capability = INT_MAX;
            for (int id = 0; id < g_device_count; ++id) {
                if (min_compute_capability > g_compute_capabilities[id]) {
                    min_compute_capability = g_compute_capabilities[id];
                }
            }

            if (g_mul_mat_q && ggml_is_quantized(src0->type) && min_compute_capability >= MIN_CC_DP4A) {
                ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul_mat_q, false, false);
            } else {
                ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul_mat_cublas, true, false);
            }
        }
    } else {
        GGML_ASSERT(false);
    }
}

void ggml_cuda_scale(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_scale, true, true);
}

void ggml_cuda_cpy(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_ASSERT(src0->backend == GGML_BACKEND_GPU);
    GGML_ASSERT(src1->backend == GGML_BACKEND_GPU);

    GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX);
    GGML_ASSERT(ggml_nbytes(src1) <= INT_MAX);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];

    CUDA_CHECK(cudaSetDevice(g_main_device));
    cudaStream_t cudaStream_main = g_cudaStreams_main[g_main_device];

    const struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    const struct ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;

    char * src0_ddc = (char *) src0_extra->data_device[g_main_device];
    char * src1_ddc = (char *) src1_extra->data_device[g_main_device];

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, cudaStream_main);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, cudaStream_main);
    } else {
        GGML_ASSERT(false);
    }

    (void) dst;
}

void ggml_cuda_dup(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_cpy(src0, dst, nullptr);
    (void) src1;
}

void ggml_cuda_diag_mask_inf(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_diag_mask_inf, true, true);
}

void ggml_cuda_soft_max(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_soft_max, true, true);
}

void ggml_cuda_rope(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int mode = ((int32_t *) dst->op_params)[2];
    const bool is_glm = mode & 4;
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_rope, true, !is_glm); // flatten support not implemented for glm
}

void ggml_cuda_nop(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    (void) src0;
    (void) src1;
    (void) dst;
}

void ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor) {
    int nrows = ggml_nrows(tensor);

    const int64_t ne0 = tensor->ne[0];

    const size_t nb1 = tensor->nb[1];

    ggml_backend backend = tensor->backend;
    struct ggml_tensor_extra_gpu * extra = new struct ggml_tensor_extra_gpu;
    memset(extra, 0, sizeof(*extra));

    for (int id = 0; id < g_device_count; ++id) {
        if (backend == GGML_BACKEND_GPU && id != g_main_device) {
            continue;
        }

        cudaSetDevice(id);

        int row_low, row_high;
        if (backend == GGML_BACKEND_GPU) {
            row_low = 0;
            row_high = nrows;
        } else if (backend == GGML_BACKEND_GPU_SPLIT) {
            row_low = id == 0 ? 0 : nrows*g_tensor_split[id];
            row_low -= row_low % GGML_CUDA_MMQ_Y;

            if (id == g_device_count - 1) {
                row_high = nrows;
            } else {
                row_high = nrows*g_tensor_split[id + 1];
                row_high -= row_high % GGML_CUDA_MMQ_Y;
            }
        } else {
            GGML_ASSERT(false);
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t nrows_split = row_high - row_low;

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += (MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)
                * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
        }

        char * buf;
        CUDA_CHECK(cudaMalloc(&buf, size));
        char * buf_host = (char*)data + offset_split;

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
        }


        CUDA_CHECK(cudaMemcpy(buf, buf_host, original_size, cudaMemcpyHostToDevice));

        extra->data_device[id] = buf;

        if (backend == GGML_BACKEND_GPU_SPLIT) {
            CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id], cudaEventDisableTiming));
        }
    }

    tensor->extra = extra;
}

void ggml_cuda_free_data(struct ggml_tensor * tensor) {
    if (!tensor || (tensor->backend != GGML_BACKEND_GPU && tensor->backend != GGML_BACKEND_GPU_SPLIT) ) {
        return;
    }

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int id = 0; id < g_device_count; ++id) {
        if (extra->data_device[id] != nullptr) {
            CUDA_CHECK(cudaSetDevice(id));
            CUDA_CHECK(cudaFree(extra->data_device[id]));
        }

        if (extra->events[id] != nullptr) {
            CUDA_CHECK(cudaSetDevice(id));
            CUDA_CHECK(cudaEventDestroy(extra->events[id]));
        }
    }

    delete extra;
}

static struct ggml_tensor_extra_gpu * g_temp_tensor_extras = nullptr;
static size_t g_temp_tensor_extra_index = 0;

static struct ggml_tensor_extra_gpu * ggml_cuda_alloc_temp_tensor_extra() {
    if (g_temp_tensor_extras == nullptr) {
        g_temp_tensor_extras = new ggml_tensor_extra_gpu[GGML_MAX_NODES];
    }

    size_t alloc_index = g_temp_tensor_extra_index;
    g_temp_tensor_extra_index = (g_temp_tensor_extra_index + 1) % GGML_MAX_NODES;
    struct ggml_tensor_extra_gpu * extra = &g_temp_tensor_extras[alloc_index];
    memset(extra, 0, sizeof(*extra));

    return extra;
}

void ggml_cuda_assign_buffers_impl(struct ggml_tensor * tensor, bool scratch, bool force_inplace) {
    if (scratch && g_scratch_size == 0) {
        return;
    }

    // recursively assign CUDA buffers until a compute tensor is found
    if (tensor->src[0] != nullptr && tensor->src[0]->backend == GGML_BACKEND_CPU) {
        const ggml_op src0_op = tensor->src[0]->op;
        if (src0_op == GGML_OP_RESHAPE || src0_op == GGML_OP_TRANSPOSE || src0_op == GGML_OP_VIEW || src0_op == GGML_OP_PERMUTE) {
            ggml_cuda_assign_buffers_impl(tensor->src[0], scratch, force_inplace);
        }
    }
    if (tensor->op == GGML_OP_CPY && tensor->src[1]->backend == GGML_BACKEND_CPU) {
        ggml_cuda_assign_buffers_impl(tensor->src[1], scratch, force_inplace);
    }

    tensor->backend = GGML_BACKEND_GPU;
    struct ggml_tensor_extra_gpu * extra;

    const bool inplace = (tensor->src[0] != nullptr && tensor->src[0]->data == tensor->data) ||
        tensor->op == GGML_OP_VIEW ||
        force_inplace;
    const size_t size = ggml_nbytes(tensor);

    CUDA_CHECK(cudaSetDevice(g_main_device));
    if (inplace && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT)) {
        struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu * ) tensor->src[0]->extra;
        char * src0_ddc = (char *) src0_extra->data_device[g_main_device];
        size_t offset = 0;
        if (tensor->op == GGML_OP_VIEW) {
            memcpy(&offset, tensor->op_params, sizeof(size_t));
        }
        extra = ggml_cuda_alloc_temp_tensor_extra();
        extra->data_device[g_main_device] = src0_ddc + offset;
    } else if (tensor->op == GGML_OP_CPY) {
        struct ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu * ) tensor->src[1]->extra;
        void * src1_ddv = src1_extra->data_device[g_main_device];
        extra = ggml_cuda_alloc_temp_tensor_extra();
        extra->data_device[g_main_device] = src1_ddv;
    } else if (scratch) {
        GGML_ASSERT(size <= g_scratch_size);
        if (g_scratch_offset + size > g_scratch_size) {
            g_scratch_offset = 0;
        }

        char * data = (char *) g_scratch_buffer;
        if (data == nullptr) {
            CUDA_CHECK(cudaMalloc(&data, g_scratch_size));
            g_scratch_buffer = data;
        }
        extra = ggml_cuda_alloc_temp_tensor_extra();
        extra->data_device[g_main_device] = data + g_scratch_offset;

        g_scratch_offset += size;

        GGML_ASSERT(g_scratch_offset <= g_scratch_size);
    } else { // allocate new buffers outside of scratch
        void * data;
        CUDA_CHECK(cudaMalloc(&data, size));
        CUDA_CHECK(cudaMemset(data, 0, size));
        extra = new ggml_tensor_extra_gpu;
        memset(extra, 0, sizeof(*extra));
        extra->data_device[g_main_device] = data;
    }

    tensor->extra = extra;
}

void ggml_cuda_assign_buffers(struct ggml_tensor * tensor) {
    ggml_cuda_assign_buffers_impl(tensor, true, false);
}

void ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor * tensor) {
    ggml_cuda_assign_buffers_impl(tensor, false, false);
}

void ggml_cuda_assign_buffers_force_inplace(struct ggml_tensor * tensor) {
    ggml_cuda_assign_buffers_impl(tensor, false, true);
}

void ggml_cuda_set_main_device(int main_device) {
    if (main_device >= g_device_count) {
        fprintf(stderr, "warning: cannot set main_device=%d because there are only %d devices. Using device %d instead.\n",
                main_device, g_device_count, g_main_device);
        return;
    }
    g_main_device = main_device;
    if (g_device_count > 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, g_main_device));
        fprintf(stderr, "%s: using device %d (%s) as main device\n", __func__, g_main_device, prop.name);
    }
}

void ggml_cuda_set_mul_mat_q(bool mul_mat_q) {
    g_mul_mat_q = mul_mat_q;
}

void ggml_cuda_set_scratch_size(size_t scratch_size) {
    g_scratch_size = scratch_size;
}

void ggml_cuda_free_scratch() {
    if (g_scratch_buffer == nullptr) {
        return;
    }

    CUDA_CHECK(cudaFree(g_scratch_buffer));
    g_scratch_buffer = nullptr;
}

bool ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor){
    ggml_cuda_func_t func;
    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU
        || (tensor->src[0] != nullptr && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (tensor->src[1] != nullptr && tensor->src[1]->backend == GGML_BACKEND_GPU);

    switch (tensor->op) {
        case GGML_OP_DUP:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_dup;
            break;
        case GGML_OP_ADD:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_add;
            break;
        case GGML_OP_MUL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_mul;
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cuda_gelu;
                    break;
                case GGML_UNARY_OP_SILU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cuda_silu;
                    break;
                default:
                    return false;
            } break;
        case GGML_OP_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_norm;
            break;
        case GGML_OP_RMS_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_rms_norm;
            break;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cuda_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cuda_mul_mat;
            break;
        case GGML_OP_SCALE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_scale;
            break;
        case GGML_OP_CPY:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_cpy;
            break;
        case GGML_OP_CONT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_dup;
            break;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_nop;
            break;
        case GGML_OP_DIAG_MASK_INF:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_diag_mask_inf;
            break;
        case GGML_OP_SOFT_MAX:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_soft_max;
            break;
        case GGML_OP_ROPE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_rope;
            break;
        default:
            return false;
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }
    func(tensor->src[0], tensor->src[1], tensor);
    return true;
}
