/**
 * llama.cpp - git 465219b9143ac01db0990bbcb0a081ef72ec2008
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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <atomic>
#include <assert.h>

#if defined(GGML_USE_HIPBLAS)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#ifdef __HIP_PLATFORM_AMD__
// for rocblas_initialize()
#include "rocblas/rocblas.h"
#endif // __HIP_PLATFORM_AMD__
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUDA_R_16F  HIPBLAS_R_16F
#define CUDA_R_32F  HIPBLAS_R_32F
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define cublasCreate hipblasCreate
#define cublasGemmEx hipblasGemmEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasStatus_t hipblasStatus_t
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEvent_t hipEvent_t
#define cudaEventDestroy hipEventDestroy
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaMemcpy hipMemcpy
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaSetDevice hipSetDevice
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent(stream, event, flags) hipStreamWaitEvent(stream, event, flags)
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#endif // defined(GGML_USE_HIPBLAS)

#include "ggml-cuda.h"
#include "ggml.h"

#define MIN_CC_DP4A   610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define CC_VOLTA      700
#define CC_OFFSET_AMD 1000000
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)

#if defined(GGML_USE_HIPBLAS)
#define __CUDA_ARCH__ 1300

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define RDNA3
#endif

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2
#endif

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int&>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        if(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    return reinterpret_cast<int&>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)
}

static __device__ __forceinline__ int __dp4a(const int a, const int b, int c) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx1030__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(__gfx1100__)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(__gfx1010__) || defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;
}
#endif // defined(GGML_USE_HIPBLAS)

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            int id;                                                                     \
            cudaGetDevice(&id);                                                         \
            fprintf(stderr, "\nCUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, \
                cudaGetErrorString(err_));                                              \
            fprintf(stderr, "current device: %d\n", id);                                \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#if CUDART_VERSION >= 12000
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            int id;                                                                     \
            cudaGetDevice(&id);                                                         \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d: %s\n",                         \
                    err_, __FILE__, __LINE__, cublasGetStatusString(err_));             \
            fprintf(stderr, "current device: %d\n", id);                                \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#else
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            int id;                                                                     \
            cudaGetDevice(&id);                                                         \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
            fprintf(stderr, "current device: %d\n", id);                                \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#endif // CUDART_VERSION >= 11

#if CUDART_VERSION >= 11100
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif // CUDART_VERSION >= 11100

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

template<typename T>
using to_t_cuda_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int k, cudaStream_t stream);
typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;

typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);
typedef void (*dot_kernel_k_t)(const void * __restrict__ vx, const int ib, const int iqs, const float * __restrict__ y, float & v);
typedef void (*cpy_kernel_t)(const char * cx, char * cdst);
typedef void (*ggml_cuda_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
typedef void (*ggml_cuda_op_mul_mat_t)(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, const cudaStream_t & stream);
typedef void (*ggml_cuda_op_flatten_t)(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream);

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
    half    dm[2];             // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == sizeof(half2) + QK_K/2 + 2, "wrong q4_K block size/padding");
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
#define CUDA_CLAMP_BLOCK_SIZE 256
#define CUDA_ROPE_BLOCK_SIZE 256
#define CUDA_ALIBI_BLOCK_SIZE 32
#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BLOCK_SIZE 256

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

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

#define MAX_STREAMS 8
static cudaStream_t g_cudaStreams[GGML_CUDA_MAX_DEVICES][MAX_STREAMS] = { nullptr };

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][MAX_STREAMS]; // events for synchronizing multiple GPUs
};

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
inline cudaError_t ggml_cuda_set_device(const int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return cudaSuccess;
    }

    return cudaSetDevice(device);
}

static int g_device_count = -1;
static int g_main_device = 0;
static int g_compute_capabilities[GGML_CUDA_MAX_DEVICES];
static float g_tensor_split[GGML_CUDA_MAX_DEVICES] = {0};
static bool g_mul_mat_q = true;

static void * g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

static cublasHandle_t g_cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

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

static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
        a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
    }
    return a;
}

template <int block_size>
static __global__ void norm_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const float eps = 1e-5f;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = (x[row*ncols + col] - mean) * inv_std;
    }
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

template <int block_size>
static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
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

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

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

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

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

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int tid = threadIdx.x;
#if QK_K == 256
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#else
    const int is = tid/16;  // 0 or 1
    const int il = tid%16;  // 0...15
    const uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;
    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

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

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#else
    const int tid = threadIdx.x;
    const int is  = tid/16;  // 0 or 1
    const int il  = tid%16;  // 0...15
    const int im  = il/8;    // 0...1
    const int in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

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

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int i = blockIdx.x;

#if QK_K == 256
    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

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
    dst_t * y = yy + i*QK_K;
    const float d = (float)x[i].dm[0];
    const float m = (float)x[i].dm[1];
    y[tid+ 0] = d * (x[i].scales[0] & 0xF) * (q[tid] & 0xF) - m * (x[i].scales[0] >> 4);
    y[tid+32] = d * (x[i].scales[1] & 0xF) * (q[tid] >>  4) - m * (x[i].scales[1] >> 4);
#endif
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int i = blockIdx.x;

#if QK_K == 256
    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

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
    dst_t * y = yy + i*QK_K + tid;
    y[ 0] = d * x[i].scales[is+0] * ((q & 0xF) - ((h >> 0) & 1 ? 0 : 16));
    y[32] = d * x[i].scales[is+2] * ((q >>  4) - ((h >> 4) & 1 ? 0 : 16));
#endif
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int i = blockIdx.x;
#if QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int ip  = tid/32;   // ip is 0 or 1
    const int il  = tid - 32*ip; // 0...32
    const int is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

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

    dst_t * y = yy + i*QK_K + 16*ip + il;

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

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

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

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

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
        const float d = (float)x[i].dm[0];
        const float m = (float)x[i].dm[1];
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

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

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

static __device__ void convert_f32(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const float * x = (const float *) vx;

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

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(const void * x, const int32_t * y, dst_t * dst, const int ncols) {
    const int col = (blockIdx.x*blockDim.x + threadIdx.x)*2;
    const int row = blockDim.y*blockIdx.y + threadIdx.y;

    if (col >= ncols) {
        return;
    }

    const int r = y[row];

    // copy x[r*ncols + col] to dst[row*ncols + col]
    const int xi = r*ncols + col;
    const int di = row*ncols + col;

    const int ib = xi/qk; // block index
    const int iqs = (xi%qk)/qr; // quant index
    const int iybs = di - di%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(x, ib, iqs, v);

    dst[iybs + iqs + 0]        = v.x;
    dst[iybs + iqs + y_offset] = v.y;
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
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

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
#else
    assert(false);
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
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
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

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
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
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);

#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = __dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * sumi;
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = __dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * __dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float & d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;

        const int sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = __dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m    = __dp4a(m,    u[i], sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const float2 dm2f = __half22float2(dm2);

    return d8 * (dm2f.x*sumi_d - dm2f.y*sumi_m);
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf = 0.0f;

#pragma unroll
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

    return d3 * sumf;
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = __dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;

#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = __dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;

#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
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

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;

#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = __dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;

#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
#else
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = __dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = __dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = __dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = __dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc[i0/2+0]*sumi_d.x + sc[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;

#else
    assert(false);
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

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI4_0) + mmq_y/QI4_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (block_q4_0 *) vx;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_0) % WARP_SIZE];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
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

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_qs[mmq_y * (WARP_SIZE) +     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI4_1) + mmq_y/QI4_1];

    *x_ql = tile_x_qs;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
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
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
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

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_1) % WARP_SIZE];
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
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

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int  tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI5_0) + mmq_y/QI5_0];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
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
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
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

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_0) % WARP_SIZE];
    }

    return vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
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

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_1) + mmq_y/QI5_1];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset < nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
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
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
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

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_1) % WARP_SIZE];
    }

    return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q8_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI8_0) + mmq_y/QI8_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2half(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q2_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI2_K) + mmq_y/QI2_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
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
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
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

    const int kbx = k / QI2_K;
    const int ky  = (k % QI2_K) * QR2_K;
    const float * y_df = (const float *) y_ds;

    int v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    const int kqsx = i * (WARP_SIZE + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    const int shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    for (int l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    const uint8_t * scales = ((const uint8_t *) &x_sc[i * (WARP_SIZE/4) + i/4 + kbx*4]) + ky/4;

    const int index_y = j * WARP_SIZE + (QR2_K*k) % WARP_SIZE;
    return vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
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

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2half(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q3_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI3_K) + mmq_y/QI3_K];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE/2)     + mmq_y/2];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const int kbx  = k / QI3_K;
    const int ky  = (k % QI3_K) * QR3_K;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * scales = ((int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

    int v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        const int kqsx = i * (WARP_SIZE + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        const int shift = 2 * ((ky % 32) / 8);
        const int vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        const int vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        const int vlh = (vh << 2) & 0x04040404;

        v[l] = __vsubss4(vll, vlh);
    }

    const int index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    return vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
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
        d8[i] = __low2half(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);

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

    const float dall = bq4_K->dm[0];
    const float dmin = bq4_K->dm[1];

    const float d8_1 = __low2float(bq8_1[0].ds);
    const float d8_2 = __low2float(bq8_1[1].ds);

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
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A

#endif
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI4_K) + mmq_y/QI4_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#else
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    const int index_y = j * WARP_SIZE + (QR4_K*k) % WARP_SIZE;
    return vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
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

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#else

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    const int8_t * s = bq5_K->scales;

    const float d = bq5_K->d;

    const float d8_1 = __low2half(bq8_1[0].ds);
    const float d8_2 = __low2half(bq8_1[1].ds);

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
    assert(false);
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A

#endif
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_K) + mmq_y/QI5_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        const int * scales = (int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    const int index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    const int index_y = j * WARP_SIZE             + (QR5_K*k) % WARP_SIZE;
    return vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
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

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2half(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q6_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI6_K) + mmq_y/QI6_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

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

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/8]);

    const int index_x = i * (QR6_K*WARP_SIZE + 1) +  QR6_K*k;
    const int index_y = j * WARP_SIZE             + (QR6_K*k) % WARP_SIZE;
    return vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}

template <int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x, int mmq_y, int nwarps,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void mul_mat_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE / qi;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = blockIdx.x*mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = blockIdx.y*mmq_x;
    const int & col_y_0 = col_dst_0;

    int   * tile_x_ql = nullptr;
    half2 * tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    __shared__ int    tile_y_qs[mmq_x * WARP_SIZE];
    __shared__ half2  tile_y_ds[mmq_x * WARP_SIZE/QI8_1];

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {0.0f};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0*blocks_per_row_x + ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   threadIdx.y, nrows_x-row_x_0-1, threadIdx.x, blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr; ++ir) {
            const int kqs = ir*WARP_SIZE + threadIdx.x;
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); // to prevent out-of-bounds memory accesses

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                const int index_y = (threadIdx.y + i) * WARP_SIZE + kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE/QI8_1)) % mmq_x;
                const int kby = threadIdx.x % (WARP_SIZE/QI8_1);
                const int col_y_eff = min(col_y_0 + ids, ncols_y-1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const half2 * dsi_src = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(WARP_SIZE/QI8_1) + kby].ds;
                half2       * dsi_dst = &tile_y_ds[ids * (WARP_SIZE/QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = __low2half(*dsi_src);
                }
            }

            __syncthreads();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE) {
                        sum[i/WARP_SIZE][j/nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                            threadIdx.x + i, threadIdx.y + j, k);
                    }
                }
            }

            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + threadIdx.y;

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + threadIdx.x + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q4_0_RDNA2;
    const int mmq_y  =  MMQ_Y_Q4_0_RDNA2;
    const int nwarps = NWARPS_Q4_0_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q4_0_RDNA1;
    const int mmq_y  =  MMQ_Y_Q4_0_RDNA1;
    const int nwarps = NWARPS_Q4_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q4_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    const int nwarps = NWARPS_Q4_0_AMPERE;

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q4_0_PASCAL;
    const int mmq_y  =  MMQ_Y_Q4_0_PASCAL;
    const int nwarps = NWARPS_Q4_0_PASCAL;

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q4_0_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_1_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_1_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q4_1_RDNA2;
    const int mmq_y  =  MMQ_Y_Q4_1_RDNA2;
    const int nwarps = NWARPS_Q4_1_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q4_1_RDNA1;
    const int mmq_y  =  MMQ_Y_Q4_1_RDNA1;
    const int nwarps = NWARPS_Q4_1_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q4_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    const int nwarps = NWARPS_Q4_1_AMPERE;

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q4_1_PASCAL;
    const int mmq_y  =  MMQ_Y_Q4_1_PASCAL;
    const int nwarps = NWARPS_Q4_1_PASCAL;

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q4_1_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q5_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q5_0_RDNA2;
    const int mmq_y  =  MMQ_Y_Q5_0_RDNA2;
    const int nwarps = NWARPS_Q5_0_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q5_0_RDNA1;
    const int mmq_y  =  MMQ_Y_Q5_0_RDNA1;
    const int nwarps = NWARPS_Q5_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q5_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    const int nwarps = NWARPS_Q5_0_AMPERE;

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q5_0_PASCAL;
    const int mmq_y  =  MMQ_Y_Q5_0_PASCAL;
    const int nwarps = NWARPS_Q5_0_PASCAL;

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q5_0_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q5_1_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q5_1_RDNA2;
    const int mmq_y  =  MMQ_Y_Q5_1_RDNA2;
    const int nwarps = NWARPS_Q5_1_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q5_1_RDNA1;
    const int mmq_y  =  MMQ_Y_Q5_1_RDNA1;
    const int nwarps = NWARPS_Q5_1_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q5_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    const int nwarps = NWARPS_Q5_1_AMPERE;

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q5_1_PASCAL;
    const int mmq_y  =  MMQ_Y_Q5_1_PASCAL;
    const int nwarps = NWARPS_Q5_1_PASCAL;

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q5_1_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q8_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q8_0_RDNA2;
    const int mmq_y  =  MMQ_Y_Q8_0_RDNA2;
    const int nwarps = NWARPS_Q8_0_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q8_0_RDNA1;
    const int mmq_y  =  MMQ_Y_Q8_0_RDNA1;
    const int nwarps = NWARPS_Q8_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q8_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    const int nwarps = NWARPS_Q8_0_AMPERE;

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q8_0_PASCAL;
    const int mmq_y  =  MMQ_Y_Q8_0_PASCAL;
    const int nwarps = NWARPS_Q8_0_PASCAL;

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q8_0_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q2_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q2_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q2_K_RDNA2;
    const int nwarps = NWARPS_Q2_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q2_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q2_K_RDNA1;
    const int nwarps = NWARPS_Q2_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q2_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    const int nwarps = NWARPS_Q2_K_AMPERE;

    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q2_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q2_K_PASCAL;
    const int nwarps = NWARPS_Q2_K_PASCAL;

    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q2_K_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q3_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q3_K_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q3_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q3_K_RDNA2;
    const int nwarps = NWARPS_Q3_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q3_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q3_K_RDNA1;
    const int nwarps = NWARPS_Q3_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q3_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    const int nwarps = NWARPS_Q3_K_AMPERE;

    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q3_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q3_K_PASCAL;
    const int nwarps = NWARPS_Q3_K_PASCAL;

    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q3_K_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_K_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q4_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q4_K_RDNA2;
    const int nwarps = NWARPS_Q4_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q4_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q4_K_RDNA1;
    const int nwarps = NWARPS_Q4_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q4_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    const int nwarps = NWARPS_Q4_K_AMPERE;

    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q4_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q4_K_PASCAL;
    const int nwarps = NWARPS_Q4_K_PASCAL;

    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q4_K_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q5_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q5_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q5_K_RDNA2;
    const int nwarps = NWARPS_Q5_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q5_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q5_K_RDNA1;
    const int nwarps = NWARPS_Q5_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q5_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    const int nwarps = NWARPS_Q5_K_AMPERE;

    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q5_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q5_K_PASCAL;
    const int nwarps = NWARPS_Q5_K_PASCAL;

    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q5_K_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q6_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q6_K_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q6_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q6_K_RDNA2;
    const int nwarps = NWARPS_Q6_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q6_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q6_K_RDNA1;
    const int nwarps = NWARPS_Q6_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q6_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    const int nwarps = NWARPS_Q6_K_AMPERE;

    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q6_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q6_K_PASCAL;
    const int nwarps = NWARPS_Q6_K_PASCAL;

    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    (void) vec_dot_q6_K_q8_1_mul_mat;
    assert(false);
#endif // __CUDA_ARCH__ >= CC_VOLTA
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

template<typename T, bool has_pos>
static __global__ void rope(const T * x, T * dst, const int ncols, const int32_t * pos, const float freq_scale,
                            const int p_delta_rows, const float theta_scale) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
    const int i2 = row/p_delta_rows;

    const int p = has_pos ? pos[i2] : 0;
    const float p0 = p*freq_scale;
    const float theta = p0*powf(theta_scale, col/2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

template<typename T, bool has_pos>
static __global__ void rope_neox(const T * x, T * dst, const int ncols, const int32_t * pos, const float freq_scale,
                                 const int p_delta_rows, const float theta_scale) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col/2;
    const int i2 = row/p_delta_rows;

    const int p = has_pos ? pos[i2] : 0;
    const float p0 = p*freq_scale;
    const float theta = p0*powf(theta_scale, col/2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + ncols/2];

    dst[i + 0]       = x0*cos_theta - x1*sin_theta;
    dst[i + ncols/2] = x0*sin_theta + x1*cos_theta;
}

static __global__ void rope_glm_f32(const float * x, float * dst, const int ncols, const int32_t * pos, const float freq_scale,
                                    const int p_delta_rows, const float theta_scale, const int n_ctx) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    const int half_n_dims = ncols/4;

    if (col >= half_n_dims) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;
    const int i2 = row/p_delta_rows;

    const float col_theta_scale = powf(theta_scale, col);
     // FIXME: this is likely wrong
    const int p = pos != nullptr ? pos[i2] : 0;

    const float theta = min(p, n_ctx - 2)*freq_scale*col_theta_scale;
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + half_n_dims];

    dst[i + 0]           = x0*cos_theta - x1*sin_theta;
    dst[i + half_n_dims] = x0*sin_theta + x1*cos_theta;

    const float block_theta = ((float)max(p - n_ctx - 2, 0))*col_theta_scale;
    const float sin_block_theta = sinf(block_theta);
    const float cos_block_theta = cosf(block_theta);

    const float x2 = x[i + half_n_dims * 2];
    const float x3 = x[i + half_n_dims * 3];

    dst[i + half_n_dims * 2] = x2*cos_block_theta - x3*sin_block_theta;
    dst[i + half_n_dims * 3] = x2*sin_block_theta + x3*cos_block_theta;
}

static __global__ void alibi_f32(const float * x, float * dst, const int ncols, const int k_rows,
                                 const int n_heads_log2_floor, const float m0, const float m1) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;

    const int k = row/k_rows;

    float m_k;
    if (k < n_heads_log2_floor) {
        m_k = powf(m0, k + 1);
    } else {
        m_k = powf(m1, 2 * (k - n_heads_log2_floor) + 1);
    }

    dst[i] = col * m_k + x[i];
}

static __global__ void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past) {
    const int col = blockDim.y*blockIdx.y + threadIdx.y;
    const int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    // dst[i] = col > n_past + row ? -INFINITY : x[i];
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
}

// the CUDA soft max implementation differs from the CPU implementation
// instead of doubles floats are used
static __global__ void soft_max_f32(const float * x, float * dst, const int ncols) {
    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;

    float max_val = -INFINITY;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        max_val = max(max_val, x[i]);
    }

    // find the max value in the block
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, mask, 32));
    }

    float tmp = 0.f;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        const float val = expf(x[i] - max_val);
        tmp += val;
        dst[i] = val;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const float inv_tmp = 1.f / tmp;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        dst[i] *= inv_tmp;
    }
}

static __global__ void scale_f32(const float * x, float * dst, const float scale, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i];
}

static __global__ void clamp_f32(const float * x, float * dst, const float min, const float max, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = x[i] < min ? min : (x[i] > max ? max : x[i]);
}

template<int qk, int qr, dequantize_kernel_t dq>
static void get_rows_cuda(const void * x, const int32_t * y, float * dst, const int nrows, const int ncols, cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ncols + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(block_num_x, nrows, 1);
    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(x, y, dst, ncols);
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
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(x, dst, ncols);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols);
    }
}

static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

static void quantize_row_q8_1_cuda(const float * x, void * vy, const int kx, const int ky, const int kx_padded, cudaStream_t stream) {
    const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ky, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK4_0, QR4_0, dequantize_q4_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK4_1, QR4_1, dequantize_q4_1><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q5_0_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK5_0, QR5_0, dequantize_q5_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q5_1_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK5_1, QR5_1, dequantize_q5_1><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q8_0_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK8_0, QR8_0, dequantize_q8_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q2_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q3_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
#else
    dequantize_block_q5_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
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
    mul_mat_vec_q<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

static void convert_fp16_to_fp32_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<1, 1, convert_f16><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void convert_fp32_to_fp16_cuda(const void * vx, half * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    dequantize_block<1, 1, convert_f32><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void convert_mul_mat_vec_f16_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    dequantize_mul_mat_vec<1, 1, convert_f16>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

static to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
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
        case GGML_TYPE_F32:
            return convert_fp32_to_fp16_cuda;
        default:
            return nullptr;
    }
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

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q4_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q8_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q8_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q8_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q2_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q2_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q2_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q3_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

#if QK_K == 256

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q3_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q3_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
#endif
}

static void ggml_mul_mat_q4_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q6_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    const int compute_capability = g_compute_capabilities[id];

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q6_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q6_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
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

static void clamp_f32_cuda(const float * x, float * dst, const float min, const float max, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
    clamp_f32<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max, k);
}

template<typename T>
static void rope_cuda(const T * x, T * dst, const int ncols, const int nrows, const int32_t * pos, const float freq_scale,
                          const int p_delta_rows, const float theta_scale, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nrows, num_blocks_x, 1);
    if (pos == nullptr) {
        rope<T, false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, pos, freq_scale, p_delta_rows, theta_scale);
    } else {
        rope<T, true><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, pos, freq_scale, p_delta_rows, theta_scale);
    }
}

template<typename T>
static void rope_neox_cuda(const T * x, T * dst, const int ncols, const int nrows, const int32_t * pos, const float freq_scale,
                          const int p_delta_rows, const float theta_scale, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nrows, num_blocks_x, 1);
    if (pos == nullptr) {
        rope_neox<T, false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, pos, freq_scale, p_delta_rows, theta_scale);
    } else {
        rope_neox<T, true><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, pos, freq_scale, p_delta_rows, theta_scale);
    }
}

static void rope_glm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const int32_t * pos, const float freq_scale,
                              const int p_delta_rows, const float theta_scale, const int n_ctx, cudaStream_t stream) {
    GGML_ASSERT(ncols % 4 == 0);
    const dim3 block_dims(CUDA_ROPE_BLOCK_SIZE/4, 1, 1);
    const int num_blocks_x = (ncols + CUDA_ROPE_BLOCK_SIZE - 1) / CUDA_ROPE_BLOCK_SIZE;
    const dim3 block_nums(num_blocks_x, nrows, 1);
    rope_glm_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, pos, freq_scale, p_delta_rows, theta_scale, n_ctx);
}

static void alibi_f32_cuda(const float * x, float * dst, const int ncols, const int nrows,
                           const int k_rows, const int n_heads_log2_floor, const float m0,
                           const float m1, cudaStream_t stream) {
    const dim3 block_dims(CUDA_ALIBI_BLOCK_SIZE, 1, 1);
    const int num_blocks_x = (ncols + CUDA_ALIBI_BLOCK_SIZE - 1) / (CUDA_ALIBI_BLOCK_SIZE);
    const dim3 block_nums(num_blocks_x, nrows, 1);
    alibi_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, k_rows, n_heads_log2_floor, m0, m1);
}

static void diag_mask_inf_f32_cuda(const float * x, float * dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream) {
    const dim3 block_dims(1, CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
    const dim3 block_nums(nrows_x, block_num_x, 1);
    diag_mask_inf_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x, rows_per_channel, n_past);
}

static void soft_max_f32_cuda(const float * x, float * dst, const int ncols_x, const int nrows_x, cudaStream_t stream) {
    const dim3 block_dims(1, WARP_SIZE, 1);
    const dim3 block_nums(nrows_x, 1, 1);
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


void ggml_init_cublas() {
    static bool initialized = false;

    if (!initialized) {

#ifdef __HIP_PLATFORM_AMD__
        // Workaround for a rocBLAS bug when using multiple graphics cards:
        // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
        rocblas_initialize();
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        CUDA_CHECK(cudaGetDeviceCount(&g_device_count));
        GGML_ASSERT(g_device_count <= GGML_CUDA_MAX_DEVICES);
        int64_t total_vram = 0;
        fprintf(stderr, "%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, g_device_count);
        for (int64_t id = 0; id < g_device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            fprintf(stderr, "  Device %ld: %s, compute capability %d.%d\n", id, prop.name, prop.major, prop.minor);

            g_tensor_split[id] = total_vram;
            total_vram += prop.totalGlobalMem;
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
            g_compute_capabilities[id] = 100*prop.major + 10*prop.minor + CC_OFFSET_AMD;
#else
            g_compute_capabilities[id] = 100*prop.major + 10*prop.minor;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        }
        for (int64_t id = 0; id < g_device_count; ++id) {
            g_tensor_split[id] /= total_vram;
        }

        for (int64_t id = 0; id < g_device_count; ++id) {
            CUDA_CHECK(ggml_cuda_set_device(id));

            // create cuda streams
            for (int64_t is = 0; is < MAX_STREAMS; ++is) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams[id][is], cudaStreamNonBlocking));
            }

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
    } else if (src->backend == GGML_BACKEND_GPU || src->backend == GGML_BACKEND_GPU_SPLIT) {
        GGML_ASSERT(src->backend != GGML_BACKEND_GPU_SPLIT || (i1_low == 0 && i1_high == src->ne[1]));
        kind = cudaMemcpyDeviceToDevice;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
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

static void ggml_cuda_op_repeat(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_d, const float * src1_d, float * dst_d, const cudaStream_t & stream) {
    // guaranteed to be an integer due to the check in ggml_can_repeat
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const int nr0 = (int)(ne0/ne00);
    const int nr1 = (int)(ne1/ne01);
    const int nr2 = (int)(ne2/ne02);
    const int nr3 = (int)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    // TODO: very inefficient, implement in a kernel, or fewer cudaMemcpyAsync calls for contiguous tensors
    for                         (int i3 = 0; i3 < nr3;  i3++) {
        for                     (int k3 = 0; k3 < ne03; k3++) {
            for                 (int i2 = 0; i2 < nr2;  i2++) {
                for             (int k2 = 0; k2 < ne02; k2++) {
                    for         (int i1 = 0; i1 < nr1;  i1++) {
                        for     (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0;  i0++) {
                                CUDA_CHECK(cudaMemcpyAsync(
                                              (char *)  dst_d + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0,
                                        (const char *) src0_d + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01,
                                        ne00*nb0, cudaMemcpyDeviceToDevice, stream));
                            }
                        }
                    }
                }
            }
        }
    }

    (void) src1;
    (void) src1_d;
}

static void ggml_cuda_op_get_rows(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_d, const float * src1_d, float * dst_d, const cudaStream_t & stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int ncols = src0->ne[0];
    const int nrows = ggml_nelements(src1);

    const int32_t * src1_i32 = (const int32_t *) src1_d;

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_cuda<1, 1, convert_f16>(src0_d, src1_i32, dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda<1, 1, convert_f32>(src0_d, src1_i32, dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_i32, dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_i32, dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_i32, dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_i32, dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_i32, dst_d, nrows, ncols, stream);
            break;
        default:
            // TODO: k-quants
            GGML_ASSERT(false);
            break;
    }
}

inline void ggml_cuda_op_add(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        add_f32_cuda(src0_dd, src1_dd, dst_dd, ggml_nelements(src0), ne10*ne11, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        add_f16_f32_f16_cuda((const half *) src0_dd, src1_dd, (half *) dst_dd, ggml_nelements(src0), main_stream);
    } else {
        GGML_ASSERT(false);
    }

    (void) src1;
    (void) dst;
}

inline void ggml_cuda_op_mul(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    mul_f32_cuda(src0_dd, src1_dd, dst_dd, ggml_nelements(src0), ne10*ne11, main_stream);

    (void) dst;
}

inline void ggml_cuda_op_gelu(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_f32_cuda(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_cuda_op_silu(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    silu_f32_cuda(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_cuda_op_norm(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    norm_f32_cuda(src0_dd, dst_dd, ne00, nrows, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_cuda_op_rms_norm(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    rms_norm_f32_cuda(src0_dd, dst_dd, ne00, nrows, eps, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_cuda_op_mul_mat_q(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, const cudaStream_t & stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    const int64_t nrows_dst = dst->backend == GGML_BACKEND_GPU && id == g_main_device ? ne0 : row_diff;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    (void) src1;
    (void) dst;
    (void) src1_ddf_i;
}

static int64_t get_row_rounding(ggml_type type) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int64_t id = 0; id < g_device_count; ++id) {
        if (g_tensor_split[id] < (id + 1 < g_device_count ? g_tensor_split[id + 1] : 1.0f)) {
            if (min_compute_capability > g_compute_capabilities[id]) {
                min_compute_capability = g_compute_capabilities[id];
            }
            if (max_compute_capability < g_compute_capabilities[id]) {
                max_compute_capability = g_compute_capabilities[id];
            }
        }
    }

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return max_compute_capability >= CC_RDNA2 ? 128 : 64;
        case GGML_TYPE_F16:
            return 1;
        case GGML_TYPE_Q2_K:
            return max_compute_capability >= CC_RDNA2 ? 128 : 32;
        case GGML_TYPE_Q3_K:
            return min_compute_capability < CC_RDNA2 ? 128 : 64;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return max_compute_capability >= CC_RDNA2 ? 128 : 64;
        default:
            GGML_ASSERT(false);
    }
#else
    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ASSERT(false);
    }
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
}

inline void ggml_cuda_op_mul_mat_vec_q(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, const cudaStream_t & stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    (void) src1;
    (void) dst;
    (void) src1_ddf_i;
    (void) src1_ncols;
    (void) src1_padded_row_size;
}

inline void ggml_cuda_op_dequantize_mul_mat_vec(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, const cudaStream_t & stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_CUDA_F16
    size_t ash;
    dfloat * src1_dfloat = nullptr; // dfloat == half

    bool src1_convert_f16 = src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_F16;

    if (src1_convert_f16) {
        src1_dfloat = (half *) ggml_cuda_pool_malloc(ne00*sizeof(half), &ash);
        ggml_cpy_f32_f16_cuda((const char *) src1_ddf_i, (char *) src1_dfloat, ne00,
                                ne00, 1, sizeof(float), 0, 0,
                                ne00, 1, sizeof(half),  0, 0, stream);
    }
#else
    const dfloat * src1_dfloat = (const dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_CUDA_F16

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            dequantize_mul_mat_vec_q4_0_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_mul_mat_vec_q8_0_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_q2_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_q3_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_q4_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_q5_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_q6_K_cuda(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_F16:
            convert_mul_mat_vec_f16_cuda(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
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

    (void) src1;
    (void) dst;
    (void) src1_ddq_i;
    (void) src1_ncols;
    (void) src1_padded_row_size;
}

inline void ggml_cuda_op_mul_mat_cublas(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, const cudaStream_t & stream) {

    GGML_ASSERT(src0_dd_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i != nullptr);


    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = dst->backend == GGML_BACKEND_GPU && id == g_main_device ? ne0 : row_diff;

    const int compute_capability = g_compute_capabilities[id];

    if (compute_capability >= CC_VOLTA && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        half * src0_as_f16 = nullptr;
        size_t src0_as = 0;
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16 = (half *) ggml_cuda_pool_malloc(ne * sizeof(half), &src0_as);
            to_fp16_cuda(src0_dd_i, src0_as_f16, ne, stream);
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16;

        half * src1_as_f16 = nullptr;
        size_t src1_as = 0;
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16 = (half *) ggml_cuda_pool_malloc(ne * sizeof(half), &src1_as);
            to_fp16_cuda(src1_ddf_i, src1_as_f16, ne, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddq_i : src1_as_f16;

        size_t dst_as = 0;
        half * dst_f16 = (half *) ggml_cuda_pool_malloc(row_diff*src1_ncols * sizeof(half), &dst_as);

        const half alpha_f16 = 1.0f;
        const half beta_f16 = 0.0f;

        CUBLAS_CHECK(cublasSetStream(g_cublas_handles[id], stream));
        CUBLAS_CHECK(
            cublasGemmEx(g_cublas_handles[id], CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f16, src0_ptr, CUDA_R_16F, ne00,
                                src1_ptr, CUDA_R_16F, ne10,
                    &beta_f16,   dst_f16, CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16, dst_dd_i, row_diff*src1_ncols, stream);

        ggml_cuda_pool_free(dst_f16, dst_as);

        if (src0_as != 0) {
            ggml_cuda_pool_free(src0_as_f16, src0_as);
        }

        if (src1_as != 0) {
            ggml_cuda_pool_free(src1_as_f16, src1_as);
        }
    }
    else {
        float * src0_ddq_as_f32 = nullptr;
        size_t src0_as = 0;

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32 = (float *) ggml_cuda_pool_malloc(row_diff*ne00 * sizeof(float), &src0_as); // NOLINT
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32, row_diff*ne00, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32;

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(g_cublas_handles[id], stream));
        CUBLAS_CHECK(
            cublasSgemm(g_cublas_handles[id], CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i, ne00,
                            src1_ddf_i,  ne10,
                    &beta,  dst_dd_i,   ldc));

        if (src0_as != 0) {
            ggml_cuda_pool_free(src0_ddq_as_f32, src0_as);
        }
    }

    (void) dst;
    (void) src1_ddq_i;
    (void) src1_padded_row_size;
}

inline void ggml_cuda_op_rope(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t nrows = ggml_nrows(src0);

    //const int n_past = ((int32_t *) dst->op_params)[0];
    const int n_dims = ((int32_t *) dst->op_params)[1];
    const int mode   = ((int32_t *) dst->op_params)[2];
    const int n_ctx  = ((int32_t *) dst->op_params)[3];
    // RoPE alteration for extended context

    float freq_base, freq_scale;
    memcpy(&freq_base,  (int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 5, sizeof(float));

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const int32_t * pos = nullptr;
    if ((mode & 1) == 0) {
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        GGML_ASSERT(src1->ne[0] == ne2);
        pos = (const int32_t *) src1_dd;
    }

    const bool is_neox = mode & 2;
    const bool is_glm  = mode & 4;

    // compute
    if (is_glm) {
        GGML_ASSERT(false);
        rope_glm_f32_cuda(src0_dd, dst_dd, ne00, nrows, pos, freq_scale, ne01, theta_scale, n_ctx, main_stream);
    } else if (is_neox) {
        GGML_ASSERT(ne00 == n_dims && "ne00 != n_dims is not implemented for CUDA yet");
        if (src0->type == GGML_TYPE_F32) {
            rope_neox_cuda((const float *)src0_dd, (float *)dst_dd, ne00, nrows, pos, freq_scale, ne01, theta_scale, main_stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_neox_cuda((const half *)src0_dd, (half *)dst_dd, ne00, nrows, pos, freq_scale, ne01, theta_scale, main_stream);
        } else {
            GGML_ASSERT(false);
        }
    } else {
        if (src0->type == GGML_TYPE_F32) {
            rope_cuda((const float *)src0_dd, (float *)dst_dd, ne00, nrows, pos, freq_scale, ne01, theta_scale, main_stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_cuda((const half *)src0_dd, (half *)dst_dd, ne00, nrows, pos, freq_scale, ne01, theta_scale, main_stream);
        } else {
            GGML_ASSERT(false);
        }
    }

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_cuda_op_alibi(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t nrows = ggml_nrows(src0);

    //const int n_past = ((int32_t *) dst->op_params)[0];
    const int n_head = ((int32_t *) dst->op_params)[1];
    float max_bias;
    memcpy(&max_bias, (int32_t *) dst->op_params + 2, sizeof(float));

    //GGML_ASSERT(ne01 + n_past == ne00);
    GGML_ASSERT(n_head == ne02);

    const int n_heads_log2_floor = 1 << (int) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

    alibi_f32_cuda(src0_dd, dst_dd, ne00, nrows, ne01, n_heads_log2_floor, m0, m1, main_stream);

    (void) src1;
    (void) src1_dd;
}

inline void ggml_cuda_op_diag_mask_inf(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int nrows0 = ggml_nrows(src0);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_cuda(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_cuda_op_soft_max(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    soft_max_f32_cuda(src0_dd, dst_dd, ne00, nrows, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_cuda_op_scale(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    // HACK: support for ggml backend interface
    if (src1->backend == GGML_BACKEND_CPU) {
        scale = ((float *) src1->data)[0];
    } else {
        // TODO: pass pointer to kernel instead of copying to host
        CUDA_CHECK(cudaMemcpy(&scale, src1->data, sizeof(float), cudaMemcpyDeviceToHost));
    }

    scale_f32_cuda(src0_dd, dst_dd, scale, ggml_nelements(src0), main_stream);
    CUDA_CHECK(cudaGetLastError());

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_cuda_op_clamp(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const float min = ((float *) dst->op_params)[0];
    const float max = ((float *) dst->op_params)[1];

    clamp_f32_cuda(src0_dd, dst_dd, min, max, ggml_nelements(src0), main_stream);
    CUDA_CHECK(cudaGetLastError());

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

static void ggml_cuda_op_flatten(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const ggml_cuda_op_flatten_t op) {
    const int64_t nrows0 = ggml_nrows(src0);

    const bool use_src1 = src1 != nullptr;
    const int64_t nrows1 = use_src1 ? ggml_nrows(src1) : 1;

    GGML_ASSERT(!use_src1 || src1->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(              dst->backend != GGML_BACKEND_GPU_SPLIT);

    ggml_tensor_extra_gpu * src0_extra =            (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = use_src1 ? (ggml_tensor_extra_gpu *) src1->extra : nullptr;
    ggml_tensor_extra_gpu * dst_extra  =            (ggml_tensor_extra_gpu *)  dst->extra;

    const bool src0_on_device =             src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT;
    const bool src1_on_device = use_src1 && src1->backend == GGML_BACKEND_GPU;
    const bool  dst_on_device =              dst->backend == GGML_BACKEND_GPU;

    const bool src1_stays_on_host = use_src1 && dst->op == GGML_OP_SCALE;

    // dd = data device
    float * src0_ddf = nullptr;
    float * src1_ddf = nullptr;
    float *  dst_ddf = nullptr;

    // as = actual size
    size_t src0_asf = 0;
    size_t src1_asf = 0;
    size_t  dst_asf = 0;

    ggml_cuda_set_device(g_main_device);
    const cudaStream_t main_stream = g_cudaStreams[g_main_device][0];

    if (src0_on_device) {
        src0_ddf = (float *) src0_extra->data_device[g_main_device];
    } else {
        src0_ddf = (float *) ggml_cuda_pool_malloc(ggml_nbytes(src0), &src0_asf);
        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_ddf, src0, 0, 0, 0, nrows0, main_stream));
    }

    if (use_src1 && !src1_stays_on_host) {
        if (src1_on_device) {
            src1_ddf = (float *) src1_extra->data_device[g_main_device];
        } else {
            src1_ddf = (float *) ggml_cuda_pool_malloc(ggml_nbytes(src1), &src1_asf);
            CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src1_ddf, src1, 0, 0, 0, nrows1, main_stream));
        }
    }
    if (dst_on_device) {
        dst_ddf = (float *) dst_extra->data_device[g_main_device];
    } else {
        dst_ddf = (float *) ggml_cuda_pool_malloc(ggml_nbytes(dst), &dst_asf);
    }

    // do the computation
    op(src0, src1, dst, src0_ddf, src1_ddf, dst_ddf, main_stream);
    CUDA_CHECK(cudaGetLastError());

    // copy dst to host if necessary
    if (!dst_on_device) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, dst_ddf, ggml_nbytes(dst), cudaMemcpyDeviceToHost, main_stream));
    }

    if (src0_asf > 0) {
        ggml_cuda_pool_free(src0_ddf, src0_asf);
    }
    if (src1_asf > 0) {
        ggml_cuda_pool_free(src1_ddf, src1_asf);
    }
    if (dst_asf > 0) {
        ggml_cuda_pool_free(dst_ddf, dst_asf);
    }

    if (dst->backend == GGML_BACKEND_CPU) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

static void ggml_cuda_set_peer_access(const int n_tokens) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_CUDA_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int id = 0; id < g_device_count; ++id) {
        CUDA_CHECK(ggml_cuda_set_device(id));

        for (int id_other = 0; id_other < g_device_count; ++id_other) {
            if (id == id_other) {
                continue;
            }
            if (id != g_main_device && id_other != g_main_device) {
                continue;
            }

            int can_access_peer;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (can_access_peer) {
                if (enable_peer_access) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                } else {
                    CUDA_CHECK(cudaDeviceDisablePeerAccess(id_other));
                }
            }
        }
    }
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

static void ggml_cuda_op_mul_mat(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    const bool convert_src1_to_q8_1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t nrows0 = ggml_nrows(src0);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    ggml_cuda_set_peer_access(ne11);

    GGML_ASSERT(dst->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src1->backend != GGML_BACKEND_GPU_SPLIT);

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    ggml_tensor_extra_gpu *  dst_extra = (ggml_tensor_extra_gpu *)  dst->extra;

    const bool src0_on_device = src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT;
    const bool src0_is_contiguous = ggml_is_contiguous(src0);

    const bool src1_is_contiguous = ggml_is_contiguous(src1);
    const int64_t src1_padded_col_size = ne10 % MATRIX_ROW_PADDING == 0 ?
        ne10 : ne10 - ne10 % MATRIX_ROW_PADDING + MATRIX_ROW_PADDING;

    const bool split = src0->backend == GGML_BACKEND_GPU_SPLIT;
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    // dd = data device
    char  *  src0_dd[GGML_CUDA_MAX_DEVICES] = {nullptr};
    float * src1_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr}; // float
    char  * src1_ddq[GGML_CUDA_MAX_DEVICES] = {nullptr}; // q8_1
    float *   dst_dd[GGML_CUDA_MAX_DEVICES] = {nullptr};

    // as = actual size
    size_t  src0_as[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src1_asf[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src1_asq[GGML_CUDA_MAX_DEVICES] = {0};
    size_t   dst_as[GGML_CUDA_MAX_DEVICES] = {0};

    int64_t  row_low[GGML_CUDA_MAX_DEVICES];
    int64_t row_high[GGML_CUDA_MAX_DEVICES];

    for (int64_t id = 0; id < g_device_count; ++id) {
        // by default, use all rows
        row_low[id]  = 0;
        row_high[id] = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type);

            if (id != 0) {
                row_low[id]  = ne01*g_tensor_split[id];
                row_low[id] -= row_low[id] % rounding;
            }

            if (id != g_device_count - 1) {
                row_high[id]  = ne01*g_tensor_split[id + 1];
                row_high[id] -= row_high[id] % rounding;
            }
        }
    }

    for (int64_t id = 0; id < g_device_count; ++id) {
        if ((!split && id != g_main_device) || row_low[id] == row_high[id]) {
            continue;
        }

        const bool src1_on_device = src1->backend == GGML_BACKEND_GPU && id == g_main_device;
        const bool  dst_on_device =  dst->backend == GGML_BACKEND_GPU && id == g_main_device;

        ggml_cuda_set_device(id);
        const cudaStream_t stream = g_cudaStreams[id][0];

        if (src0_on_device && src0_is_contiguous) {
            src0_dd[id] = (char *) src0_extra->data_device[id];
        } else {
            const size_t size_src0_ddq = split ? (row_high[id]-row_low[id])*ne00 * src0_ts/src0_bs : ggml_nbytes(src0);
            src0_dd[id] = (char *) ggml_cuda_pool_malloc(ggml_nbytes(src0), &src0_as[id]);
        }

        if (src1_on_device && src1_is_contiguous) {
            src1_ddf[id] = (float *) src1_extra->data_device[id];
        } else {
            src1_ddf[id] = (float *) ggml_cuda_pool_malloc(ggml_nbytes(src1), &src1_asf[id]);
        }

        if (convert_src1_to_q8_1) {
            src1_ddq[id] = (char *) ggml_cuda_pool_malloc(nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs, &src1_asq[id]);

            if (src1_on_device && src1_is_contiguous) {
                quantize_row_q8_1_cuda(src1_ddf[id], src1_ddq[id], ne10, nrows1, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dst_on_device) {
            dst_dd[id] = (float *) dst_extra->data_device[id];
        } else {
            const size_t size_dst_ddf = split ? (row_high[id]-row_low[id])*ne1*sizeof(float) : ggml_nbytes(dst);
            dst_dd[id] = (float *) ggml_cuda_pool_malloc(size_dst_ddf, &dst_as[id]);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && g_device_count > 1) {
        CUDA_CHECK(ggml_cuda_set_device(g_main_device));
        CUDA_CHECK(cudaEventRecord(src0_extra->events[g_main_device][0], g_cudaStreams[g_main_device][0]));
    }

    const int64_t src1_col_stride = split && g_device_count > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int64_t id = 0; id < g_device_count; ++id) {
            if ((!split && id != g_main_device) || row_low[id] == row_high[id]) {
                continue;
            }

            const bool src1_on_device = src1->backend == GGML_BACKEND_GPU && id == g_main_device;
            const bool  dst_on_device =  dst->backend == GGML_BACKEND_GPU && id == g_main_device;
            const int64_t row_diff = row_high[id] - row_low[id];

            ggml_cuda_set_device(id);
            const cudaStream_t stream = g_cudaStreams[id][is];

            // wait for main GPU data if necessary
            if (split && (id != g_main_device || is != 0)) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[g_main_device][0], 0));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  src0_dd[id] + (i0/i02_divisor) * ne01*ne00*src0_ts/src0_bs;
                float * src1_ddf_i = src1_ddf[id] + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = src1_ddq[id] +  src1_ddq_i_offset;
                float *   dst_dd_i =   dst_dd[id] + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (dst->backend == GGML_BACKEND_GPU && id == g_main_device) {
                    dst_dd_i += row_low[id]; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1->backend == GGML_BACKEND_GPU && src1_is_contiguous) {
                    if (id != g_main_device) {
                        if (convert_src1_to_q8_1) {
                            char * src1_ddq_i_source = src1_ddq[g_main_device] + src1_ddq_i_offset;
                            CUDA_CHECK(cudaMemcpyAsync(src1_ddq_i, src1_ddq_i_source, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs,
                                                    cudaMemcpyDeviceToDevice, stream));
                        } else {
                            float * src1_ddf_i_source = (float *) src1_extra->data_device[g_main_device];
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyAsync(src1_ddf_i, src1_ddf_i_source, src1_ncols*ne10*sizeof(float),
                                                    cudaMemcpyDeviceToDevice, stream));
                        }
                    }
                } else if (src1->backend == GGML_BACKEND_CPU || (src1_on_device && !src1_is_contiguous)) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                   src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ASSERT(false);
                }

                if (convert_src1_to_q8_1 && (src1->backend == GGML_BACKEND_CPU || !src1_is_contiguous)) {
                    quantize_row_q8_1_cuda(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, src1_padded_col_size, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && (!src0_on_device || !src0_is_contiguous) && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, row_low[id], row_high[id], stream));
                }

                // do the computation
                op(src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                   row_low[id], row_high[id], src1_ncols, src1_padded_col_size, stream);
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
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + row_low[id];
                        CUDA_CHECK(cudaMemcpy2DAsync(dhf_dst_i, ne0*sizeof(float), dst_dd_i, row_diff*sizeof(float),
                                                    row_diff*sizeof(float), src1_ncols, kind, stream));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), kind, stream));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != g_main_device || is != 0)) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }
            }
        }
    }

    for (int64_t id = 0; id < g_device_count; ++id) {
        CUDA_CHECK(ggml_cuda_set_device(id));

        // free buffers again when done
        if (src0_as[id] > 0) {
            ggml_cuda_pool_free(src0_dd[id], src0_as[id]);
        }
        if (src1_asf[id] > 0) {
            ggml_cuda_pool_free(src1_ddf[id], src1_asf[id]);
        }
        if (src1_asq[id] > 0) {
            ggml_cuda_pool_free(src1_ddq[id], src1_asq[id]);
        }
        if (dst_as[id] > 0) {
            ggml_cuda_pool_free(dst_dd[id], dst_as[id]);
        }
    }

    // main device waits for all other devices to be finished
    if (split && g_device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= MAX_STREAMS ? is_max : MAX_STREAMS;

        CUDA_CHECK(ggml_cuda_set_device(g_main_device));
        for (int64_t id = 0; id < g_device_count; ++id) {
            for (int64_t is = 0; is < is_max; ++is) {
                CUDA_CHECK(cudaStreamWaitEvent(g_cudaStreams[g_main_device][0], src0_extra->events[id][is], 0));
            }
        }
    }

    if (dst->backend == GGML_BACKEND_CPU) {
        CUDA_CHECK(ggml_cuda_set_device(g_main_device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

static void ggml_cuda_repeat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_repeat);
}

static void ggml_cuda_get_rows(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_get_rows);
}

static void ggml_cuda_add(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_add);
}

static void ggml_cuda_mul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_mul);
}

static void ggml_cuda_gelu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_gelu);
}

static void ggml_cuda_silu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_silu);
}

static void ggml_cuda_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_norm);
}

static void ggml_cuda_rms_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_rms_norm);
}

bool ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
            src1->type == GGML_TYPE_F32 &&
             dst->type == GGML_TYPE_F32 &&
            (ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
}

static void ggml_cuda_mul_mat_vec_p021(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
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

    CUDA_CHECK(ggml_cuda_set_device(g_main_device));
    cudaStream_t main_stream = g_cudaStreams[g_main_device][0];

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    void * src0_ddq = src0_extra->data_device[g_main_device];

    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_main_device];

    ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_main_device];

    ggml_mul_mat_p021_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}

static void ggml_cuda_mul_mat_vec_nc(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
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

    CUDA_CHECK(ggml_cuda_set_device(g_main_device));
    cudaStream_t main_stream = g_cudaStreams[g_main_device][0];

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    void * src0_ddq = src0_extra->data_device[g_main_device];

    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_main_device];

    ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_main_device];

    const int64_t row_stride_x = nb01 / sizeof(half);
    const int64_t channel_stride_x = nb02 / sizeof(half);

    ggml_mul_mat_vec_nc_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, main_stream);
}

static void ggml_cuda_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    bool all_on_device = (src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT) &&
        src1->backend == GGML_BACKEND_GPU && dst->backend == GGML_BACKEND_GPU;

    int64_t min_compute_capability = INT_MAX;
    for (int64_t id = 0; id < g_device_count; ++id) {
        if (min_compute_capability > g_compute_capabilities[id]
                && g_tensor_split[id] < (id + 1 < g_device_count ? g_tensor_split[id + 1] : 1.0f)) {
            min_compute_capability = g_compute_capabilities[id];
        }
    }

    if (all_on_device && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_p021(src0, src1, dst);
    } else if (all_on_device && !ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_nc(src0, src1, dst);
    } else if (src0->type == GGML_TYPE_F32) {
        ggml_cuda_op_mul_mat(src0, src1, dst, ggml_cuda_op_mul_mat_cublas, false);
    } else if (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F16) {
        if (src1->ne[1] == 1 && src0->ne[0] % GGML_CUDA_DMMV_X == 0) {

#ifdef GGML_CUDA_FORCE_DMMV
            const bool use_mul_mat_vec_q = false;
#else
            const bool use_mul_mat_vec_q = min_compute_capability >= MIN_CC_DP4A && ggml_is_quantized(src0->type);
#endif // GGML_CUDA_FORCE_DMMV

            if (use_mul_mat_vec_q) {
                ggml_cuda_op_mul_mat(src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, true);
            } else {
                ggml_cuda_op_mul_mat(src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, false);
            }
        } else {
            if (g_mul_mat_q && ggml_is_quantized(src0->type) && min_compute_capability >= MIN_CC_DP4A) {
                ggml_cuda_op_mul_mat(src0, src1, dst, ggml_cuda_op_mul_mat_q, true);
            } else {
                ggml_cuda_op_mul_mat(src0, src1, dst, ggml_cuda_op_mul_mat_cublas, false);
            }
        }
    } else {
        GGML_ASSERT(false);
    }
}

static void ggml_cuda_scale(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_scale);
}

static void ggml_cuda_clamp(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_clamp);
}

static void ggml_cuda_cpy(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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

    CUDA_CHECK(ggml_cuda_set_device(g_main_device));
    cudaStream_t main_stream = g_cudaStreams[g_main_device][0];

    const ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    const ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;

    char * src0_ddc = (char *) src0_extra->data_device[g_main_device];
    char * src1_ddc = (char *) src1_extra->data_device[g_main_device];

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, main_stream);
    } else {
        fprintf(stderr, "%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ASSERT(false);
    }

    (void) dst;
}

static void ggml_cuda_dup(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_cpy(src0, dst, nullptr);
    (void) src1;
}

static void ggml_cuda_diag_mask_inf(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_diag_mask_inf);
}

static void ggml_cuda_soft_max(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_soft_max);
}

static void ggml_cuda_rope(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0)); // TODO: this restriction is temporary until non-cont support is implemented
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_rope);
}

static void ggml_cuda_alibi(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_alibi);
}

static void ggml_cuda_nop(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    (void) src0;
    (void) src1;
    (void) dst;
}

void ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor) {
    const int64_t nrows = ggml_nrows(tensor);

    const int64_t ne0 = tensor->ne[0];

    const size_t nb1 = tensor->nb[1];

    ggml_backend_type backend = tensor->backend;
    ggml_tensor_extra_gpu * extra = new struct ggml_tensor_extra_gpu;
    memset(extra, 0, sizeof(*extra));

    for (int64_t id = 0; id < g_device_count; ++id) {
        if (backend == GGML_BACKEND_GPU && id != g_main_device) {
            continue;
        }

        ggml_cuda_set_device(id);

        int64_t row_low, row_high;
        if (backend == GGML_BACKEND_GPU) {
            row_low = 0;
            row_high = nrows;
        } else if (backend == GGML_BACKEND_GPU_SPLIT) {
            const int64_t rounding = get_row_rounding(tensor->type);

            row_low = id == 0 ? 0 : nrows*g_tensor_split[id];
            row_low -= row_low % rounding;

            if (id == g_device_count - 1) {
                row_high = nrows;
            } else {
                row_high = nrows*g_tensor_split[id + 1];
                row_high -= row_high % rounding;
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
            for (int64_t is = 0; is < MAX_STREAMS; ++is) {
                CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
            }
        }
    }

    tensor->extra = extra;
}

void ggml_cuda_free_data(struct ggml_tensor * tensor) {
    if (!tensor || (tensor->backend != GGML_BACKEND_GPU && tensor->backend != GGML_BACKEND_GPU_SPLIT) ) {
        return;
    }

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int64_t id = 0; id < g_device_count; ++id) {
        if (extra->data_device[id] != nullptr) {
            CUDA_CHECK(ggml_cuda_set_device(id));
            CUDA_CHECK(cudaFree(extra->data_device[id]));
        }

        for (int64_t is = 0; is < MAX_STREAMS; ++is) {
            if (extra->events[id][is] != nullptr) {
                CUDA_CHECK(ggml_cuda_set_device(id));
                CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
            }
        }
    }

    delete extra;
}

static ggml_tensor_extra_gpu * g_temp_tensor_extras = nullptr;
static size_t g_temp_tensor_extra_index = 0;

static ggml_tensor_extra_gpu * ggml_cuda_alloc_temp_tensor_extra() {
    if (g_temp_tensor_extras == nullptr) {
        g_temp_tensor_extras = new ggml_tensor_extra_gpu[GGML_MAX_NODES];
    }

    size_t alloc_index = g_temp_tensor_extra_index;
    g_temp_tensor_extra_index = (g_temp_tensor_extra_index + 1) % GGML_MAX_NODES;
    ggml_tensor_extra_gpu * extra = &g_temp_tensor_extras[alloc_index];
    memset(extra, 0, sizeof(*extra));

    return extra;
}

static void ggml_cuda_assign_buffers_impl(struct ggml_tensor * tensor, bool scratch, bool force_inplace, bool no_alloc) {
    if (scratch && g_scratch_size == 0) {
        return;
    }

    tensor->backend = GGML_BACKEND_GPU;

    // recursively assign CUDA buffers until a compute tensor is found
    if (tensor->src[0] != nullptr && tensor->src[0]->backend == GGML_BACKEND_CPU) {
        const ggml_op src0_op = tensor->src[0]->op;
        if (src0_op == GGML_OP_RESHAPE || src0_op == GGML_OP_TRANSPOSE || src0_op == GGML_OP_VIEW || src0_op == GGML_OP_PERMUTE) {
            ggml_cuda_assign_buffers_impl(tensor->src[0], scratch, force_inplace, no_alloc);
        }
    }
    if (tensor->op == GGML_OP_CPY && tensor->src[1]->backend == GGML_BACKEND_CPU) {
        ggml_cuda_assign_buffers_impl(tensor->src[1], scratch, force_inplace, no_alloc);
    }

    if (scratch && no_alloc) {
        return;
    }

    ggml_tensor_extra_gpu * extra;

    const bool inplace = (tensor->src[0] != nullptr && tensor->src[0]->data == tensor->data) ||
        tensor->op == GGML_OP_VIEW ||
        force_inplace;
    const size_t size = ggml_nbytes(tensor);

    CUDA_CHECK(ggml_cuda_set_device(g_main_device));
    if (inplace && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT)) {
        ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu * ) tensor->src[0]->extra;
        char * src0_ddc = (char *) src0_extra->data_device[g_main_device];
        size_t offset = 0;
        if (tensor->op == GGML_OP_VIEW) {
            memcpy(&offset, tensor->op_params, sizeof(size_t));
        }
        extra = ggml_cuda_alloc_temp_tensor_extra();
        extra->data_device[g_main_device] = src0_ddc + offset;
    } else if (tensor->op == GGML_OP_CPY) {
        ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu * ) tensor->src[1]->extra;
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

void ggml_cuda_assign_scratch_offset(struct ggml_tensor * tensor, size_t offset) {
    if (g_scratch_size == 0) {
        return;
    }
    if (g_scratch_buffer == nullptr) {
        ggml_cuda_set_device(g_main_device);
        CUDA_CHECK(cudaMalloc(&g_scratch_buffer, g_scratch_size));
    }

    ggml_tensor_extra_gpu * extra = ggml_cuda_alloc_temp_tensor_extra();

    const bool inplace = (tensor->src[0] != nullptr && tensor->src[0]->data == tensor->data) ||
        tensor->op == GGML_OP_VIEW;

    if (inplace && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT)) {
        ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu * ) tensor->src[0]->extra;
        char * src0_ddc = (char *) src0_extra->data_device[g_main_device];
        size_t view_offset = 0;
        if (tensor->op == GGML_OP_VIEW) {
            memcpy(&view_offset, tensor->op_params, sizeof(size_t));
        }
        extra->data_device[g_main_device] = src0_ddc + view_offset;
    } else {
        extra->data_device[g_main_device] = (char *) g_scratch_buffer + offset;
    }

    tensor->extra = extra;
}

void ggml_cuda_copy_to_device(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    CUDA_CHECK(ggml_cuda_set_device(g_main_device));
    CUDA_CHECK(cudaMemcpy(extra->data_device[g_main_device], tensor->data, ggml_nbytes(tensor), cudaMemcpyHostToDevice));
}

void ggml_cuda_assign_buffers(struct ggml_tensor * tensor) {
    ggml_cuda_assign_buffers_impl(tensor, true, false, false);
}

void ggml_cuda_assign_buffers_no_alloc(struct ggml_tensor * tensor) {
    ggml_cuda_assign_buffers_impl(tensor, true, false, true);
}

void ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor * tensor) {
    ggml_cuda_assign_buffers_impl(tensor, false, false, false);
}

void ggml_cuda_assign_buffers_force_inplace(struct ggml_tensor * tensor) {
    ggml_cuda_assign_buffers_impl(tensor, false, true, false);
}

void ggml_cuda_set_main_device(const int main_device) {
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

void ggml_cuda_set_mul_mat_q(const bool mul_mat_q) {
    g_mul_mat_q = mul_mat_q;
}

void ggml_cuda_set_scratch_size(const size_t scratch_size) {
    // this is a hack to not completely break llama.cpp when using multiple models or contexts simultaneously
    // it still won't always work as expected, but it's better than nothing
    if (scratch_size > g_scratch_size) {
        ggml_cuda_free_scratch();
    }
    g_scratch_size = std::max(g_scratch_size, scratch_size);
}

void ggml_cuda_free_scratch() {
    if (g_scratch_buffer == nullptr) {
        return;
    }

    CUDA_CHECK(cudaFree(g_scratch_buffer));
    g_scratch_buffer = nullptr;
}

bool ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    ggml_cuda_func_t func;
    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU
        || (tensor->src[0] != nullptr && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (tensor->src[1] != nullptr && tensor->src[1]->backend == GGML_BACKEND_GPU);

    if (!any_on_device && tensor->op != GGML_OP_MUL_MAT) {
        return false;
    }

    switch (tensor->op) {
        case GGML_OP_REPEAT:
            func = ggml_cuda_repeat;
            break;
        case GGML_OP_GET_ROWS:
            func = ggml_cuda_get_rows;
            break;
        case GGML_OP_DUP:
            func = ggml_cuda_dup;
            break;
        case GGML_OP_ADD:
            func = ggml_cuda_add;
            break;
        case GGML_OP_MUL:
            func = ggml_cuda_mul;
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    func = ggml_cuda_gelu;
                    break;
                case GGML_UNARY_OP_SILU:
                    func = ggml_cuda_silu;
                    break;
                default:
                    return false;
            } break;
        case GGML_OP_NORM:
            func = ggml_cuda_norm;
            break;
        case GGML_OP_RMS_NORM:
            func = ggml_cuda_rms_norm;
            break;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cuda_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cuda_mul_mat;
            break;
        case GGML_OP_SCALE:
            func = ggml_cuda_scale;
            break;
        case GGML_OP_CLAMP:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_clamp;
            break;
        case GGML_OP_CPY:
            func = ggml_cuda_cpy;
            break;
        case GGML_OP_CONT:
            func = ggml_cuda_dup;
            break;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            func = ggml_cuda_nop;
            break;
        case GGML_OP_DIAG_MASK_INF:
            func = ggml_cuda_diag_mask_inf;
            break;
        case GGML_OP_SOFT_MAX:
            func = ggml_cuda_soft_max;
            break;
        case GGML_OP_ROPE:
            func = ggml_cuda_rope;
            break;
        case GGML_OP_ALIBI:
            func = ggml_cuda_alibi;
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

int ggml_cuda_get_device_count() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    return device_count;
}

void ggml_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

#define UNUSED GGML_UNUSED

struct ggml_backend_context_cuda {
};

static const char * ggml_backend_cuda_name(ggml_backend_t backend) {
    return GGML_CUDA_NAME;

    UNUSED(backend);
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_context_cuda * cuda_ctx = (ggml_backend_context_cuda *)backend->context;
    delete cuda_ctx;
    delete backend;
}

struct ggml_backend_buffer_context_cuda {
    void * device;

    ggml_tensor_extra_gpu * temp_tensor_extras = nullptr;
    size_t temp_tensor_extra_index = 0;

    ~ggml_backend_buffer_context_cuda() {
        delete[] temp_tensor_extras;
    }

    ggml_tensor_extra_gpu * ggml_cuda_alloc_temp_tensor_extra() {
        if (temp_tensor_extras == nullptr) {
            temp_tensor_extras = new ggml_tensor_extra_gpu[GGML_MAX_NODES];
        }

        size_t alloc_index = temp_tensor_extra_index;
        temp_tensor_extra_index = (temp_tensor_extra_index + 1) % GGML_MAX_NODES;
        ggml_tensor_extra_gpu * extra = &temp_tensor_extras[alloc_index];
        memset(extra, 0, sizeof(*extra));

        return extra;
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_buffer_context_cuda * ctx = (ggml_backend_buffer_context_cuda *)buffer->context;
    CUDA_CHECK(cudaFree(ctx->device));
    delete ctx;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_buffer_context_cuda * ctx = (ggml_backend_buffer_context_cuda *)buffer->context;
    return ctx->device;
}

static size_t ggml_backend_cuda_buffer_get_alloc_size(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    int64_t row_low = 0;
    int64_t row_high = ggml_nrows(tensor);
    int64_t nrows_split = row_high - row_low;

    size_t size = ggml_nbytes_split(tensor, nrows_split);

    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += (MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)
                * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
        }
    }

    return size;

    UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_buffer_context_cuda * ctx = (ggml_backend_buffer_context_cuda *)buffer->context;

    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        assert(tensor->view_src->buffer->backend == buffer->backend);
        tensor->backend = tensor->view_src->backend;
        tensor->extra = tensor->view_src->extra;
        return;
    }

    ggml_tensor_extra_gpu * extra = ctx->ggml_cuda_alloc_temp_tensor_extra();

    extra->data_device[g_main_device] = tensor->data;

    tensor->backend = GGML_BACKEND_GPU;
    tensor->extra = extra;

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        int64_t row_low = 0;
        int64_t row_high = ggml_nrows(tensor);
        int64_t nrows_split = row_high - row_low;

        size_t original_size = ggml_nbytes_split(tensor, nrows_split);
        size_t padded_size = ggml_backend_cuda_buffer_get_alloc_size(tensor->buffer, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, g_cudaStreams[g_main_device][0]));
        }
    }

    UNUSED(buffer);
}

static struct ggml_backend_buffer_i cuda_backend_buffer_interface = {
    /* .free_buffer    = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base       = */ ggml_backend_cuda_buffer_get_base,
    /* .get_alloc_size = */ ggml_backend_cuda_buffer_get_alloc_size,
    /* .init_tensor    = */ ggml_backend_cuda_buffer_init_tensor,
    /* .free_tensor    = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_cuda_alloc_buffer(ggml_backend_t backend, size_t size) {
    ggml_cuda_set_device(g_main_device);

    ggml_backend_buffer_context_cuda * ctx = new ggml_backend_buffer_context_cuda;
    CUDA_CHECK(cudaMalloc(&ctx->device, size));
    return ggml_backend_buffer_init(backend, cuda_backend_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_get_alignment(ggml_backend_t backend) {
    return 128;
    UNUSED(backend);
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);

    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, g_cudaStreams[g_main_device][0]));

    UNUSED(backend);
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, g_cudaStreams[g_main_device][0]));

    UNUSED(backend);
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    CUDA_CHECK(cudaStreamSynchronize(g_cudaStreams[g_main_device][0]));

    UNUSED(backend);
}

static ggml_backend_graph_plan_t ggml_backend_cuda_graph_plan_create(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_ASSERT(!"not implemented");

    return nullptr;

    UNUSED(backend);
    UNUSED(cgraph);
}

static void ggml_backend_cuda_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(!"not implemented");

    UNUSED(backend);
    UNUSED(plan);
}

static void ggml_backend_cuda_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(!"not implemented");

    UNUSED(backend);
    UNUSED(plan);
}

static void ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_cuda_set_device(g_main_device);

    ggml_compute_params params = {};
    params.type = GGML_TASK_COMPUTE;
    params.ith = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        assert(node->backend == GGML_BACKEND_GPU);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->backend == GGML_BACKEND_GPU);
            }
        }

        bool ok = ggml_cuda_compute_forward(&params, node);
        if (!ok) {
            fprintf(stderr, "%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);

#if 0
        if (node->type == GGML_TYPE_F32) {
            cudaDeviceSynchronize();
            std::vector<float> tmp(ggml_nelements(node), 0.0f);
            cudaMemcpy(tmp.data(), node->data, ggml_nelements(node)*sizeof(float), cudaMemcpyDeviceToHost);
            printf("\n%s (%s) (%s %s) (%s %s): ", node->name, ggml_op_name(node->op),
                ggml_type_name(node->src[0]->type),
                node->src[1] ? ggml_type_name(node->src[1]->type) : "none",
                node->src[0]->name,
                node->src[1] ? node->src[1]->name : "none");
            double sum = 0.0;
            double sq_sum = 0.0;
            for (int i = 0; i < ggml_nelements(node); i++) {
                printf("%f ", tmp[i]);
                sum += tmp[i];
                sq_sum += tmp[i]*tmp[i];
            }
            printf("\n");
            printf("sum: %f, ", sum);
            printf("sq_sum: %f\n", sq_sum);
        }
#endif
    }

    UNUSED(backend);
}

static ggml_backend_i cuda_backend_i = {
    /* .get_name            = */ ggml_backend_cuda_name,
    /* .free                = */ ggml_backend_cuda_free,
    /* .alloc_buffer        = */ ggml_backend_cuda_alloc_buffer,
    /* .get_alignment       = */ ggml_backend_cuda_get_alignment,
    /* .set_tensor_async    = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_cuda_get_tensor_async,
    /* .synchronize         = */ ggml_backend_cuda_synchronize,
    /* .cpy_tensor_from     = */ nullptr,
    /* .cpy_tensor_to       = */ nullptr,
    /* .graph_plan_create   = */ ggml_backend_cuda_graph_plan_create,
    /* .graph_plan_free     = */ ggml_backend_cuda_graph_plan_free,
    /* .graph_plan_compute  = */ ggml_backend_cuda_graph_plan_compute,
    /* .graph_compute       = */ ggml_backend_cuda_graph_compute,
    /* .supports_op         = */ nullptr,
};

ggml_backend_t ggml_backend_cuda_init() {
    ggml_init_cublas(); // TODO: remove from ggml.c

    ggml_backend_context_cuda * ctx = new ggml_backend_context_cuda;

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .interface = */ cuda_backend_i,
        /* .context   = */ ctx
    };

    return cuda_backend;
}
