#pragma once

#include "ggml.h"
#include "ggml-cuda.h"

#include <memory>

#if defined(GGML_USE_HIPBLAS)
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#endif
#include "ggml-common.h"

#include <cstdio>
#include <array>
#include <cassert>
#include <cfloat>
#include <string>
#include <vector>

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
#define cublasComputeType_t hipblasDatatype_t //deprecated, new hipblasComputeType_t not in 5.6
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasStatus_t hipblasStatus_t
#define cudaDataType_t hipblasDatatype_t //deprecated, new hipblasDatatype not in 5.6
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventDestroy hipEventDestroy
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#ifdef GGML_HIP_UMA
#define cudaMalloc hipMallocManaged
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size)
#else
#define cudaMalloc hipMalloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#endif
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaSetDevice hipSetDevice
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamFireAndForget hipStreamFireAndForget
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamPerThread hipStreamPerThread
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent(stream, event, flags) hipStreamWaitEvent(stream, event, flags)
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define __trap abort
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#else
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020

#endif // defined(GGML_USE_HIPBLAS)

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
#define CUDART_HMAX   11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)
#define CUDART_HMASK  12000 // CUDA 12.0, min. ver. for half2 -> uint mask comparisons

#define CC_PASCAL     600
#define MIN_CC_DP4A   610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define CC_VOLTA      700
#define CC_AMPERE     800
#define CC_OFFSET_AMD 1000000
#define CC_RDNA1      (CC_OFFSET_AMD + 1010)
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)
#define CC_RDNA3      (CC_OFFSET_AMD + 1100)

// define this if you want to always fallback to MMQ kernels and not use cuBLAS for matrix multiplication
// on modern hardware, using cuBLAS is recommended as it utilizes F16 tensor cores which are very performant
// for large computational tasks. the drawback is that this requires some extra amount of VRAM:
// -  7B quantum model: +100-200 MB
// - 13B quantum model: +200-400 MB
//
//#define GGML_CUDA_FORCE_MMQ

// TODO: improve this to be correct for more hardware
//       for example, currently fails for GeForce GTX 1660 which is TURING arch (> VOLTA) but does not have tensor cores
#if !defined(GGML_CUDA_FORCE_MMQ)
#define CUDA_USE_TENSOR_CORES
#endif

#define MMVQ_MAX_BATCH_SIZE  8 // max batch size to use MMVQ kernels
#define  MMQ_MAX_BATCH_SIZE 32 // max batch size to use MMQ kernels when tensor cores are available

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define GGML_CUDA_MAX_STREAMS 8

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

#define CUDA_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)

#if CUDART_VERSION >= 12000
    static const char * cublas_get_error_str(const cublasStatus_t err) {
        return cublasGetStatusString(err);
    }
#else
    static const char * cublas_get_error_str(const cublasStatus_t err) {
        switch (err) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            default: return "unknown error";
        }
    }
#endif // CUDART_VERSION >= 12000

#define CUBLAS_CHECK(err) CUDA_CHECK_GEN(err, CUBLAS_STATUS_SUCCESS, cublas_get_error_str)

#if !defined(GGML_USE_HIPBLAS)
static const char * cu_get_error_str(CUresult err) {
    const char * err_str;
    cuGetErrorString(err, &err_str);
    return err_str;
}
#define CU_CHECK(err) CUDA_CHECK_GEN(err, CUDA_SUCCESS, cu_get_error_str)
#endif

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
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
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
    return reinterpret_cast<int &>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;
}

static __device__ __forceinline__ int __dp4a(const int a, const int b, int c) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx1030__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3)
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

#if defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 50600000
// __shfl_xor() for half2 was added in ROCm 5.6
static __device__ __forceinline__ half2 __shfl_xor(half2 var, int laneMask, int width) {
    typedef union half2_b32 {
        half2 val;
        int   b32;
    } half2_b32_t;
    half2_b32_t tmp;
    tmp.val = var;
    tmp.b32 = __shfl_xor(tmp.b32, laneMask, width);
    return tmp.val;
}
#endif // defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 50600000
#endif // defined(GGML_USE_HIPBLAS)

#define FP16_AVAILABLE (defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ >= CC_PASCAL

#define FP16_MMA_AVAILABLE !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_VOLTA

static bool fast_fp16_available(const int cc) {
    return cc >= CC_PASCAL && cc != 610;
}

static bool fp16_mma_available(const int cc) {
    return cc < CC_OFFSET_AMD && cc >= CC_VOLTA;
}

[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
           file_name, line, function_name, arch);
    GGML_UNUSED(arch_list);
#else
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ASSERT(false && "NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
        a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
    }
    return a;
}

static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#if FP16_AVAILABLE

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        const half2 a_other = __shfl_xor_sync(0xffffffff, a, mask, 32);
        reinterpret_cast<half&>(a.x) +=  __low2half(a_other);
        reinterpret_cast<half&>(a.y) += __high2half(a_other);
    }
    return a;
#else
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, mask, 32));
    }
    return a;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

static __device__ __forceinline__ half ggml_cuda_hmax(const half a, const half b) {
#if FP16_AVAILABLE

#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && CUDART_VERSION < CUDART_HMAX
    return __float2half(fmaxf(__half2float(a), __half2float(b)));
#else
    return __hmax(a, b);
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && CUDART_VERSION < CUDART_HMAX

#else
   NO_DEVICE_CODE;
   GGML_UNUSED(b);
   return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half2 ggml_cuda_hmax2(const half2 a, const half2 b) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))

#if CUDART_VERSION >= CUDART_HMAX
    return __hmax2(a, b);
#else
    half2 ret;
    reinterpret_cast<half&>(ret.x) = __float2half(fmaxf( __low2float(a),  __low2float(b)));
    reinterpret_cast<half&>(ret.y) = __float2half(fmaxf(__high2float(a), __high2float(b)));
    return ret;
#endif // CUDART_VERSION >= CUDART_HMAX

#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
}

static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
#pragma unroll
   for (int mask = 16; mask > 0; mask >>= 1) {
       x = ggml_cuda_hmax2(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
   }
   return x;
#else
   GGML_UNUSED(x);
   NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
}

#if CUDART_VERSION < CUDART_HMASK
static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float( __low2half(a)) > float( __low2half(b)));
    const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
    return mask_low | mask_high;
}
#endif // CUDART_VERSION < 12000

// TODO: move to ggml-common.h
static const __device__ int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, dfloat2 & v);

static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return powf(base, exph);
}

//////////////////////

struct ggml_cuda_device_info {
    int device_count;

    struct cuda_device_info {
        int     cc;                 // compute capability
        int     nsm;                // number of streaming multiprocessors
        size_t  smpb;               // max. shared memory per block
        bool    vmm;                // virtual memory support
        size_t  vmm_granularity;    // granularity of virtual memory
        size_t  total_vram;
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

    std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};
};

const ggml_cuda_device_info & ggml_cuda_info();

void ggml_cuda_set_device(int device);
int ggml_cuda_get_device();

struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool & pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_cuda_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc &&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc &&) = delete;
};


// backend interface

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]; // events for synchronizing multiple GPUs
};


#if (CUDART_VERSION >= 12000) && defined(GGML_CUDA_USE_GRAPHS)
#define USE_CUDA_GRAPH
#endif

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op node_op;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
};

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
    int number_consecutive_updates = 0;
    std::vector<ggml_graph_node_properties> ggml_graph_properties;
    std::vector<char **> updated_kernel_arg;
#endif
};

struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaEvent_t copy_event = nullptr;

    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

    std::unique_ptr<ggml_cuda_graph> cuda_graph;

    explicit ggml_backend_cuda_context(int device) :
        device(device),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_context() {
        if (copy_event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(copy_event));
        }
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
                if (streams[i][j] != nullptr) {
                    CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
                }
            }
            if (cublas_handles[i] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
            }
        }
    }

    cudaStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
        }
        return streams[device][stream];
    }

    cudaStream_t stream() {
        return stream(device, 0);
    }

    cublasHandle_t cublas_handle(int device) {
        if (cublas_handles[device] == nullptr) {
            ggml_cuda_set_device(device);
            CUBLAS_CHECK(cublasCreate(&cublas_handles[device]));
            CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH));
        }
        return cublas_handles[device];
    }

    cublasHandle_t cublas_handle() {
        return cublas_handle(device);
    }

    // pool
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES];

    static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device);

    ggml_cuda_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(device);
        }
        return *pools[device];
    }

    ggml_cuda_pool & pool() {
        return pool(device);
    }
};
