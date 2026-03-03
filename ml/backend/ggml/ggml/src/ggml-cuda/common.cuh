#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <memory>

#if defined(GGML_USE_HIP)
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#if defined(GGML_USE_MUSA)
#define GGML_COMMON_DECL_MUSA
#define GGML_COMMON_IMPL_MUSA
#endif
#endif
#include "ggml-common.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#elif defined(GGML_USE_MUSA)
#include "vendors/musa.h"
#else
#include "vendors/cuda.h"
#endif // defined(GGML_USE_HIP)

extern bool reserving_graph;

// If we are reserving the graph, pointers might be invalid and will fail if cudaMemcpyAsync tries to validate them.
// However, since we don't actually expect a result, we don't need to actually do the memcpy.
static cudaError_t cudaMemcpyAsyncReserve ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 ) {
    if (!reserving_graph) {
        return cudaMemcpyAsync(dst, src, count, kind, stream);
    } else {
        return cudaSuccess;
    }
}

static cudaError_t cudaMemcpy2DAsyncReserve ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 ) {
    if (!reserving_graph) {
        return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
    } else {
        return cudaSuccess;
    }
}

static cudaError_t cudaMemsetAsyncReserve ( void* devPtr, int value, size_t count, cudaStream_t stream = 0 ) {
    if (!reserving_graph) {
        return cudaMemsetAsync(devPtr, value, count, stream);
    } else {
        return cudaSuccess;
    }
}

#undef cudaMemcpyAsync
#define cudaMemcpyAsync cudaMemcpyAsyncReserve
#undef cudaMemcpy2DAsync
#define cudaMemcpy2DAsync cudaMemcpy2DAsyncReserve
#undef cudaMemsetAsync
#define cudaMemsetAsync cudaMemsetAsyncReserve

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
#define CUDART_HMAX   11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)
#define CUDART_HMASK  12000 // CUDA 12.0, min. ver. for half2 -> uint mask comparisons

#define GGML_CUDA_CC_PASCAL          600
#define GGML_CUDA_CC_DP4A            610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_CUDA_CC_VOLTA           700
#define GGML_CUDA_CC_TURING          750
#define GGML_CUDA_CC_AMPERE          800
#define GGML_CUDA_CC_ADA_LOVELACE    890
#define GGML_CUDA_CC_OFFSET_AMD      0x1000000
#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_IS_NVIDIA(cc)   (cc < GGML_CUDA_CC_OFFSET_MTHREADS)

// AMD
// GCN/CDNA, wave size is 64
#define GGML_CUDA_CC_GCN4       (GGML_CUDA_CC_OFFSET_AMD + 0x803)  // Tonga, Fiji, Polaris, minimum for fast fp16
#define GGML_CUDA_CC_VEGA       (GGML_CUDA_CC_OFFSET_AMD + 0x900)  // Vega56/64, minimum for fp16 dual issue
#define GGML_CUDA_CC_VEGA20     (GGML_CUDA_CC_OFFSET_AMD + 0x906)  // MI50/Radeon VII, minimum for dp4a
#define GGML_CUDA_CC_CDNA1      (GGML_CUDA_CC_OFFSET_AMD + 0x908)  // MI100, minimum for MFMA, acc registers
#define GGML_CUDA_CC_CDNA2      (GGML_CUDA_CC_OFFSET_AMD + 0x910)  // MI210, minimum acc register renameing
#define GGML_CUDA_CC_CDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x942)  // MI300

// RDNA removes MFMA, dp4a, xnack, acc registers, wave size is 32
#define GGML_CUDA_CC_RDNA1      (GGML_CUDA_CC_OFFSET_AMD + 0x1010) // RX 5000
#define GGML_CUDA_CC_RDNA2      (GGML_CUDA_CC_OFFSET_AMD + 0x1030) // RX 6000, minimum for dp4a
#define GGML_CUDA_CC_RDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x1100) // RX 7000, minimum for WMMA
#define GGML_CUDA_CC_RDNA3_5    (GGML_CUDA_CC_OFFSET_AMD + 0x1150) // AI 370, AI Max 395 laptops.
#define GGML_CUDA_CC_RDNA4      (GGML_CUDA_CC_OFFSET_AMD + 0x1200) // RX 9000

#define GGML_CUDA_CC_IS_AMD(cc)     (cc >= GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_RDNA(cc)    (cc >= GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_RDNA1(cc)   (cc >= GGML_CUDA_CC_RDNA1 && cc < GGML_CUDA_CC_RDNA2)
#define GGML_CUDA_CC_IS_RDNA2(cc)   (cc >= GGML_CUDA_CC_RDNA2 && cc < GGML_CUDA_CC_RDNA3)
#define GGML_CUDA_CC_IS_RDNA3_0(cc) (cc >= GGML_CUDA_CC_RDNA3 && cc < GGML_CUDA_CC_RDNA3_5)
#define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_RDNA3(cc)   (GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc))
#define GGML_CUDA_CC_IS_RDNA4(cc)   (cc >= GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_GCN(cc)     (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_CDNA1)
#define GGML_CUDA_CC_IS_CDNA(cc)    (cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_CDNA1(cc)   (cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_CDNA2)
#define GGML_CUDA_CC_IS_CDNA2(cc)   (cc >= GGML_CUDA_CC_CDNA2 && cc < GGML_CUDA_CC_CDNA3)
#define GGML_CUDA_CC_IS_CDNA3(cc)   (cc >= GGML_CUDA_CC_CDNA3 && cc < GGML_CUDA_CC_RDNA1)

// Moore Threads
#define MUSART_HMASK 40300 // MUSA rc4.3, min. ver. for half2 -> uint mask comparisons

#define GGML_CUDA_CC_QY1 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x210) // MTT S80, MTT S3000
#define GGML_CUDA_CC_QY2 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x220) // MTT S4000
#define GGML_CUDA_CC_PH1 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x310) // MTT S5000

#define GGML_CUDA_CC_IS_MTHREADS(cc) (cc >= GGML_CUDA_CC_OFFSET_MTHREADS && cc < GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_QY1(cc)      (cc >= GGML_CUDA_CC_QY1 && cc < GGML_CUDA_CC_QY2)
#define GGML_CUDA_CC_IS_QY2(cc)      (cc >= GGML_CUDA_CC_QY2 && cc < GGML_CUDA_CC_PH1)
#define GGML_CUDA_CC_IS_PH1(cc)      (cc >= GGML_CUDA_CC_PH1)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#    define GGML_CUDA_USE_CUB
#endif  // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070

#ifdef __CUDA_ARCH_LIST__
constexpr bool ggml_cuda_has_arch_impl(int) {
    return false;
}

template<class ... Archs>
constexpr bool ggml_cuda_has_arch_impl(const int arch, const int first, Archs... rest) {
    return arch == first || ggml_cuda_has_arch_impl(arch, rest...);
}

constexpr bool ggml_cuda_has_arch(const int arch) {
    return ggml_cuda_has_arch_impl(arch, __CUDA_ARCH_LIST__);
}

constexpr int ggml_cuda_highest_compiled_arch_impl(const int /*arch*/, const int cur) {
    if (cur == 0) {
        return -1;
    }
    return cur;
}

template<class ... Archs>
constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur, const int first, Archs... rest) {
    if (first <= arch && first > cur) {
        return ggml_cuda_highest_compiled_arch_impl(arch, first, rest...);
    } else {
        return ggml_cuda_highest_compiled_arch_impl(arch, cur, rest...);
    }
}

constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    return ggml_cuda_highest_compiled_arch_impl(arch, 0, __CUDA_ARCH_LIST__);
}
#else
static int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}
#endif // __CUDA_ARCH_LIST__

// ---------------------------------------------------------------------------------------------------------

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

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

#if CUDART_VERSION >= 12000 || defined(GGML_USE_MUSA)
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

#if !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
static const char * cu_get_error_str(CUresult err) {
    const char * err_str;
    cuGetErrorString(err, &err_str);
    return err_str;
}
#define CU_CHECK(err) CUDA_CHECK_GEN(err, CUDA_SUCCESS, cu_get_error_str)
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                                       \
        do {                                                                                                   \
            static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };                         \
            const int   id                                                = ggml_cuda_get_device();            \
            if (!shared_memory_limit_raised[id]) {                                                             \
                CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes)); \
                shared_memory_limit_raised[id] = true;                                                         \
            }                                                                                                  \
        } while (0)
#else
#    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes) \
        do {                                             \
            GGML_UNUSED(nbytes);                         \
        } while (0)
#endif // !(defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif // CUDART_VERSION >= 11010

#if (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))
#define GGML_USE_VMM
#endif // (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL

#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif // defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610

#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
#define AMD_MFMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)

#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
#define AMD_WMMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(RDNA4)

// The Volta instructions are in principle available on Turing or newer but they are effectively unusable:
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if !defined(GGML_CUDA_NO_FA) && !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ < 220)
#define FLASH_ATTN_AVAILABLE
#endif // !defined(GGML_CUDA_NO_FA) && !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ < 220)

static bool fp16_available(const int cc) {
    return ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_PASCAL ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_PH1);
}

static bool fast_fp16_available(const int cc) {
    return GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_NVIDIA(cc) && fp16_available(cc) && ggml_cuda_highest_compiled_arch(cc) != 610) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && fp16_available(cc));
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fast_fp16_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc != 610) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fp16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

static bool bf16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE) ||
        GGML_CUDA_CC_IS_CDNA(cc) || cc >= GGML_CUDA_CC_RDNA3 ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_PH1);
}

static bool fp32_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_CDNA(cc);
}

static bool amd_mfma_available(const int cc) {
#if !defined(GGML_HIP_NO_MMQ_MFMA)
    return GGML_CUDA_CC_IS_CDNA(cc);
#else
    return false;
#endif //!defined(GGML_HIP_NO_MMQ_MFMA)
}

static bool amd_wmma_available(const int cc) {
    return (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));
}

static bool volta_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;
}

static bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

static bool ampere_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

static bool cp_async_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
    return 64;
#else
    return 32;
#endif // defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#ifdef GGML_USE_HIP
    return 16;
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // GGML_USE_HIP
}


[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {

#if defined(GGML_USE_HIP)
    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
           file_name, line, function_name, arch);
    GGML_UNUSED(arch_list);
#else
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
#endif // defined(GGML_USE_HIP)
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning

#if defined(GGML_USE_MUSA)
    __builtin_unreachable();
#endif // defined(GGML_USE_MUSA)
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

// The compiler is always able to unroll loops if they contain continue expressions.
// In such cases loop unrolling can still be achieved via recursion:
template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    return __reduce_add_sync(0xffffffff, x);
#else
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, offset, width);
        a.y += __shfl_xor_sync(0xffffffff, a.y, offset, width);
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#ifdef FP16_AVAILABLE
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, offset, width));
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_all(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __all_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) && x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __any_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

template<typename T, int width = WARP_SIZE>
static __device__ __forceinline__ T warp_prefix_inclusive_sum(T x) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const T t = __shfl_up_sync(0xffffffff, x, offset, width);
        if (lane_id >= offset) {
            x += t;
        }
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_prefix_inclusive_sum(float2 a) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const float t_x = __shfl_up_sync(0xffffffff, a.x, offset, width);
        const float t_y = __shfl_up_sync(0xffffffff, a.y, offset, width);
        if (lane_id >= offset) {
            a.x += t_x;
            a.y += t_y;
        }
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_prefix_inclusive_sum(half2 a) {
#ifdef FP16_AVAILABLE
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const half2 t = __shfl_up_sync(0xffffffff, a, offset, width);
        if (lane_id >= offset) {
            a = __hadd2(a, t);
        }
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half ggml_cuda_hmax(const half a, const half b) {
#ifdef FP16_AVAILABLE

#if !defined(GGML_USE_HIP) && CUDART_VERSION < CUDART_HMAX
    return __float2half(fmaxf(__half2float(a), __half2float(b)));
#else
    return __hmax(a, b);
#endif // !defined(GGML_USE_HIP) && CUDART_VERSION < CUDART_HMAX

#else
   NO_DEVICE_CODE;
   GGML_UNUSED(b);
   return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half2 ggml_cuda_hmax2(const half2 a, const half2 b) {
#if defined(GGML_USE_HIP)
    return half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
#elif CUDART_VERSION >= CUDART_HMAX
    return __hmax2(a, b);
#else
    half2 ret;
    reinterpret_cast<half&>(ret.x) = __float2half(fmaxf( __low2float(a),  __low2float(b)));
    reinterpret_cast<half&>(ret.y) = __float2half(fmaxf(__high2float(a), __high2float(b)));
    return ret;
#endif
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL || defined(GGML_USE_HIP)
#pragma unroll
   for (int offset = width/2; offset > 0; offset >>= 1) {
       x = ggml_cuda_hmax2(x, __shfl_xor_sync(0xffffffff, x, offset, width));
   }
   return x;
#else
   GGML_UNUSED(x);
   NO_DEVICE_CODE;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL || defined(GGML_USE_HIP)
}

#if (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || defined(GGML_USE_HIP) || \
    (defined(MUSART_VERSION) && MUSART_VERSION < MUSART_HMASK)
static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float( __low2half(a)) > float( __low2half(b)));
    const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
    return mask_low | mask_high;
}
#endif // (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || defined(GGML_USE_HIP) || (defined(MUSART_VERSION) && MUSART_VERSION < MUSART_HMASK)

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx900__)
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

#else // defined(GGML_USE_HIP)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)

#endif // defined(GGML_USE_HIP)
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x*u.x;
    acc += v.y*u.y;
}

#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))
#define V_DOT2_F32_F16_AVAILABLE
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const half2 v, const half2 u) {
#ifdef V_DOT2_F32_F16_AVAILABLE
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(v*u);
    acc += tmp.x + tmp.y;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x * tmpu.x;
    acc += tmpv.y * tmpu.y;
#endif // FAST_FP16_AVAILABLE
#endif // V_DOT2_F32_F16_AVAILABLE
}

static __device__ __forceinline__ void ggml_cuda_mad(half2 & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc += v*u;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst have different types.
//     The function is intended for copies between registers and SRAM/VRAM to make the compiler emit the right instructions.
//     If dst and src point at different address spaces then they are guaranteed to not be aliased.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
        "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
        "If you use it to do more bytes per copy than ggml_cuda_max_cpy_bytes() the reads and writes may not be coalesced. "
        "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float) e;
#else
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
#endif // CUDART_VERSION >= 12050
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, float2 & v);

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

template <ggml_type type>
struct ggml_cuda_type_traits;

template<>
struct ggml_cuda_type_traits<GGML_TYPE_F16> {
    static constexpr int qk = 1;
    static constexpr int qr = 1;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;
    static constexpr int qr = QR4_0;
    static constexpr int qi = QI4_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_1> {
    static constexpr int qk = QK4_1;
    static constexpr int qr = QR4_1;
    static constexpr int qi = QI4_1;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;
    static constexpr int qr = QR5_0;
    static constexpr int qi = QI5_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1;
    static constexpr int qr = QR5_1;
    static constexpr int qi = QI5_1;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR8_0;
    static constexpr int qi = QI8_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_MXFP4> {
    static constexpr int qk = QK_MXFP4;
    static constexpr int qr = QR_MXFP4;
    static constexpr int qi = QI_MXFP4;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_K;
    static constexpr int qi = QI2_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_K;
    static constexpr int qi = QI3_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XXS;
    static constexpr int qi = QI2_XXS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XS;
    static constexpr int qi = QI2_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_S;
    static constexpr int qi = QI2_S;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_XXS;
    static constexpr int qi = QI3_XXS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_S;
    static constexpr int qi = QI1_S;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_M> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_M;
    static constexpr int qi = QI1_M;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_NL> {
    static constexpr int qk = QK4_NL;
    static constexpr int qr = QR4_NL;
    static constexpr int qi = QI4_NL;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_S;
    static constexpr int qi = QI3_S;
};

//////////////////////

struct ggml_cuda_device_info {
    int device_count;

    struct cuda_device_info {
        int     cc;                 // compute capability
        int     nsm;                // number of streaming multiprocessors
        size_t  smpb;               // max. shared memory per block
        size_t  smpbo;              // max. shared memory per block (with opt-in)
        bool    integrated;         // Device is integrated as opposed to discrete
        bool    vmm;                // virtual memory support
        size_t  vmm_granularity;    // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;          // Number of threads in a dispatch
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

    virtual bool alloc_memory() = 0;
    virtual size_t alloc_size() = 0;
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


#if (defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS)) || defined(GGML_MUSA_GRAPHS)
#define USE_CUDA_GRAPH
#endif

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op node_op;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
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
#endif
};

struct ggml_cuda_concurrent_event {
    std::vector<cudaEvent_t> join_events;
    cudaEvent_t              fork_event = nullptr;

    int                                          n_streams = 0;
    std::unordered_map<const ggml_tensor *, int> stream_mapping;

    // Original order of nodes in this concurrent region (before interleaving)
    // Used to restore grouping for fusion within streams
    std::vector<const ggml_tensor *> original_order;

    const ggml_tensor * join_node;

    ggml_cuda_concurrent_event() = default;

    ggml_cuda_concurrent_event(const ggml_cuda_concurrent_event &) = delete;
    ggml_cuda_concurrent_event & operator=(const ggml_cuda_concurrent_event &) = delete;

    explicit ggml_cuda_concurrent_event(int n_streams) : n_streams(n_streams) {
        join_events.resize(n_streams);

        for (size_t i = 0; i < join_events.size(); ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&join_events[i], cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
    }

    ggml_cuda_concurrent_event(ggml_cuda_concurrent_event && other) noexcept
    : join_events(std::move(other.join_events))
    , fork_event(other.fork_event)
    , n_streams(other.n_streams)
    , stream_mapping(std::move(other.stream_mapping))
    , original_order(std::move(other.original_order))
    , join_node(other.join_node) {
        other.fork_event = nullptr;
    }

    // 1. check if any branches write to overlapping memory ranges (except the join node)
    // 2. check whether all srcs are either within the branch or outside the nodes covered by ggml_cuda_concurrent_event
    // we assume all nodes have the same buffer
    bool is_valid() const {
        std::vector<std::vector<std::pair<int64_t, int64_t>>> write_ranges;
        write_ranges.resize(n_streams);

        // get join_node's memory range to exclude from overlap checking.
        // multiple nodes can use join_node's buffer; we synchronize on the join node.
        const ggml_tensor * join_t     = join_node->view_src ? join_node->view_src : join_node;
        const int64_t       join_start = (int64_t) join_t->data;
        const int64_t       join_end   = join_start + ggml_nbytes(join_t);

        for (const auto & [tensor, stream] : stream_mapping) {
            const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t) t->data;
            const int64_t       t_end   = t_start + ggml_nbytes(t);

            // skip tensors that overlap with join_node's buffer.
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // concurrent streams begin from 1
            write_ranges[stream - 1].emplace_back(t_start, t_end);
        }

        for (int i = 0; i < n_streams; ++i) {
            // sorts first by start then by end of write range
            std::sort(write_ranges[i].begin(), write_ranges[i].end());
        }

        bool writes_overlap = false;
        bool dependent_srcs = false;
        for (const auto & [tensor, stream] : stream_mapping) {
            const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t) t->data;
            const int64_t       t_end   = t_start + ggml_nbytes(t);

            // skip tensors that overlap with join_node's buffer
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // check if this buffer's write data overlaps with another stream's
            std::pair<int64_t, int64_t> data_range = std::make_pair(t_start, t_end);
            for (int i = 0; i < n_streams; ++i) {
                if (i == stream - 1) {
                    continue;
                }
                auto it = std::lower_bound(write_ranges[i].begin(), write_ranges[i].end(), data_range);

                if (it != write_ranges[i].end()) {
                    const std::pair<int64_t, int64_t> & other = *it;

                    // std::lower_bound returns the first element where other >= data_range (lexicographically).
                    // This guarantees other.first >= data_range.first.
                    // Therefore, overlap occurs iff other.first < data_range.second
                    // (i.e., the other range starts before this range ends).
                    if (other.first < data_range.second) {
                        GGML_LOG_DEBUG("Writes overlap for %s", tensor->name);
                        writes_overlap = true;
                        break;
                    }
                }
            }

            //check if all srcs are either in branch or don't have a branch
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (!tensor->src[i]) {
                    continue;
                }

                auto it = stream_mapping.find(tensor->src[i]);

                if (it == stream_mapping.end()) {
                    continue;
                }

                if (it->second != stream) {
                    dependent_srcs = true;
                    break;
                }
            }

            if (dependent_srcs || writes_overlap) {
                break;
            }
        }

        return !writes_overlap && !dependent_srcs;
    }

    ~ggml_cuda_concurrent_event() {
        if (fork_event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(fork_event));
        }
        for (cudaEvent_t e : join_events) {
            if (e != nullptr) {
                CUDA_CHECK(cudaEventDestroy(e));
            }
        }
    }
};

struct ggml_cuda_stream_context {
    std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> concurrent_events;

    void reset() {
        concurrent_events.clear();
    }
};

struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaEvent_t copy_event = nullptr;

    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

    std::unique_ptr<ggml_cuda_graph> cuda_graph;

    int curr_stream_no = 0;

    explicit ggml_backend_cuda_context(int device) :
        device(device),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ggml_cuda_stream_context concurrent_stream_context;

    ~ggml_backend_cuda_context();

    cudaStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
        }
        return streams[device][stream];
    }

    cudaStream_t stream() { return stream(device, curr_stream_no); }

    ggml_cuda_stream_context & stream_context() { return concurrent_stream_context; }

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
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];

    static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device, int stream_no, bool alloc);

    ggml_cuda_pool & pool(int device) {
        if (pools[device][curr_stream_no] == nullptr) {
            bool alloc = true;
            if (pools[device][0] != nullptr) {
                alloc = pools[device][0]->alloc_memory();
            }
            pools[device][curr_stream_no] = new_pool_for_device(device, curr_stream_no, alloc);
        }
        return *pools[device][curr_stream_no];
    }

    ggml_cuda_pool & pool() {
        return pool(device);
    }

    void pool_set_alloc(bool alloc) {
        GGML_ASSERT(pools[device][curr_stream_no] == nullptr || pools[device][curr_stream_no]->alloc_memory() == alloc);

        if (pools[device][curr_stream_no] == nullptr) {
            pools[device][curr_stream_no] = new_pool_for_device(device, curr_stream_no, alloc);
        }
    }

    size_t pool_get_alloc_size(int stream_no) {
        if (pools[device][stream_no] == nullptr) {
            return 0;
        }

        return pools[device][stream_no]->alloc_size();
    }
};

struct ggml_cuda_mm_fusion_args_host {
    const ggml_tensor * x_bias = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * gate_bias = nullptr;
    ggml_glu_op glu_op;
};
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias = nullptr;
    const void * gate = nullptr;
    const void * gate_bias = nullptr;
    ggml_glu_op glu_op;
};
