#pragma once

#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
// for rocblas_initialize()
#include "rocblas/rocblas.h"

#if defined(GGML_HIP_ROCWMMA_FATTN)
#include <rocwmma/rocwmma-version.hpp>
#endif // defined(GGML_HIP_ROCWMMA_FATTN)

#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUDA_R_16F  HIPBLAS_R_16F
#define CUDA_R_16BF HIPBLAS_R_16B
#define CUDA_R_32F  HIPBLAS_R_32F
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hipMemAllocationGranularityRecommended
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_CHECK(fn) {hipError_t err = fn; if(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#define __shfl_up_sync(mask, var, laneMask, width) __shfl_up(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define __all_sync(mask, var) __all(var)
#define __any_sync(mask, var) __any(var)
#define cublasStrsmBatched hipblasStrsmBatched
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
#define cublasOperation_t hipblasOperation_t
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
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
#define cudaMalloc hipMalloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaMallocManaged hipMallocManaged
#define cudaMemAdvise hipMemAdvise
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
#define cuDeviceGet hipDeviceGet
#define CUdevice hipDevice_t
#define CUdeviceptr hipDeviceptr_t
#define cuMemUnmap hipMemUnmap
#define CUmemAccessDesc hipMemAccessDesc
#define cuMemAddressFree hipMemAddressFree
#define cuMemRelease hipMemRelease
#define CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t
#define cuMemCreate hipMemCreate
#define cuMemAddressReserve hipMemAddressReserve
#define cuMemMap hipMemMap
#define cuMemSetAccess hipMemSetAccess
#define cuMemGetAllocationGranularity hipMemGetAllocationGranularity
#define CUmemAllocationProp hipMemAllocationProp
#define cuDeviceGetAttribute hipDeviceGetAttribute
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamFireAndForget hipStreamFireAndForget
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamPerThread hipStreamPerThread
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphNode_t hipGraphNode_t
#define cudaKernelNodeParams hipKernelNodeParams
#define cudaKernelNodeParams hipKernelNodeParams
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaErrorGraphExecUpdateFailure hipErrorGraphExecUpdateFailure
#define cudaGraphExecUpdateResult hipGraphExecUpdateResult
#define cudaGraphNodeType hipGraphNodeType
#define cudaGraphNodeTypeKernel hipGraphNodeTypeKernel
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphKernelNodeSetParams hipGraphKernelNodeSetParams
#define cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#define cudaGraphKernelNodeGetParams hipGraphKernelNodeGetParams
#define cudaGraphNodeGetType hipGraphNodeGetType
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphExecUpdate hipGraphExecUpdate
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaGraph_t hipGraph_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define __trap() do { abort(); __builtin_unreachable(); } while(0)
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

#if HIP_VERSION >= 60500000
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#define cublasComputeType_t hipblasComputeType_t
#define cudaDataType_t hipDataType
#else
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define cublasComputeType_t hipblasDatatype_t
#define cudaDataType_t hipblasDatatype_t
#endif // HIP_VERSION >= 6050000

#if !defined(__HIP_PLATFORM_AMD__)
#error "The HIP backend supports only AMD targets"
#endif // !defined(__HIP_PLATFORM_AMD__)

#define __CUDA_ARCH__ 1300

#if defined(__gfx900__) || defined(__gfx906__)
#define GCN5
#endif // defined(__gfx900__) || defined(__gfx906__)

#if defined(__gfx803__)
#define GCN4
#endif // defined(__gfx803__)

#if defined(GCN5) || defined(GCN4)
#define GCN
#endif // defined(GCN5) || defined(GCN4)

#if defined(__gfx942__)
#define CDNA3
#endif // defined(__gfx942__)

#if defined(__gfx90a__)
#define CDNA2
#endif // defined(__gfx90a__)

#if defined(__gfx908__)
#define CDNA1
#endif // defined(__gfx908__)

#if defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#define CDNA // For the entire family
#endif // defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#if defined(__GFX12__)
#define RDNA4
#endif // defined(__GFX12__)

#if defined(__GFX11__)
#define RDNA3
#endif // defined(__GFX11__)

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2
#endif

#if defined(__gfx1010__) || defined(__gfx1012__)
#define RDNA1
#endif // defined(__gfx1010__) || defined(__gfx1012__)

#if defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#define RDNA // For the entire family
#endif // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

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

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}
