#pragma once

#include "common.cuh"

#if defined(GGML_USE_MUSA)
#define GGML_USE_WMMA_FATTN
#endif // defined(GGML_USE_MUSA)

#if defined(GGML_HIP_ROCWMMA_FATTN)
#if defined(CDNA) && (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
#define GGML_USE_WMMA_FATTN
#elif defined(CDNA)
#warning "rocwmma fattn on CDNA is broken on rocwmma v2.0.0, expect degraded performance"
#endif // defined(CDNA) && (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
#if defined(RDNA3)
#define GGML_USE_WMMA_FATTN
#endif // defined(RDNA3)
#if defined(RDNA4) && ROCWMMA_VERSION_MAJOR > 1
#define GGML_USE_WMMA_FATTN
#elif defined(RDNA4)
#warning "rocwmma fattn is not suported on RDNA4 on rocwmma < v2.0.0, expect degraded performance"
#endif // defined(RDNA4) && ROCWMMA_VERSION_MAJOR > 1
#endif // defined(GGML_HIP_ROCWMMA_FATTN)

// WMMA flash attention requires FP16 matrix instructions to be available for ggml code.
static bool ggml_cuda_should_use_wmma_fattn(const int cc) {
#if defined(GGML_USE_HIP) && !defined(GGML_HIP_ROCWMMA_FATTN)
    return false;
#else
    if ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_MTHREADS(cc)) {
        return true;
    } else if (GGML_CUDA_CC_IS_CDNA(cc)){
#if defined(GGML_HIP_ROCWMMA_FATTN) && (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
        return true;
#else
        return false;
#endif // defined(GGML_HIP_ROCWMMA_FATTN) (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
    } else if (GGML_CUDA_CC_IS_RDNA4(cc)) {
#if defined(GGML_HIP_ROCWMMA_FATTN) && ROCWMMA_VERSION_MAJOR > 1
        return true;
#else
        return false;
#endif // defined(GGML_HIP_ROCWMMA_FATTN) && ROCWMMA_VERSION_MAJOR > 1
    } else {
        return false;
    }
#endif // defined(GGML_USE_HIP) && !defined(GGML_HIP_ROCWMMA_FATTN)
}

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
