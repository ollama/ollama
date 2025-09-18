#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec-f16.cuh"
#include "fattn-vec-f32.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
        return;
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const bool use_gqa_opt = mask && max_bias == 0.0f;

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            GGML_ASSERT(gqa_ratio % 16 == 0);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_F16_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f16_case<D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

static void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16 )

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)

    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

#define FATTN_VEC_F32_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f32_case<D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

static void ggml_cuda_flash_attn_ext_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC_F32  = 100,
    BEST_FATTN_KERNEL_VEC_F16  = 110,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    const int cc = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(KQV);

    switch (K->ne[0]) {
        case  64:
        case 128:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case  80:
        case  96:
        case 112:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!fp16_mma_available(cc) && !turing_mma_available(cc)) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!turing_mma_available(cc) || gqa_ratio % 16 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
#ifdef GGML_CUDA_FA_ALL_QUANTS
            if (K->ne[0] != 128 && K->ne[0] != 64) {
                return BEST_FATTN_KERNEL_NONE;
            }
#else
            if (K->ne[0] != 128) {
                return BEST_FATTN_KERNEL_NONE;
            }
#endif // GGML_CUDA_FA_ALL_QUANTS
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    switch (V->type) {
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            if (K->ne[0] != 128) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % (2*warp_size) == 0;

    // If Turing tensor cores available, use them except for some cases with batch size 1:
    if (turing_mma_available(cc)) {
        const bool gqa_opt_applies = gqa_ratio % 2 == 0 && mask; // The mma-based kernels have GQA-specific optimizations
        const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
        const bool mma_faster_for_rtx4000 = Q->ne[3] > 1 || (gqa_ratio > 4 && K->ne[1] >= 8192);
        const bool mma_faster_for_bs1 = gqa_opt_applies && !mma_needs_data_conversion &&
            (cc < GGML_CUDA_CC_ADA_LOVELACE || mma_faster_for_rtx4000);
        if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1) {
            if (prec == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
                return BEST_FATTN_KERNEL_VEC_F16;
            }
            return BEST_FATTN_KERNEL_VEC_F32;
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use kernels specializes for small batch sizes if possible:
    if (Q->ne[1] <= 8 && can_use_vector_kernel) {
        if (prec == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            return BEST_FATTN_KERNEL_VEC_F16;
        }
        return BEST_FATTN_KERNEL_VEC_F32;
    }

    // For large batch sizes, use the WMMA kernel if possible:
    if (fp16_mma_available(cc)) {
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    // If there is no suitable kernel for tensor cores or small batch sizes, use the generic kernel for large batch sizes:
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC_F32:
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC_F16:
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
