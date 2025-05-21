#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-f32.cuh"

#define FATTN_KQ_STRIDE_TILE_F32 32

template<int D, int ncols, int nwarps, bool use_logit_softcap> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_f32(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
#ifdef FP16_MMA_AVAILABLE
    NO_DEVICE_CODE;
    return;
#endif // FP16_MMA_AVAILABLE
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
        GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
        GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
        GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
        GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02);
        GGML_UNUSED(ne03); GGML_UNUSED(ne10); GGML_UNUSED(ne11);
        GGML_UNUSED(ne12); GGML_UNUSED(ne13); GGML_UNUSED(ne31);
        GGML_UNUSED(nb31); GGML_UNUSED(nb01); GGML_UNUSED(nb02);
        GGML_UNUSED(nb03); GGML_UNUSED(nb11); GGML_UNUSED(nb12);
        GGML_UNUSED(nb13); GGML_UNUSED(nb21); GGML_UNUSED(nb22);
        GGML_UNUSED(nb23); GGML_UNUSED(ne0); GGML_UNUSED(ne1);
        GGML_UNUSED(ne2); GGML_UNUSED(ne3);
        NO_DEVICE_CODE;
        return;
    }

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.z              + nb01*ic0);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.z / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V    + nb12*(blockIdx.z / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask + ne11*ic0;

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slope = get_alibi_slope(max_bias, blockIdx.z, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    __shared__ float KQ[ncols*FATTN_KQ_STRIDE_TILE_F32];

    __shared__ float KV_tmp[FATTN_KQ_STRIDE_TILE_F32][D + 1]; // Pad D to avoid memory bank conflicts.
    float2 * KV_tmp2 = (float2 *) KV_tmp;

    float kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols/nwarps] = {0.0f};

    float2 VKQ[ncols/nwarps][(D/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Convert Q to half2 and store in registers:
    __shared__ float Q_f[ncols][D];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += 2*WARP_SIZE) {
            float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i0/2 + threadIdx.x] : make_float2(0.0f, 0.0f);
            Q_f[j][i0 + 0*WARP_SIZE + threadIdx.x] = tmp.x * scale;
            Q_f[j][i0 + 1*WARP_SIZE + threadIdx.x] = tmp.y * scale;
        }
    }

    __syncthreads();

    for (int k_VKQ_0 = blockIdx.y*FATTN_KQ_STRIDE_TILE_F32; k_VKQ_0 < ne11; k_VKQ_0 += gridDim.y*FATTN_KQ_STRIDE_TILE_F32) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 2*WARP_SIZE) {
                const half2 tmp = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ_0/2 + threadIdx.x];
                KV_tmp[i_KQ][k_KQ_0 + 0*WARP_SIZE + threadIdx.x] =  __low2float(tmp);
                KV_tmp[i_KQ][k_KQ_0 + 1*WARP_SIZE + threadIdx.x] = __high2float(tmp);
            }
        }

        __syncthreads();

        float sum[FATTN_KQ_STRIDE_TILE_F32/WARP_SIZE][ncols/nwarps] = {{0.0f}};

#pragma unroll
        for (int k_KQ = 0; k_KQ < D; ++k_KQ) {
            float K_k[FATTN_KQ_STRIDE_TILE_F32/WARP_SIZE];
            float Q_k[ncols/nwarps];

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
                const int i_KQ = i_KQ_0 + threadIdx.x;

                K_k[i_KQ_0/WARP_SIZE] = KV_tmp[i_KQ][k_KQ];
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                Q_k[j_KQ_0/nwarps] = Q_f[j_KQ][k_KQ];
            }

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += K_k[i_KQ_0/WARP_SIZE] * Q_k[j_KQ_0/nwarps];
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                if (use_logit_softcap) {
                    sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] = logit_softcap * tanhf(sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                }

                sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += mask ? slope*__half2float(maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new[j_KQ_0/nwarps] = fmaxf(kqmax_new[j_KQ_0/nwarps], sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);

                KQ[j_KQ*FATTN_KQ_STRIDE_TILE_F32 + i_KQ] = sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max(kqmax_new[j0/nwarps]);
            const float KQ_max_scale = expf(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]);
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

            float kqsum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F32; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float diff = KQ[j*FATTN_KQ_STRIDE_TILE_F32 + i] - kqmax[j0/nwarps];
                const float val = expf(diff);
                kqsum_add += val;
                KQ[j*FATTN_KQ_STRIDE_TILE_F32 + i] = val;
            }
            kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale + kqsum_add;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                VKQ[j0/nwarps][i0/WARP_SIZE].x *= KQ_max_scale;
                VKQ[j0/nwarps][i0/WARP_SIZE].y *= KQ_max_scale;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F32; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                KV_tmp2[k*(D/2) + i].x =  __low2float(V_h2[(k_VKQ_0 + k)*stride_KV2 + i]);
                KV_tmp2[k*(D/2) + i].y = __high2float(V_h2[(k_VKQ_0 + k)*stride_KV2 + i]);
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < FATTN_KQ_STRIDE_TILE_F32; ++k) {
            float2 V_k[(D/2)/WARP_SIZE];
            float  KQ_k[ncols/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                V_k[i0/WARP_SIZE] = KV_tmp2[k*(D/2) + i];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0/nwarps] = KQ[j*FATTN_KQ_STRIDE_TILE_F32 + k];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    VKQ[j0/nwarps][i0/WARP_SIZE].x += V_k[i0/WARP_SIZE].x*KQ_k[j0/nwarps];
                    VKQ[j0/nwarps][i0/WARP_SIZE].y += V_k[i0/WARP_SIZE].y*KQ_k[j0/nwarps];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        float kqsum_j = kqsum[j_VKQ_0/nwarps];
        kqsum_j = warp_reduce_sum(kqsum_j);

#pragma unroll
        for (int i00 = 0; i00 < D; i00 += 2*WARP_SIZE) {
            const int i0 = i00 + 2*threadIdx.x;

            float2 dst_val = VKQ[j_VKQ_0/nwarps][i0/(2*WARP_SIZE)];
            if (gridDim.y == 1) {
                dst_val.x /= kqsum_j;
                dst_val.y /= kqsum_j;
            }
            const int j_dst = (ic0 + j_VKQ)*gridDim.y + blockIdx.y;
            dst[j_dst*D*gridDim.z + D*blockIdx.z + i0 + 0] = dst_val.x;
            dst[j_dst*D*gridDim.z + D*blockIdx.z + i0 + 1] = dst_val.y;
        }

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[((ic0 + j_VKQ)*gridDim.z + blockIdx.z) * gridDim.y + blockIdx.y] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
        }
    }
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
    GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02);
    GGML_UNUSED(ne03); GGML_UNUSED(ne10); GGML_UNUSED(ne11);
    GGML_UNUSED(ne12); GGML_UNUSED(ne13); GGML_UNUSED(ne31);
    GGML_UNUSED(nb31); GGML_UNUSED(nb01); GGML_UNUSED(nb02);
    GGML_UNUSED(nb03); GGML_UNUSED(nb11); GGML_UNUSED(nb12);
    GGML_UNUSED(nb13); GGML_UNUSED(nb21); GGML_UNUSED(nb22);
    GGML_UNUSED(nb23); GGML_UNUSED(ne0); GGML_UNUSED(ne1);
    GGML_UNUSED(ne2); GGML_UNUSED(ne3);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int cols_per_block, bool use_logit_softcap>
void launch_fattn_tile_f32_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            constexpr int    D             = 64;
            constexpr int    nwarps        = 8;
            constexpr size_t nbytes_shared = 0;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, use_logit_softcap>;
            launch_fattn<D, cols_per_block, 1>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, FATTN_KQ_STRIDE_TILE_F32, true, true, false);
        } break;
        case 128: {
            constexpr int    D             = 128;
            constexpr int    nwarps        = 8;
            constexpr size_t nbytes_shared = 0;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, use_logit_softcap>;
            launch_fattn<D, cols_per_block, 1>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, FATTN_KQ_STRIDE_TILE_F32, true, true, false);
        } break;
        default: {
            GGML_ABORT("FlashAttention without tensor cores only supports head sizes 64 and 128.");
        } break;
    }
}

void ggml_cuda_flash_attn_ext_tile_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] <= 16) {
        constexpr int cols_per_block = 16;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f32_64_128<cols_per_block, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f32_64_128<cols_per_block, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_f32_64_128<cols_per_block, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_f32_64_128<cols_per_block, use_logit_softcap>(ctx, dst);
    }
}
