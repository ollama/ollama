#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-f16.cuh"

#define FATTN_KQ_STRIDE_TILE_F16 64

template<int D, int ncols, int nwarps, int parallel_blocks, bool use_logit_softcap> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_f16(
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
#ifdef FP16_AVAILABLE
    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.y              + nb01*ic0);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V    + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask + ne11*ic0;

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slopef = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    __shared__ half KQ[ncols*FATTN_KQ_STRIDE_TILE_F16];
    half2 * KQ2 = (half2 *) KQ;

    __shared__ half2 KV_tmp[FATTN_KQ_STRIDE_TILE_F16][D/2 + 1]; // Pad D to avoid memory bank conflicts.

    half kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -HALF_MAX_HALF;
    }
    half2 kqsum[ncols/nwarps] = {{0.0f, 0.0f}};

    half2 VKQ[ncols/nwarps][(D/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Convert Q to half2 and store in registers:
    __shared__ half2 Q_h2[ncols][D/2];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i] : make_float2(0.0f, 0.0f);
            Q_h2[j][i] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
        }
    }

    __syncthreads();

    const int k_start = parallel_blocks == 1 ? 0 : ip*FATTN_KQ_STRIDE_TILE_F16;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE_TILE_F16) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        half kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;

                KV_tmp[i_KQ][k_KQ] = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
            }
        }

        __syncthreads();

        half2 sum2[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE][ncols/nwarps] = {{{0.0f, 0.0f}}};

#pragma unroll
        for (int k_KQ = 0; k_KQ < D/2; ++k_KQ) {
            half2 K_k[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE];
            half2 Q_k[ncols/nwarps];

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
                const int i_KQ = i_KQ_0 + threadIdx.x;

                K_k[i_KQ_0/WARP_SIZE] = KV_tmp[i_KQ][k_KQ];
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                Q_k[j_KQ_0/nwarps] = Q_h2[j_KQ][k_KQ];
            }

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += K_k[i_KQ_0/WARP_SIZE]*Q_k[j_KQ_0/nwarps];
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                half sum;
                if (use_logit_softcap) {
                    const float2 tmp = __half22float2(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                    sum = logit_softcap * tanhf(tmp.x + tmp.y);
                } else {
                    sum = __low2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]) + __high2half(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                }
                sum += mask ? slopeh*maskh[j_KQ*ne11 + k_VKQ_0 + i_KQ] : __float2half(0.0f);

                kqmax_new[j_KQ_0/nwarps] = ggml_cuda_hmax(kqmax_new[j_KQ_0/nwarps], sum);

                KQ[j_KQ*FATTN_KQ_STRIDE_TILE_F16 + i_KQ] = sum;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max(kqmax_new[j0/nwarps]);
            const half2 KQ_max_scale = __half2half2(hexp(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]));
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F16/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const half2 diff = KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + i] - __half2half2(kqmax[j0/nwarps]);
                const half2 val = h2exp(diff);
                kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale + val;
                KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + i] = val;
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                VKQ[j0/nwarps][i0/WARP_SIZE] *= KQ_max_scale;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                KV_tmp[k][i] = V_h2[(k_VKQ_0 + k)*stride_KV2 + i];
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += 2) {
            half2  V_k[(D/2)/WARP_SIZE][2];
            half2 KQ_k[ncols/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                V_k[i0/WARP_SIZE][0] = KV_tmp[k0 + 0][i];
                V_k[i0/WARP_SIZE][1] = KV_tmp[k0 + 1][i];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0/nwarps] = KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + k0/2];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    VKQ[j0/nwarps][i0/WARP_SIZE] += V_k[i0/WARP_SIZE][0]* __low2half2(KQ_k[j0/nwarps]);
                    VKQ[j0/nwarps][i0/WARP_SIZE] += V_k[i0/WARP_SIZE][1]*__high2half2(KQ_k[j0/nwarps]);
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

        half kqsum_j = __low2half(kqsum[j_VKQ_0/nwarps]) + __high2half(kqsum[j_VKQ_0/nwarps]);
        kqsum_j = warp_reduce_sum((float)kqsum_j);

#pragma unroll
        for (int i00 = 0; i00 < D; i00 += 2*WARP_SIZE) {
            const int i0 = i00 + 2*threadIdx.x;

            half2 dst_val = VKQ[j_VKQ_0/nwarps][i0/(2*WARP_SIZE)];
            if (parallel_blocks == 1) {
                dst_val /= __half2half2(kqsum_j);
            }
            const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 0] =  __low2float(dst_val);
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 1] = __high2float(dst_val);
        }

        if (parallel_blocks != 1 && threadIdx.x == 0) {
            dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
        }
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int cols_per_block, int parallel_blocks, bool use_logit_softcap>
void launch_fattn_tile_f16_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            constexpr int      D = 64;
            constexpr int nwarps = 8;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f16<D, cols_per_block, nwarps, parallel_blocks, use_logit_softcap>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        } break;
        case 128: {
            constexpr int      D = 128;
            constexpr int nwarps = 8;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f16<D, cols_per_block, nwarps, parallel_blocks, use_logit_softcap>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        } break;
        default: {
            GGML_ABORT("FlashAttention without tensor cores only supports head sizes 64 and 128.");
        } break;
    }
}

void ggml_cuda_flash_attn_ext_tile_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] <= 16) {
        constexpr int cols_per_block = 16;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 32;
        constexpr int parallel_blocks = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    constexpr int parallel_blocks = 1;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, use_logit_softcap>(ctx, dst);
    }
}
