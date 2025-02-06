#include "common.cuh"
#include "fattn-common.cuh"

#ifdef FP16_MMA_AVAILABLE
#include <mma.h>
#endif // FP16_MMA_AVAILABLE

// D == head size, VKQ_stride == num VKQ rows calculated in parallel:
template<int D, int ncols, int nwarps, int VKQ_stride, int parallel_blocks, typename KQ_acc_t, bool use_logit_softcap>
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_ext_f16(
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
#ifdef FP16_MMA_AVAILABLE
    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = ncols*(blockIdx.x / parallel_blocks); // Index of the first Q/QKV column to work on.
    const int ip  =        blockIdx.x % parallel_blocks;  // Index in group of blocks running for the same column in parallel.

    static_assert(D <= FATTN_KQ_STRIDE, "D must be <= FATTN_KQ_STRIDE.");
    static_assert(ncols == 8 || ncols % 16 == 0, "ncols must be 8 or a multiple of 16.");
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0, "If ncols == 8 then D % frag_m must be 0.");
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::row_major> frag_a_K;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_a_V;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_b;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, KQ_acc_t>                      frag_c_KQ;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, half>                          frag_c_VKQ;

    constexpr int KQ_stride_tc  = nwarps*frag_m; // Number of KQ rows calculated in parallel.
    constexpr int VKQ_ratio = KQ_stride_tc/VKQ_stride; // Number of parallel VKQ accumulators needed to keep all warps busy.
    static_assert(VKQ_ratio <= nwarps, "VKQ_ratio must be <= nwarps.");

    // Pad internal representation of KQ, KQV to reduce shared memory bank conflicts:
    constexpr int D_padded = D + 8;
    constexpr int kqs_padded = FATTN_KQ_STRIDE + 8;
    constexpr int kqar = sizeof(KQ_acc_t)/sizeof(half);

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f   = (const float *) (Q + nb02* blockIdx.y              + nb01*ic0);
    const half  * K_h   = (const half  *) (K + nb12*(blockIdx.y / gqa_ratio));
    const half  * V_h   = (const half  *) (V + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half  * maskh = (const half  *)  mask + (nb31/sizeof(half))* ic0;
    const half2 * mask2 = (const half2 *)  mask + (nb31/sizeof(half))*(ic0/2);

    const int stride_Q  = nb01 / sizeof(float);
    const int stride_KV = nb11 / sizeof(half);

    const float slopef = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);
    const half2 slope2 = make_half2(slopef, slopef);

    const half2 logit_softcap_2 = make_half2(logit_softcap, logit_softcap);

    frag_b Q_b[D/16][ncols/frag_n];

    // A single buffer for temporarily holding tiles of KQ and VKQ parts:
    constexpr int mem_KQ = ncols*kqs_padded*kqar;
    constexpr int mem_VKQ_parts = VKQ_ratio*ncols*D_padded;
    __shared__ half KQ[mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts];
    float * KQ_f = (float *) KQ;
    half2 * KQ2 = (half2 *) KQ;

    float    KQ_rowsum_f[ncols/nwarps] = {0.0f};
    float       KQ_max_f[ncols/nwarps];
    float KQ_max_scale_f[ncols/nwarps] = {0.0f};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_f[j] = -FLT_MAX/2.0f;
    }

    half2    KQ_rowsum_h2[ncols/nwarps] = {{0.0f, 0.0f}};
    half2       KQ_max_h2[ncols/nwarps];
    half2 KQ_max_scale_h2[ncols/nwarps] = {{0.0f, 0.0f}};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_h2[j] = make_half2(-HALF_MAX_HALF, -HALF_MAX_HALF);
    }

    __shared__ half VKQ[ncols*D_padded]; // Accumulator for final VKQ slice.
    half2 * VKQ2 = (half2 *) VKQ;
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D/2 && i >= D/2) {
                break;
            }
            VKQ2[j*(D_padded/2) + i] = make_half2(0.0f, 0.0f);
        }
    }

    // Convert Q to half and apply scale, temporarily store in KQ:
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D && i >= D) {
                break;
            }
            KQ[j*D_padded + i] = ic0 + j < ne01 ? Q_f[j*stride_Q + i] * scale : 0.0f;
        }
    }

    __syncthreads();

    // Load Q into tensor core fragments/registers since it will be used frequently:
#pragma unroll
    for (int i0 = 0; i0 < D; i0 += 16) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
            nvcuda::wmma::load_matrix_sync(Q_b[i0/16][j0/frag_n], KQ + j0*D_padded + i0, D_padded);
        }
    }

    __syncthreads();

    // Iterate over ne11 == previous tokens:
    for (int k_VKQ_0 = ip*FATTN_KQ_STRIDE; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE) {
        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE; i_KQ_0 += KQ_stride_tc) {
            frag_c_KQ KQ_c[ncols/frag_n];
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(KQ_c[j], 0.0f);
            }
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 16) {
                frag_a_K K_a;
                nvcuda::wmma::load_matrix_sync(K_a, K_h + (k_VKQ_0 + i_KQ_0 + frag_m*threadIdx.y)*stride_KV + k_KQ_0, stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    nvcuda::wmma::mma_sync(KQ_c[j], K_a, Q_b[k_KQ_0/16][j], KQ_c[j]);
                }
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync((KQ_acc_t *) KQ + j0*kqs_padded + i_KQ_0 + frag_m*threadIdx.y, KQ_c[j0/frag_n], kqs_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (std::is_same<KQ_acc_t, float>::value) {
                float KQ_f_tmp[FATTN_KQ_STRIDE / WARP_SIZE];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/WARP_SIZE] = KQ_f[j*kqs_padded + k];

                    if (use_logit_softcap) {
                        KQ_f_tmp[k0/WARP_SIZE] = logit_softcap*tanhf(KQ_f_tmp[k0/WARP_SIZE]);
                    }
                }

                float KQ_max_new = KQ_max_f[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/WARP_SIZE] += mask ? __half2float(slopeh*maskh[j*(nb31/sizeof(half)) + k_VKQ_0 + k]) : 0.0f;
                    KQ_max_new = max(KQ_max_new, KQ_f_tmp[k0/WARP_SIZE]);
                }
                KQ_max_new = warp_reduce_max(KQ_max_new);

                const float diff = KQ_max_f[j0/nwarps] - KQ_max_new;
                KQ_max_scale_f[j0/nwarps] = expf(diff);
                if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_max_scale_f[j0/nwarps] = 0.0f;
                }
                KQ_max_f[j0/nwarps] = KQ_max_new;

                float KQ_rowsum_add = 0.0f;
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    const float diff = KQ_f_tmp[k0/WARP_SIZE] - KQ_max_f[j0/nwarps];
                    KQ_f_tmp[k0/WARP_SIZE] = expf(diff);
                    if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                        KQ_f_tmp[k0/WARP_SIZE] = 0.0f;
                    }
                    KQ_rowsum_add += KQ_f_tmp[k0/WARP_SIZE];
                    KQ[j*(kqar*kqs_padded) + k] = KQ_f_tmp[k0/WARP_SIZE];
                }
                KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_f[j0/nwarps] = KQ_max_scale_f[j0/nwarps]*KQ_rowsum_f[j0/nwarps] + KQ_rowsum_add;
            } else {
                half2 KQ2_tmp[FATTN_KQ_STRIDE/(2*WARP_SIZE)];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/WARP_SIZE] = KQ2[j*(kqs_padded/2) + k];

                    if (use_logit_softcap) {
                        // There is no dedicated tangens hyperbolicus function for half2.
                        KQ2_tmp[k0/WARP_SIZE] = h2exp(KQ2_tmp[k0/WARP_SIZE]*make_half2(2.0f, 2.0f));
                        KQ2_tmp[k0/WARP_SIZE] = (KQ2_tmp[k0/WARP_SIZE] - make_half2(1.0f, 1.0f))
                                               /(KQ2_tmp[k0/WARP_SIZE] + make_half2(1.0f, 1.0f));

                        KQ2_tmp[k0/WARP_SIZE] *= logit_softcap_2;
                    }
                }

                half2 KQ_max_new = KQ_max_h2[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/WARP_SIZE] += mask ? slope2*mask2[(j*ne11 + k_VKQ_0)/2 + k] : make_half2(0.0f, 0.0f);
                    KQ_max_new = ggml_cuda_hmax2(KQ_max_new, KQ2_tmp[k0/WARP_SIZE]);
                }
                KQ_max_new = __half2half2(warp_reduce_max(ggml_cuda_hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))));
                const half2 diff = KQ_max_h2[j0/nwarps] - KQ_max_new;
                KQ_max_scale_h2[j0/nwarps] = h2exp(diff);
                const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                *((uint32_t *) &KQ_max_scale_h2[j0/nwarps]) &= ftz_mask;
                KQ_max_h2[j0/nwarps] = KQ_max_new;

                half2 KQ_rowsum_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    const half2 diff = KQ2_tmp[k0/WARP_SIZE] - KQ_max_h2[j0/nwarps];
                    KQ2_tmp[k0/WARP_SIZE] = h2exp(diff);
                    const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                    *((uint32_t *) &KQ2_tmp[k0/WARP_SIZE]) &= ftz_mask;
                    KQ_rowsum_add += KQ2_tmp[k0/WARP_SIZE];
                    KQ2[j*(kqs_padded/2) + k] = KQ2_tmp[k0/WARP_SIZE];
                }
                KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_h2[j0/nwarps] = KQ_max_scale_h2[j0/nwarps]*KQ_rowsum_h2[j0/nwarps] + KQ_rowsum_add;
            }
        }

        __syncthreads();

        frag_b KQ_b[FATTN_KQ_STRIDE/(VKQ_ratio*16)][ncols/frag_n];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio)*16;
                nvcuda::wmma::load_matrix_sync(
                    KQ_b[k0/(VKQ_ratio*16)][j0/frag_n],
                    KQ + j0*(kqar*kqs_padded) + k,
                    kqar*kqs_padded);
            }
        }

        frag_c_VKQ VKQ_c[D/VKQ_stride][ncols/frag_n];
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(VKQ_c[i_VKQ_0/VKQ_stride][j], 0.0f);
            }

#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio)*16;

                frag_a_V v_a;
                nvcuda::wmma::load_matrix_sync(v_a, V_h + (k_VKQ_0 + k)*stride_KV + i_VKQ_0 + frag_m*(threadIdx.y/VKQ_ratio), stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    nvcuda::wmma::mma_sync(VKQ_c[i_VKQ_0/VKQ_stride][j], v_a, KQ_b[k0/(VKQ_ratio*16)][j], VKQ_c[i_VKQ_0/VKQ_stride][j]);
                }
            }
        }

        __syncthreads();

        const int offset_k = (threadIdx.y % VKQ_ratio) * (ncols*D_padded);
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += VKQ_stride) {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync(
                    KQ + offset_k + j0*D_padded + i_KQ_0 + frag_m*(threadIdx.y/VKQ_ratio),
                    VKQ_c[i_KQ_0/VKQ_stride][j0/frag_n],
                    D_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            half2 VKQ_scale;
            if (std::is_same<KQ_acc_t, float>::value) {
                VKQ_scale = make_half2(KQ_max_scale_f[j0/nwarps], KQ_max_scale_f[j0/nwarps]);
            } else {
                VKQ_scale = KQ_max_scale_h2[j0/nwarps];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                if (i0 + WARP_SIZE > D/2 && i >= D/2) {
                    break;
                }

                half2 VKQ_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int l = 0; l < VKQ_ratio; ++l) {
                    VKQ_add += KQ2[l*(ncols*D_padded/2) + j*(D_padded/2) + i];
                }
                VKQ2[j*(D_padded/2) + i] = VKQ_scale*VKQ2[j*(D_padded/2) + i] + VKQ_add;
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j_VKQ = j0 + threadIdx.y;
        if (ic0 + j_VKQ >= ne01) {
            return;
        }
        const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;

        float KQ_rowsum_j;
        if (std::is_same<KQ_acc_t, float>::value) {
            KQ_rowsum_j = KQ_rowsum_f[j0/nwarps];
        } else {
            KQ_rowsum_j = __low2float(KQ_rowsum_h2[j0/nwarps]) + __high2float(KQ_rowsum_h2[j0/nwarps]);
        }

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D && i >= D) {
                break;
            }
            float dst_val = VKQ[j_VKQ*D_padded + i];
            if (parallel_blocks == 1) {
                dst_val /= KQ_rowsum_j;
            }
            dst[j_dst*gridDim.y*D + blockIdx.y*D + i] = dst_val;
        }

        if (parallel_blocks == 1 || threadIdx.x != 0) {
            continue;
        }

        float2 dst_meta_val;
        if (std::is_same<KQ_acc_t, float>::value) {
            dst_meta_val.x = KQ_max_f[j0/nwarps];
        } else {
            dst_meta_val.x = __low2float(KQ_max_h2[j0/nwarps]);
        }
        dst_meta_val.y = KQ_rowsum_j;
        dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = dst_meta_val;
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_MMA_AVAILABLE
}

constexpr int get_max_power_of_2(int x) {
    return x % 2 == 0 ? 2*get_max_power_of_2(x/2) : 1;
}

static_assert(get_max_power_of_2(1) == 1, "Test failed.");
static_assert(get_max_power_of_2(2) == 2, "Test failed.");
static_assert(get_max_power_of_2(4) == 4, "Test failed.");
static_assert(get_max_power_of_2(6) == 2, "Test failed.");

// Number of VKQ rows calculated in parallel:
constexpr int get_VKQ_stride(int D, int nwarps, int frag_m) {
    return (get_max_power_of_2(D/frag_m) < nwarps ? get_max_power_of_2(D/frag_m) : nwarps)*frag_m;
}

static_assert(get_VKQ_stride(128, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride(128, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride(128, 4, 32) == 128, "Test failed.");
static_assert(get_VKQ_stride( 64, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride( 64, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 64, 4, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 80, 1, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 2, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 4, 16) ==  16, "Test failed.");

template <int D, int cols_per_block, typename KQ_acc_t>
void ggml_cuda_flash_attn_ext_wmma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    constexpr int nwarps = 4;

    constexpr int frag_m = cols_per_block == 8 && D % 32 == 0 ? 32 : 16;
    const int blocks_num_pb1 = ((Q->ne[1] + cols_per_block - 1) / cols_per_block)*Q->ne[2]*Q->ne[3];
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (4*blocks_num_pb1 < 2*nsm) {
        constexpr int parallel_blocks = 4;
        fattn_kernel_t fattn_kernel;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
        } else {
            constexpr bool use_logit_softcap = true;
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
        }
        launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        return;
    }
    if (2*blocks_num_pb1 < 2*nsm) {
        constexpr int parallel_blocks = 2;
        fattn_kernel_t fattn_kernel;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
        } else {
            constexpr bool use_logit_softcap = true;
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
        }
        launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        return;
    }
    constexpr int parallel_blocks = 1;
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<
            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<
            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t, use_logit_softcap>;
    }
    launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
}

#define DECL_FATTN_WMMA_F16_CASE(D, cols_per_block, KQ_acc_t)                         \
    template void ggml_cuda_flash_attn_ext_wmma_f16_case                              \
    <D, cols_per_block, KQ_acc_t>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_WMMA_F16_CASE( 64, 16, float);
extern DECL_FATTN_WMMA_F16_CASE( 80, 16, float);
extern DECL_FATTN_WMMA_F16_CASE( 96, 16, float);
extern DECL_FATTN_WMMA_F16_CASE(112, 16, float);
extern DECL_FATTN_WMMA_F16_CASE(128, 16, float);
extern DECL_FATTN_WMMA_F16_CASE(256, 16, float);

extern DECL_FATTN_WMMA_F16_CASE( 64, 32, float);
extern DECL_FATTN_WMMA_F16_CASE( 80, 32, float);
extern DECL_FATTN_WMMA_F16_CASE( 96, 32, float);
extern DECL_FATTN_WMMA_F16_CASE(112, 32, float);
extern DECL_FATTN_WMMA_F16_CASE(128, 32, float);
// extern DECL_FATTN_WMMA_F16_CASE(256, 16, float);

extern DECL_FATTN_WMMA_F16_CASE( 64,  8, half);
extern DECL_FATTN_WMMA_F16_CASE( 96,  8, half);
extern DECL_FATTN_WMMA_F16_CASE(128,  8, half);
extern DECL_FATTN_WMMA_F16_CASE(256,  8, half);

extern DECL_FATTN_WMMA_F16_CASE( 64, 16, half);
extern DECL_FATTN_WMMA_F16_CASE( 80, 16, half);
extern DECL_FATTN_WMMA_F16_CASE( 96, 16, half);
extern DECL_FATTN_WMMA_F16_CASE(112, 16, half);
extern DECL_FATTN_WMMA_F16_CASE(128, 16, half);
extern DECL_FATTN_WMMA_F16_CASE(256, 16, half);

extern DECL_FATTN_WMMA_F16_CASE( 64, 32, half);
extern DECL_FATTN_WMMA_F16_CASE( 80, 32, half);
extern DECL_FATTN_WMMA_F16_CASE( 96, 32, half);
extern DECL_FATTN_WMMA_F16_CASE(112, 32, half);
extern DECL_FATTN_WMMA_F16_CASE(128, 32, half);
extern DECL_FATTN_WMMA_F16_CASE(256, 16, half);
