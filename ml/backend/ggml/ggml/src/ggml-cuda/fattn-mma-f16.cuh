#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

using namespace ggml_cuda_mma;

typedef tile<16,  8, half2> tile_A;
typedef tile< 8,  8, half2> tile_B;
typedef tile<16,  8, half2> tile_B_16;
typedef tile<16,  8, float> tile_C_KQ;
typedef tile<16, 16, float> tile_C_KQ_16;
typedef tile<16,  4, half2> tile_C_VKQ;
typedef tile<16,  8, half2> tile_C_VKQ_16;

template<int D, int nwarps, int KQ_per_iter>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV) {
    constexpr int D2_padded = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.

    // If cp.async is available, load up to the highest power of 2 in D asynchronously:
#ifdef CP_ASYNC_AVAILABLE
    static_assert(D >= 64 && D < 512, "bad D");
    constexpr int k0_sync_start = D/2 < 64 ? 32 : (D/2 < 128 ? 64 : 128);

    const unsigned int tile_KV_32 = __cvta_generic_to_shared(tile_KV);

    constexpr int preload = 64;
    constexpr int h2_per_chunk = 16/sizeof(half2);
    constexpr int chunks_per_row = k0_sync_start / h2_per_chunk;
    constexpr int stride_i = WARP_SIZE / chunks_per_row;
#pragma unroll
    for (int i0 = 0; i0 < KQ_per_iter; i0 += nwarps*stride_i) {
        const int i = i0 + threadIdx.y*stride_i + (chunks_per_row == WARP_SIZE ? 0 : threadIdx.x / chunks_per_row);
        const int k = (chunks_per_row == WARP_SIZE ? threadIdx.x : threadIdx.x % chunks_per_row)*h2_per_chunk;

        cp_async_cg_16<preload>(tile_KV_32 + (i*D2_padded + k)*sizeof(half2), KV + i*stride_KV + k);
    }
#else
    constexpr int k0_sync_start = 0;
#endif // CP_ASYNC_AVAILABLE
    static_assert(k0_sync_start % WARP_SIZE == 0, "bad k0_sync_start");

    // If D is not a power of 2, the rest is loaded synchronously.
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    static_assert(KQ_per_iter % (4*nwarps) == 0, "out of bounds");
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start = stride_k == WARP_SIZE ? k0_sync_start : D/2 - (D/2) % (2*stride_k);
        const int k0_stop  =                                         D/2 - (D/2) % (1*stride_k);
        const int stride_i = WARP_SIZE / stride_k;

        if (k0_start == k0_stop || k0_stop <= k0_sync_start) {
            continue;
        }

#pragma unroll
        for (int i0 = 0; i0 < KQ_per_iter; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                tile_KV[i*D2_padded + k] = KV[i*stride_KV + k];
            }
        }
    }
}

template<int ncols1, int nwarps, int KQ_per_iter>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half2 * const __restrict__ mask_h2, half2 * const __restrict__ tile_mask, const int stride_mask) {
    static_assert(KQ_per_iter == 2*WARP_SIZE || KQ_per_iter == WARP_SIZE, "bad KQ_per_iter");
#ifdef CP_ASYNC_AVAILABLE
    constexpr int preload = KQ_per_iter * sizeof(half);
    constexpr int cols_per_warp = 8*WARP_SIZE/KQ_per_iter;
    constexpr int stride_j = nwarps * cols_per_warp;

    const unsigned int tile_mask_32 = __cvta_generic_to_shared(tile_mask);

#pragma unroll
    for (int j0 = 0; j0 < ncols1; j0 += stride_j) {
        const int j = j0 + threadIdx.y*cols_per_warp +
            (KQ_per_iter == 2*WARP_SIZE ? threadIdx.x / (WARP_SIZE/4) : threadIdx.x / (WARP_SIZE/8));

        if (j0 + stride_j > ncols1 && j >= ncols1) {
            break;
        }

        const int i = 4 * (KQ_per_iter == 2*WARP_SIZE ? threadIdx.x % (WARP_SIZE/4) : threadIdx.x % (WARP_SIZE/8));

        cp_async_cg_16<preload>(tile_mask_32 + j*(KQ_per_iter*sizeof(half) + 16) + i*sizeof(half2), mask_h2 + j*stride_mask + i);
    }
#else
    constexpr int cols_per_warp = 2*WARP_SIZE/KQ_per_iter;
    constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
    for (int j0 = 0; j0 < ncols1; j0 += stride_j) {
        const int j = j0 + threadIdx.y*cols_per_warp + (KQ_per_iter == 2*WARP_SIZE ? 0 : threadIdx.x / (WARP_SIZE/2));

        if (j0 + stride_j > ncols1 && j >= ncols1) {
            break;
        }

        const int i = KQ_per_iter == 2*WARP_SIZE ? threadIdx.x : threadIdx.x % (WARP_SIZE/2);

        tile_mask[j*(KQ_per_iter/2 + 4) + i] = mask_h2[j*stride_mask + i];
    }
#endif // CP_ASYNC_AVAILABLE
}

template<int D, int ncols1, int ncols2, int nwarps, int KQ_per_iter, int ntiles, bool use_logit_softcap, bool needs_fixup, bool is_fixup, bool last_iter>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half2  * const __restrict__ mask_h2,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const int ne01,
        const int ne02,
        const int stride_KV,
        const int stride_mask,
        const int jt,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half2        * const __restrict__ tile_mask,
        const tile_B * const __restrict__ Q_B,
        tile_C_VKQ   * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int kb0) {
#ifdef NEW_MMA_AVAILABLE
    constexpr int cols_per_warp   = ntiles * tile_B::I;
    constexpr int cols_per_thread = ntiles == 1 ? 2 : ntiles;
    constexpr int np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
    constexpr int D2_padded       = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.

    const int k_VKQ_0 = kb0 * KQ_per_iter;
    tile_C_KQ KQ_C[KQ_per_iter/(np*tile_C_KQ::I) * ntiles];

    // Use wide variants of tiles if ntiles >= 2.
    tile_B_16     * Q_B_16   = (tile_B_16     *) Q_B;
    tile_C_VKQ_16 * VKQ_C_16 = (tile_C_VKQ_16 *) VKQ_C;
    tile_C_KQ_16  * KQ_C_16  = (tile_C_KQ_16  *) KQ_C;

#ifdef CP_ASYNC_AVAILABLE
    cp_async_wait_all();
    __syncthreads();
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(V_h2 + k_VKQ_0*stride_KV, tile_V, stride_KV);
#else
    if (ncols2 > 1 || mask_h2) {
        flash_attn_ext_f16_load_mask<ncols1, nwarps, KQ_per_iter>(mask_h2 + k_VKQ_0/2, tile_mask, stride_mask);
    }
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(K_h2 + k_VKQ_0*stride_KV, tile_K, stride_KV);
    __syncthreads();
#endif // CP_ASYNC_AVAILABLE

    // Calculate tile of KQ:
#pragma unroll
    for (int i_KQ_00 = 0; i_KQ_00 < KQ_per_iter; i_KQ_00 += np*tile_A::I) {
        const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*tile_A::I;
#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += tile_A::J) {
            tile_A K_A;
            load_ldmatrix(K_A, tile_K + i_KQ_0*D2_padded + k_KQ_0, D2_padded);
            if (ntiles == 1) {
                mma(KQ_C[i_KQ_00/(np*tile_A::I)], K_A, Q_B[k_KQ_0/tile_A::J]);
            } else {
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
                    // Wide version of KQ_C is column-major => swap A and B.
                    mma(KQ_C_16[i_KQ_00/(np*tile_A::I) * ntiles/2 + t], Q_B_16[k_KQ_0/tile_A::J * ntiles/2 + t], K_A);
                }
            }
        }
    }

#ifndef CP_ASYNC_AVAILABLE
    __syncthreads(); // Only needed if tile_K == tile_V.
#endif // CP_ASYNC_AVAILABLE

    if (use_logit_softcap) {
        static_assert(KQ_per_iter % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < KQ_per_iter/(np*tile_C_KQ::I) * ntiles; ++i) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if (ntiles == 1) {
        if (ncols2 > 1 || mask_h2) {
#pragma unroll
            for (int i00 = 0; i00 < KQ_per_iter; i00 += np*tile_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*tile_C_KQ::I;
#pragma unroll
                for (int l = 0; l < tile_C_KQ::ne; ++l) {
                    const int i = i0 + tile_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*tile_C_KQ::J + tile_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*tile_C_KQ::I)].x[l] += slope *
                        __half2float(((const half *) tile_mask)[j*(KQ_per_iter + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(KQ_per_iter % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*tile_C_KQ::I); ++k) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_max_new[l % 2] = fmaxf(KQ_max_new[l % 2], KQ_C[k].x[l]);
            }
        }

        // Values per KQ column are spread across 8 threads, does not need full warp reduce:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(KQ_per_iter % (np*tile_C_KQ::I) == 0, "bad loop size");

#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*tile_C_KQ::I); ++k) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_C[k].x[l] = expf(KQ_C[k].x[l] - KQ_max_new[l % 2]);

                KQ_rowsum_add[l % 2] += KQ_C[k].x[l];
            }
        }
    } else { // ntiles > 1
        if (ncols2 > 1 || mask_h2) {
#pragma unroll
            for (int i00 = 0; i00 < KQ_per_iter; i00 += np*tile_C_KQ_16::J) {
                const int i0 = i00 + (threadIdx.y % np)*tile_C_KQ_16::J;
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_KQ_16::ne; l0 += 2) {
                        const int i = (i0 + tile_C_KQ_16::get_j(l0)) / 2;
                        const int j = ((threadIdx.y / np)*cols_per_warp + t*tile_C_KQ_16::I + tile_C_KQ_16::get_i(l0)) / ncols2;

                        const float2 tmp = __half22float2(tile_mask[j*(KQ_per_iter/2 + 4) + i]);
                        const int KQ_index = i00/(np*tile_C_KQ_16::J) * ntiles/2 + t;
                        KQ_C_16[KQ_index].x[l0 + 0] += slope*tmp.x;
                        KQ_C_16[KQ_index].x[l0 + 1] += slope*tmp.y;
                    }
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(KQ_per_iter % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*tile_C_KQ_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                for (int l = 0; l < tile_C_KQ_16::ne; ++l) {
                    const int KQ_index = 2*t + (l/2) % 2;
                    KQ_max_new[KQ_index] = fmaxf(KQ_max_new[KQ_index], KQ_C_16[k*ntiles/2 + t].x[l]);
                }
            }
        }

        // Values per KQ column are spread across 4 threads, does not need full warp reduce:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 2; offset >= 1; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(KQ_per_iter % (np*tile_C_KQ_16::J) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*tile_C_KQ_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                for (int l = 0; l < tile_C_KQ_16::ne; ++l) {
                    const int KQ_index = 2*t + (l/2) % 2;

                    KQ_C_16[k*ntiles/2 + t].x[l] = expf(KQ_C_16[k*ntiles/2 + t].x[l] - KQ_max_new[KQ_index]);

                    KQ_rowsum_add[KQ_index] += KQ_C_16[k*ntiles/2 + t].x[l];
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            KQ_max_scale[col] = expf(KQ_max[col] - KQ_max_new[col]);
            KQ_max[col] = KQ_max_new[col];

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

        if (ntiles == 1) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < D/tile_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < tile_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < D/tile_C_VKQ_16::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_VKQ_16::ne; l0 += 2) {
                        VKQ_C_16[i*ntiles/2 + col/2].x[l0 + col % 2] *= KQ_max_scale_h2;
                    }
                }
            }
        }
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    tile_B B[KQ_per_iter/(np*2*tile_B::J) * ntiles];
    tile_B_16 * B_16 = (tile_B_16 *) B;
    static_assert(KQ_per_iter % (np*2*tile_B::J) == 0, "bad loop size");
    if (ntiles == 1) {
#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*2*tile_B::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < KQ_per_iter/(np*2*tile_B_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
                B_16[k*ntiles/2 + t] = get_half2(KQ_C_16[k*ntiles/2 + t]);
            }
        }
    }

#ifdef CP_ASYNC_AVAILABLE
    // Preload K tile for next iteration:
    cp_async_wait_all();
    __syncthreads();
    if (!last_iter) {
        if (ncols2 > 1 || mask_h2) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, KQ_per_iter>(mask_h2 + (k_VKQ_0 + KQ_per_iter)/2, tile_mask, stride_mask);
        }
        flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(K_h2 + (k_VKQ_0 + KQ_per_iter)*stride_KV, tile_K, stride_KV);
    }
#else
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(V_h2 + k_VKQ_0*stride_KV, tile_V, stride_KV);
    __syncthreads();
#endif // CP_ASYNC_AVAILABLE

    // Calculate VKQ tile:
#pragma unroll
    for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += tile_C_VKQ::I) {
        static_assert((KQ_per_iter/2) % (np*tile_A::J) == 0, "bad loop size");
#pragma unroll
        for (int k00 = 0; k00 < KQ_per_iter/2; k00 += np*tile_A::J) {
            const int k0 = k00 + (threadIdx.y % np)*tile_A::J;

            tile_A A;
            load_ldmatrix_trans(A, tile_V + 2*k0*D2_padded + i_VKQ_0/2, D2_padded);
            if (ntiles == 1) {
                mma(VKQ_C[i_VKQ_0/tile_C_VKQ::I], A, B[k00/(np*tile_A::J)]);
            } else {
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
                    // Wide version of VKQ_C is column-major => swap A and B.
                    mma(VKQ_C_16[i_VKQ_0/tile_C_VKQ::I * ntiles/2 + t], B_16[k00/(np*tile_A::J) * ntiles/2 + t], A);
                }
            }
        }
    }

#ifndef CP_ASYNC_AVAILABLE
    __syncthreads(); // Only needed if tile_K == tile_V.
#endif // CP_ASYNC_AVAILABLE

#else
    GGML_UNUSED(Q_f2); GGML_UNUSED(K_h2); GGML_UNUSED(V_h2);
    GGML_UNUSED(mask_h2); GGML_UNUSED(dstk); GGML_UNUSED(dstk_fixup);
    GGML_UNUSED(scale); GGML_UNUSED(slope); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(stride_KV);
    GGML_UNUSED(stride_mask); GGML_UNUSED(jt); GGML_UNUSED(tile_K);
    GGML_UNUSED(stride_mask); GGML_UNUSED(jt); GGML_UNUSED(tile_K);
    GGML_UNUSED(tile_V); GGML_UNUSED(tile_mask); GGML_UNUSED(Q_B);
    GGML_UNUSED(VKQ_C); GGML_UNUSED(KQ_max); GGML_UNUSED(KQ_rowsum);
    GGML_UNUSED(kb0);
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template<int D, int ncols1, int ncols2, int nwarps, int KQ_per_iter, int ntiles, bool use_logit_softcap, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half2  * const __restrict__ mask_h2,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const int ne01,
        const int ne02,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_KV,
        const int stride_mask,
        const int jt,
        const int kb0_start,
        const int kb0_stop) {
#ifdef NEW_MMA_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int ncols           = ncols1 * ncols2;
    constexpr int cols_per_warp   = ntiles * tile_B::I;
    constexpr int cols_per_thread = ntiles == 1 ? 2 : ntiles;
    constexpr int np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    static_assert(D           % nwarps == 0, "bad D");
    static_assert(KQ_per_iter % nwarps == 0, "bad KQ_per_iter");

    constexpr int D2_padded = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.

    // Temporary shared buffer for loading K/V data with KQ_per_iter*D logical elements:
    extern __shared__ half2 tile_K[];
#ifdef CP_ASYNC_AVAILABLE
    half2 * tile_V    = tile_K + KQ_per_iter*D2_padded;
#else
    half2 * tile_V    = tile_K;
#endif // CP_ASYNC_AVAILABLE
    half2 * tile_mask = tile_V + KQ_per_iter*D2_padded;

    tile_B       Q_B[D/(2*tile_B::J) * ntiles];
    tile_C_VKQ VKQ_C[D/tile_C_VKQ::I * ntiles];

    tile_B_16     * Q_B_16   = (tile_B_16     *) Q_B;
    tile_C_VKQ_16 * VKQ_C_16 = (tile_C_VKQ_16 *) VKQ_C;

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Temporarily load Q data into tile_K, will be loaded into registers afterwards.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start  = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
        const int k0_stop   =                             D/2 - (D/2) % (1*stride_k);
        const int stride_jc = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if (jt*ncols1 + j < ne01) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_K[jc*D2_padded + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_K[jc*D2_padded + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < D/2; k0 += tile_B::J) {
            if (ntiles == 1) {
                load_ldmatrix(Q_B[k0/tile_B::J], tile_K + j0*D2_padded + k0, D2_padded);
            } else {
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
                    load_ldmatrix(Q_B_16[k0/tile_B_16::J * ntiles/2 + t],
                        tile_K + (j0 + t*tile_B_16::I)*D2_padded + k0, D2_padded);
                }
            }
        }
    }

    __syncthreads();

    // Preload mask and K data for first iteration when using cp_async:
#ifdef CP_ASYNC_AVAILABLE
    if (ncols2 > 1 || mask_h2) {
        flash_attn_ext_f16_load_mask<ncols1, nwarps, KQ_per_iter>(mask_h2 + kb0_start*KQ_per_iter/2, tile_mask, stride_mask);
    }
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(K_h2 + kb0_start*KQ_per_iter*stride_KV, tile_K, stride_KV);
#endif // CP_ASYNC_AVAILABLE

    // Iterate over ne11 == previous tokens:
    for (int kb0 = kb0_start; kb0 < kb0_stop-1; ++kb0) {
        constexpr bool last_iter = false;
        flash_attn_ext_f16_iter<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, mask_h2, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_KV, stride_mask, jt, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    }
    { // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
        constexpr bool last_iter = true;
        flash_attn_ext_f16_iter<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, mask_h2, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_KV, stride_mask, jt, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0_stop-1);
    }

    // With cp_async there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
#ifdef CP_ASYNC_AVAILABLE
    if (nwarps*cols_per_warp > KQ_per_iter) {
        __syncthreads();
    }
#endif // CP_ASYNC_AVAILABLE

    // Finally, sum up partial KQ rowsums.
    // The partial sums are spread across 8/4 threads each, does not need full reduce.
    {
        constexpr int offset_first = ntiles == 1 ? 16 : 2;
        constexpr int offset_last  = ntiles == 1 ?  4 : 1;
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, WARP_SIZE);
            }
        }
    }

    // Write VKQ accumulators to shared memory in column-major format.
    // It's faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // Also for np > 1 the combination is done via these values in shared memory.
    if (ntiles == 1) {
        const int jc_cwd = threadIdx.y*tile_B::I + tile_B::get_i(-1); // jc combine write data
#pragma unroll
        for (int k0 = 0; k0 < D/2; k0 += tile_B::J) {
            const tile_B B = get_transposed(VKQ_C[k0/tile_B::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
            for (int l = 0; l < tile_B::ne; ++l) {
                const int k = k0 + tile_B::get_j(l);

                tile_K[jc_cwd*D2_padded + k] = B.x[l];
            }
        }
    } else {
#pragma unroll
        for (int t = 0; t < ntiles/2; ++t) {
            const int j0 = threadIdx.y*cols_per_warp + t*tile_C_VKQ_16::I;
#pragma unroll
            for (int k0 = 0; k0 < D/2; k0 += tile_C_VKQ_16::J) {
#pragma unroll
                for (int l = 0; l < tile_C_VKQ_16::ne; ++l) {
                    const int j = j0 + tile_C_VKQ_16::get_i(l);
                    const int k = k0 + tile_C_VKQ_16::get_j(l);

                    tile_K[j*D2_padded + k] = VKQ_C_16[k0/tile_C_VKQ_16::J * ntiles/2 + t].x[l];
                }
            }
        }
    }

    if constexpr (ntiles == 1) {
        const int jc_cwmo = (threadIdx.x % (2*tile_C_VKQ::J)) / tile_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*tile_C_VKQ::J) + 2*tile_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*tile_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_K)[jc_cwm*(D2_padded/2) + D/4] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < tile_B::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < tile_B::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        static_assert(ntiles == 2 || ntiles == 4, "bad ntiles");
        const int jc_cwm = threadIdx.y*cols_per_warp // jc combine write meta
            + (ntiles == 4 ? ((threadIdx.x % 4) / 2) * tile_C_VKQ_16::I : 0)
            + tile_C_VKQ_16::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && (ntiles == 4 || threadIdx.x % 4 < cols_per_thread)) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_K)[jc_cwm*(D2_padded/2) + D/4] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && (ntiles == 4 || threadIdx.x % 4 < ntiles)) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && (ntiles == 4 || threadIdx.x % 4 < ntiles)) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    static_assert(np == 1 || ntiles == 1 || ntiles == 2, "bad ntiles");
    if (np > 1 && threadIdx.y % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        constexpr int nmeta = np*cols_per_warp >= WARP_SIZE ? np*cols_per_warp/WARP_SIZE : 1;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < WARP_SIZE ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_K) + jc_meta*(D2_padded/2) + D/4;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * WARP_SIZE * D2_padded/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset >= WARP_SIZE) {
                continue;
            }
            KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
        }

        float KQ_cms[nmeta]; // KQ combine max scale per warp.
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset >= WARP_SIZE) {
                continue;
            }
            KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
        }

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= WARP_SIZE || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * WARP_SIZE * D2_padded/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        // Combined KQ max + rowsum.
        static_assert(cols_per_warp <= WARP_SIZE);
        if (needs_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    }

    if (np > 1) {
        __syncthreads();
    }

    if (np == 1 || threadIdx.y % np == 0) {
        // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
        // The values after that are for the partial results of the individual blocks.
        float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(D/2));

#pragma unroll
        for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
            const int k0_start  = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
            const int k0_stop   =                             D/2 - (D/2) % (1*stride_k);
            const int stride_jc = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                continue;
            }

#pragma unroll
            for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                    break;
                }

                const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                const int j_dst = jc_dst / ncols2;
                const int c_dst = jc_dst % ncols2;

                if (!is_fixup && jt*ncols1 + j_dst >= ne01) {
                    continue;
                }

                const float * meta_j = (const float *) tile_K + jc_tile_K*D2_padded + D/2;
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                    for (int ip = 0; ip < np; ++ip) {
                        const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * D2_padded + 0];
                        const float2 dstk_val_add = __half22float2(tile_K[(jc_tile_K + ip*cols_per_warp) * D2_padded + k]);
                        dstk_val.x += dstk_val_add.x*KQ_crs;
                        dstk_val.y += dstk_val_add.y*KQ_crs;
                    }

                    if (!needs_fixup && !is_fixup) {
                        const float KQ_rowsum_j = meta_j[1];
                        dstk_val.x /= KQ_rowsum_j;
                        dstk_val.y /= KQ_rowsum_j;
                    }

                    if (is_fixup) {
                        dstk_fixup_data[jc_dst*(D/2) + k] = dstk_val;
                    } else {
                        dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(D/2) + k] = dstk_val;
                    }
                }
            }
        }
    }

    if (np > 1) {
        __syncthreads();
    }
#else
    GGML_UNUSED(Q_f2); GGML_UNUSED(K_h2); GGML_UNUSED(V_h2);
    GGML_UNUSED(mask_h2); GGML_UNUSED(dstk); GGML_UNUSED(dstk_fixup);
    GGML_UNUSED(scale); GGML_UNUSED(slope); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(stride_Q1);
    GGML_UNUSED(stride_Q2); GGML_UNUSED(stride_KV); GGML_UNUSED(stride_mask);
    GGML_UNUSED(jt); GGML_UNUSED(kb0_start); GGML_UNUSED(kb0_stop);
    NO_DEVICE_CODE;
#endif // NEW_MMA_AVAILABLE
}

template<int D, int ncols1, int ncols2, int nwarps, int KQ_per_iter, int ntiles, bool use_logit_softcap>
__launch_bounds__(nwarps*WARP_SIZE, 2)
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
#if defined(FLASH_ATTN_AVAILABLE) && defined(NEW_MMA_AVAILABLE)

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(FATTN_KQ_STRIDE % KQ_per_iter == 0, "bad KQ_per_iter");

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_KV   = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half2);

    const int iter_k = ne11 / FATTN_KQ_STRIDE;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

    constexpr int kb_niter = FATTN_KQ_STRIDE / KQ_per_iter; // Number of kernel iterations per assigned KQ slice.

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = (blockIdx.x + 0)*iter_k*iter_j*(ne02/ncols2) / gridDim.x;
    const int kbc_stop = (blockIdx.x + 1)*iter_k*iter_j*(ne02/ncols2) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int channel = kbc / (iter_k*iter_j);
        const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

        const float2 * Q_f2    = (const float2 *) (Q + nb02* channel*ncols2);
        const half2  * K_h2    = (const half2  *) (K + nb12*(channel*ncols2 / gqa_ratio));
        const half2  * V_h2    = (const half2  *) (V + nb12*(channel*ncols2 / gqa_ratio)); // K and V have same shape
        const half2  * mask_h2 = ncols2 > 1 || mask ? (const half2  *) mask + (nb31/sizeof(half2))*jt*ncols1 : nullptr;
        float2       * dstk    = ((float2 *) dst) + channel*(ncols2 * D/2);

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, channel, n_head_log2, m0, m1) : 1.0f;

        const int kb0_start_kernel = kb0_start * kb_niter;
        const int kb0_stop_kernel  = kb0_stop  * kb_niter;

        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, stride_Q1, stride_Q2, stride_KV, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is working on the beginning of a tile.
            flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, stride_Q1, stride_Q2, stride_KV, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int channel = kbc / (iter_k*iter_j);
    const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

    const float2 * Q_f2    = (const float2 *) (Q + nb02* channel*ncols2);
    const half2  * K_h2    = (const half2  *) (K + nb12*(channel*ncols2 / gqa_ratio));
    const half2  * V_h2    = (const half2  *) (V + nb12*(channel*ncols2 / gqa_ratio)); // K and V have same shape
    const half2  * mask_h2 = ncols2 > 1 || mask ? (const half2  *) mask + (nb31/sizeof(half2))*jt*ncols1 : nullptr;
    float2       * dstk    = ((float2 *) dst) + channel*(ncols2 * D/2);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, channel, n_head_log2, m0, m1) : 1.0f;

    const int kb0_start_kernel = kb0_start * kb_niter;
    const int kb0_stop_kernel  = kb0_stop  * kb_niter;

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h2, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, stride_Q1, stride_Q2, stride_KV, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
    GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap); GGML_UNUSED(ne00);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03); GGML_UNUSED(ne10);
    GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13); GGML_UNUSED(ne31);
    GGML_UNUSED(nb31); GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13); GGML_UNUSED(nb21);
    GGML_UNUSED(nb22); GGML_UNUSED(nb23); GGML_UNUSED(ne0); GGML_UNUSED(ne1);
    GGML_UNUSED(ne2); GGML_UNUSED(ne3);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(NEW_MMA_AVAILABLE)
}

template <int D, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int ncols         = ncols1 * ncols2;
    constexpr int KQ_per_iter   = D <= 128 && ncols1 <= 64 ? 64 : 32;
    constexpr int nwarps        = (KQ_per_iter == 32 && ncols <= 16) ? 2 : 4;
    constexpr int ntiles        = ncols <= 8 ? 1 : (ncols <= 64 ? 2 : 4);
    constexpr int cols_per_warp = ntiles * tile_B::I;

    static_assert(D     %    tile_B::J  == 0, "bad D");
    static_assert(ncols % cols_per_warp == 0, "bad ncols");

    const ggml_tensor * KQV = dst;
    const int id    = ggml_cuda_get_device();
    const int cc    = ggml_cuda_info().devices[id].cc;

    const int KQ_shared_rows = cp_async_available(cc) ? 2*KQ_per_iter : KQ_per_iter;

    const size_t nbytes_shared_KV      = KQ_shared_rows       * (D           + 8) * sizeof(half);
    const size_t nbytes_shared_mask    = ncols1               * (KQ_per_iter + 8) * sizeof(half);
    const size_t nbytes_shared_combine = nwarps*cols_per_warp * (D           + 8) * sizeof(half);

    const size_t nbytes_shared_total = std::max(nbytes_shared_KV + nbytes_shared_mask, nbytes_shared_combine);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap>;
    }

    launch_fattn<D, ncols1, ncols2, KQ_per_iter>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, FATTN_KQ_STRIDE, true, true, true);
}


#define DECL_FATTN_MMA_F16_CASE(D, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                     \
    <D, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(D, ncols) \
    extern DECL_FATTN_MMA_F16_CASE(D, (ncols)/1, 1); \
    extern DECL_FATTN_MMA_F16_CASE(D, (ncols)/2, 2); \
    extern DECL_FATTN_MMA_F16_CASE(D, (ncols)/4, 4); \
    extern DECL_FATTN_MMA_F16_CASE(D, (ncols)/8, 8); \

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256,   8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256,  16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256,  32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256,  64)

// Kernels with ncols == 128 are only 4% faster due to register pressure.
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 128) // Needs too much shared memory.
