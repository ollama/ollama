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

// Config options for specific head sizes.
// Should not affect results, only speed/register pressure/shared memory use.
//
// nbatch_fa:      number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
// nwarps_max:     maximum number of warps per CUDA block, up to 8 warps in total can run per SM (given enough shared memory).
// Q_in_reg:       whether the Q values should be kept permanently in registers.
// nstages_target: targeted number of pipeline stages for cp_async (if available), 0 means synchronous data loading.
// nbatch_K2:      number of K half2 values in direction of DKQ to load in parallel.
// nbatch_V2:      number of V half2 values in direction of DV to load in parallel.
// nbatch_combine: number of VKQ half2 values in direction of DV to combine in parallel.

template <int DKQ, int DV>
struct fattn_mma_f16_config;

template <>
struct fattn_mma_f16_config< 64,  64> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static int get_nbatch_K2_host(const int /*cc*/, const int /*ncols*/) {
        return 32;
    }

    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 32;
    }

    static int get_nbatch_V2_host(const int /*cc*/, const int /*ncols*/) {
        return 32;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 32;
    }

    static int get_nbatch_combine_host(const int /*cc*/, const int /*ncols*/) {
        return 32;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 32;
    }
};

template <>
struct fattn_mma_f16_config< 80,  80> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static int get_nbatch_K2_host(const int /*cc*/, const int /*ncols*/) {
        return 40;
    }

    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 40;
    }

    static int get_nbatch_V2_host(const int /*cc*/, const int /*ncols*/) {
        return 40;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 40;
    }

    static int get_nbatch_combine_host(const int /*cc*/, const int /*ncols*/) {
        return 40;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 40;
    }
};

template <>
struct fattn_mma_f16_config< 96,  96> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static int get_nbatch_K2_host(const int /*cc*/, const int /*ncols*/) {
        return 48;
    }

    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 48;
    }

    static int get_nbatch_V2_host(const int /*cc*/, const int /*ncols*/) {
        return 48;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 48;
    }

    static int get_nbatch_combine_host(const int /*cc*/, const int /*ncols*/) {
        return 48;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 48;
    }
};

template <>
struct fattn_mma_f16_config<112, 112> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static int get_nbatch_K2_host(const int /*cc*/, const int /*ncols*/) {
        return 56;
    }

    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 56;
    }

    static int get_nbatch_V2_host(const int /*cc*/, const int /*ncols*/) {
        return 56;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 56;
    }

    static int get_nbatch_combine_host(const int /*cc*/, const int /*ncols*/) {
        return 56;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 56;
    }
};

template <>
struct fattn_mma_f16_config<128, 128> {
    static constexpr int  nbatch_fa      = 64;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static int get_nbatch_K2_host(const int /*cc*/, const int /*ncols*/) {
        return 64;
    }

    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 64;
    }

    static int get_nbatch_V2_host(const int /*cc*/, const int /*ncols*/) {
        return 64;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 64;
    }

    static int get_nbatch_combine_host(const int /*cc*/, const int /*ncols*/) {
        return 64;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 64;
    }
};

template <>
struct fattn_mma_f16_config<256, 256> {
    static constexpr int  nbatch_fa      = 32;
    static constexpr int  nwarps_max     = 4;
    static constexpr bool Q_in_reg       = true;
    static constexpr int  nstages_target = 2;

    static int get_nbatch_K2_host(const int /*cc*/, const int /*ncols*/) {
        return 128;
    }

    static constexpr __device__ int get_nbatch_K2_device(int /*ncols*/) {
        return 128;
    }

    static int get_nbatch_V2_host(const int /*cc*/, const int /*ncols*/) {
        return 128;
    }

    static constexpr __device__ int get_nbatch_V2_device(int /*ncols*/) {
        return 128;
    }

    static int get_nbatch_combine_host(const int cc, const int ncols) {
        if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) {
            return ncols <= 16 ? 128 : 64;
        }
        return 64;
    }

    static constexpr __device__ int get_nbatch_combine_device(int ncols) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
        return ncols <= 16 ? 128 : 64;
#else
        GGML_UNUSED(ncols);
        return 128;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    }
};

template <>
struct fattn_mma_f16_config<576, 512> {
    static constexpr int  nbatch_fa      = 32;
    static constexpr int  nwarps_max     = 8;
    static constexpr bool Q_in_reg       = false;
    static constexpr int  nstages_target = 1;

    static int get_nbatch_K2_host(const int cc, const int ncols) {
        if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) {
            return ncols <= 16 ? 96 : 160;
        }
        return ncols <= 16 ? 288 : 160;
    }

    static constexpr __device__ int get_nbatch_K2_device(int ncols) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
        return ncols <= 16 ? 96 : 160;
#else
        return ncols <= 16 ? 288 : 160;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    }

    static int get_nbatch_V2_host(const int cc, const int ncols) {
        if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) {
            return ncols <= 16 ? 64 : 128;
        }
        return ncols <= 16 ? 256 : 128;
    }

    static constexpr __device__ int get_nbatch_V2_device(int ncols) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
        return ncols <= 16 ? 64 : 128;
#else
        return ncols <= 16 ? 256 : 128;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    }

    static int get_nbatch_combine_host(const int /*cc*/, const int /*ncols*/) {
        return 128;
    }

    static constexpr __device__ int get_nbatch_combine_device(int /*ncols*/) {
        return 128;
    }
};

// ------------------------------------------------------------------------------------------------------------------

template<int stride_tile, int nwarps, int nbatch_fa, bool use_cp_async>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV) {

    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    // The minimum granularity with cp.async is 16 bytes, with synchronous data loading it's 4 bytes.

    if (use_cp_async) {
        constexpr int preload = 64;
        constexpr int h2_per_chunk = 16/sizeof(half2);
        const int chunks_per_row = D2 / h2_per_chunk;

        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);

        auto load = [&] __device__ (auto n) {
            const int stride_k = WARP_SIZE >> n;
            const int k0_start = stride_k == WARP_SIZE ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV + i*stride_KV + k*h2_per_chunk);
                }
            }
        };
        ggml_cuda_unroll<5>{}(load);
    } else {
        static_assert(nbatch_fa % (4*nwarps) == 0, "out of bounds");
        auto load = [&] __device__ (const int n) {
            const int stride_k = WARP_SIZE >> n;
            const int k0_start = stride_k == WARP_SIZE ? 0 : D2 - D2 % (2*stride_k);
            const int k0_stop  =                             D2 - D2 % (1*stride_k);
            const int stride_i = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_KV[i*stride_tile + k] = KV[i*stride_KV + k];
                }
            }
        };
        ggml_cuda_unroll<3>{}(load);
    }
}

template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half2 * const __restrict__ mask_h2, half2 * const __restrict__ tile_mask, const int stride_mask) {
    static_assert(nbatch_fa == 2*WARP_SIZE || WARP_SIZE % nbatch_fa == 0, "bad KQ_per_iter");

    if (use_cp_async) {
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j0 = 0; j0 < ncols1; j0 += stride_j) {
            const int j = j0 + threadIdx.y*cols_per_warp +
                (nbatch_fa == 2*WARP_SIZE ? threadIdx.x / (WARP_SIZE/4) : threadIdx.x / (WARP_SIZE/cols_per_warp));

            if (j0 + stride_j > ncols1 && j >= ncols1) {
                break;
            }

            const int i = 4 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half2), mask_h2 + j*stride_mask + i);
        }
        return;
    }

    constexpr int cols_per_warp = 2*WARP_SIZE/nbatch_fa;
    constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
    for (int j0 = 0; j0 < ncols1; j0 += stride_j) {
        const int j = j0 + threadIdx.y*cols_per_warp + (nbatch_fa == 2*WARP_SIZE ? 0 : threadIdx.x / (WARP_SIZE/cols_per_warp));

        if (j0 + stride_j > ncols1 && j >= ncols1) {
            break;
        }

        const int i = nbatch_fa == 2*WARP_SIZE ? threadIdx.x : threadIdx.x % (WARP_SIZE/cols_per_warp);

        tile_mask[j*(nbatch_fa/2 + 4) + i] = mask_h2[j*stride_mask + i];
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, int ntiles,
    bool use_logit_softcap, bool mla, bool needs_fixup, bool is_fixup, bool last_iter>
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
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half2        * const __restrict__ tile_mask,
        const tile_B * const __restrict__ Q_B,
        tile_C_VKQ   * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int kb0) {
#ifdef TURING_MMA_AVAILABLE
    typedef fattn_mma_f16_config<DKQ, DV> c;

#ifdef CP_ASYNC_AVAILABLE
    constexpr int nstages = c::nstages_target;
#else
    constexpr int nstages = 0;
#endif // CP_ASYNC_AVAILABLE

    constexpr int cols_per_warp   = ntiles * tile_B::I;
    constexpr int cols_per_thread = ntiles == 1 ? 2 : ntiles;
    constexpr int np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
    constexpr int ncols           = ncols1 * ncols2;
    constexpr int nbatch_K2       = c::get_nbatch_K2_device(ncols);
    constexpr int nbatch_V2       = c::get_nbatch_V2_device(ncols);

    constexpr int stride_tile_Q = DKQ/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;

    static_assert(!mla || nbatch_K2 >= nbatch_V2, "bad nbatch_K2, nbatch_V2 for MLA");
    constexpr int stride_tile_V = mla ? stride_tile_K : nbatch_V2 + 4;

    const int k_VKQ_0 = kb0 * c::nbatch_fa;
    tile_C_KQ KQ_C[c::nbatch_fa/(np*tile_C_KQ::I) * ntiles];

    // Use wide variants of tiles if ntiles >= 2.
    tile_B_16     * Q_B_16   = (tile_B_16     *) Q_B;
    tile_C_VKQ_16 * VKQ_C_16 = (tile_C_VKQ_16 *) VKQ_C;
    tile_C_KQ_16  * KQ_C_16  = (tile_C_KQ_16  *) KQ_C;

    if constexpr (nstages > 1) {
        static_assert(!mla, "multi-stage loading not implemented for MLA");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, c::nbatch_fa, use_cp_async>
            (V_h2 + int64_t(k_VKQ_0)*stride_V, tile_V, nbatch_V2, stride_V);
    } else {
        constexpr bool use_cp_async = nstages == 1;
        if (ncols2 > 1 || mask_h2) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, c::nbatch_fa, use_cp_async>(mask_h2 + k_VKQ_0/2, tile_mask, stride_mask);
        }
    }

#pragma unroll
    for (int k0_start = 0; k0_start < DKQ/2; k0_start += nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;
        const int k0_diff = k0_stop - k0_start;

        if (nstages <= 1) {
            constexpr bool use_cp_async = nstages == 1;
            flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, c::nbatch_fa, use_cp_async>
                (K_h2 + int64_t(k_VKQ_0)*stride_K + k0_start, tile_K, k0_diff, stride_K);
            if (use_cp_async) {
                cp_async_wait_all();
            }
            __syncthreads();
        }

        // Calculate tile of KQ:
        if constexpr (c::Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < c::nbatch_fa; i_KQ_00 += np*tile_A::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*tile_A::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += tile_A::J) {
                    tile_A K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
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
        } else {
            static_assert(ntiles == 2, "ntiles != 2 not implemented");
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += tile_A::J) {
                load_ldmatrix(Q_B_16[0], tile_Q + (threadIdx.y / np)*(tile_B_16::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < c::nbatch_fa; i_KQ_00 += np*tile_A::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*tile_A::I;

                    tile_A K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);

                    // Wide version of KQ_C is column-major => swap A and B.
                    mma(KQ_C_16[i_KQ_00/(np*tile_A::I)], Q_B_16[0], K_A);
                }
            }
        }

        if (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }

    if (use_logit_softcap) {
        static_assert(c::nbatch_fa % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < c::nbatch_fa/(np*tile_C_KQ::I) * ntiles; ++i) {
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
            for (int i00 = 0; i00 < c::nbatch_fa; i00 += np*tile_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*tile_C_KQ::I;
#pragma unroll
                for (int l = 0; l < tile_C_KQ::ne; ++l) {
                    const int i = i0 + tile_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*tile_C_KQ::J + tile_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*tile_C_KQ::I)].x[l] += slope *
                        __half2float(((const half *) tile_mask)[j*(c::nbatch_fa + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(c::nbatch_fa % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*tile_C_KQ::I); ++k) {
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

        static_assert(c::nbatch_fa % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*tile_C_KQ::I); ++k) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_C[k].x[l] = expf(KQ_C[k].x[l] - KQ_max_new[l % 2]);

                KQ_rowsum_add[l % 2] += KQ_C[k].x[l];
            }
        }
    } else { // ntiles > 1
        if (ncols2 > 1 || mask_h2) {
#pragma unroll
            for (int i00 = 0; i00 < c::nbatch_fa; i00 += np*tile_C_KQ_16::J) {
                const int i0 = i00 + (threadIdx.y % np)*tile_C_KQ_16::J;
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_KQ_16::ne; l0 += 2) {
                        const int i = (i0 + tile_C_KQ_16::get_j(l0)) / 2;
                        const int j = ((threadIdx.y / np)*cols_per_warp + t*tile_C_KQ_16::I + tile_C_KQ_16::get_i(l0)) / ncols2;

                        const float2 tmp = __half22float2(tile_mask[j*(c::nbatch_fa/2 + 4) + i]);
                        const int KQ_index = i00/(np*tile_C_KQ_16::J) * ntiles/2 + t;
                        KQ_C_16[KQ_index].x[l0 + 0] += slope*tmp.x;
                        KQ_C_16[KQ_index].x[l0 + 1] += slope*tmp.y;
                    }
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(c::nbatch_fa % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*tile_C_KQ_16::J); ++k) {
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

        static_assert(c::nbatch_fa % (np*tile_C_KQ_16::J) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*tile_C_KQ_16::J); ++k) {
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
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

        if (ntiles == 1) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < DV/tile_C_VKQ::I; ++i) {
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
                for (int i = 0; i < DV/tile_C_VKQ_16::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_VKQ_16::ne; l0 += 2) {
                        VKQ_C_16[i*ntiles/2 + col/2].x[l0 + col % 2] *= KQ_max_scale_h2;
                    }
                }
            }
        }
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    tile_B B[c::nbatch_fa/(np*2*tile_B::J) * ntiles];
    tile_B_16 * B_16 = (tile_B_16 *) B;
    static_assert(c::nbatch_fa % (np*2*tile_B::J) == 0, "bad loop size");
    if (ntiles == 1) {
#pragma unroll
        for (int k = 0; k < c::nbatch_fa/(np*2*tile_B::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < c::nbatch_fa/(np*2*tile_B_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
                B_16[k*ntiles/2 + t] = get_half2(KQ_C_16[k*ntiles/2 + t]);
            }
        }
    }

    if (nstages > 1) {
        // Preload K tile for next iteration:
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if (!last_iter) {
            if (ncols2 > 1 || mask_h2) {
                flash_attn_ext_f16_load_mask<ncols1, nwarps, c::nbatch_fa, use_cp_async>
                    (mask_h2 + (k_VKQ_0 + c::nbatch_fa)/2, tile_mask, stride_mask);
            }
            flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, c::nbatch_fa, use_cp_async>
                (K_h2 + int64_t(k_VKQ_0 + c::nbatch_fa)*stride_K, tile_K, nbatch_K2, stride_K);
        }
    }


    // For MLA K and V have the same data.
    // Therefore, iterate over V in reverse and re-use the data if possible.
    static_assert(!mla || nstages <= 1, "combination of MLA and multi-stage loading not implemented");
    constexpr int reusable_cutoff = mla ? (DKQ - 1) - (DKQ - 1) % (2*nbatch_K2) - (DKQ - DV) : DV;
#pragma unroll
    for (int i0_stop = DV; i0_stop > 0; i0_stop -= 2*nbatch_V2) {
        const int i0_start = i0_stop - 2*nbatch_V2 > 0 ? i0_stop - 2*nbatch_V2 : 0;
        const int i0_diff  = i0_stop - i0_start;

        if (nstages <= 1 && i0_start < reusable_cutoff) {
            constexpr bool use_cp_async = nstages == 1;
            flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, c::nbatch_fa, use_cp_async>
                (V_h2 + int64_t(k_VKQ_0)*stride_V + i0_start/2, tile_V, i0_diff/2, stride_V);
            if (use_cp_async) {
                cp_async_wait_all();
            }
            __syncthreads();
        }
        const half2 * tile_V_i = i0_start < reusable_cutoff ? tile_V : tile_V + (i0_start - reusable_cutoff)/2;

        // Calculate VKQ tile:
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += tile_C_VKQ::I) {
            static_assert((c::nbatch_fa/2) % (np*tile_A::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < c::nbatch_fa/2; k00 += np*tile_A::J) {
                const int k0 = k00 + (threadIdx.y % np)*tile_A::J;

                tile_A A;
                load_ldmatrix_trans(A, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
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

        if (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }
#else
    GGML_UNUSED(Q_f2); GGML_UNUSED(K_h2); GGML_UNUSED(V_h2);
    GGML_UNUSED(mask_h2); GGML_UNUSED(dstk); GGML_UNUSED(dstk_fixup);
    GGML_UNUSED(scale); GGML_UNUSED(slope); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(stride_K); GGML_UNUSED(stride_V);
    GGML_UNUSED(stride_mask); GGML_UNUSED(tile_K);
    GGML_UNUSED(tile_V); GGML_UNUSED(tile_mask); GGML_UNUSED(Q_B);
    GGML_UNUSED(VKQ_C); GGML_UNUSED(KQ_max); GGML_UNUSED(KQ_rowsum);
    GGML_UNUSED(kb0); GGML_UNUSED(tile_Q);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, int ntiles, bool use_logit_softcap, bool mla, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half2  * const __restrict__ mask_h2,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const int ne01,
        const int ne02,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int kb0_start,
        const int kb0_stop) {
#ifdef TURING_MMA_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    typedef fattn_mma_f16_config<DKQ, DV> c;

#ifdef CP_ASYNC_AVAILABLE
    constexpr int nstages = c::nstages_target;
#else
    constexpr int nstages = 0;
#endif // CP_ASYNC_AVAILABLE

    constexpr int ncols           = ncols1 * ncols2;
    constexpr int cols_per_warp   = ntiles * tile_B::I;
    constexpr int cols_per_thread = ntiles == 1 ? 2 : ntiles;
    constexpr int np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
    constexpr int nbatch_K2       = c::get_nbatch_K2_device(ncols);
    constexpr int nbatch_V2       = c::get_nbatch_V2_device(ncols);

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = DKQ/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;

    static_assert(!mla || nbatch_K2 >= nbatch_V2, "bad nbatch_K2, nbatch_V2 for MLA");
    constexpr int stride_tile_V = mla ? stride_tile_K : nbatch_V2 + 4;
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = c::Q_in_reg ? tile_Q                                : tile_Q + ncols        * stride_tile_Q;
    half2 * tile_V    = nstages > 1 ? tile_K + c::nbatch_fa * stride_tile_K : tile_K;
    half2 * tile_mask = nstages > 1 ? tile_V + c::nbatch_fa * stride_tile_V : tile_V + c::nbatch_fa * stride_tile_KV_max;

    tile_B       Q_B[(c::Q_in_reg ? DKQ/(2*tile_B::J) : 1) * ntiles];
    tile_C_VKQ VKQ_C[DV/tile_C_VKQ::I  * ntiles];

    tile_B_16     * Q_B_16   = (tile_B_16     *) Q_B;
    tile_C_VKQ_16 * VKQ_C_16 = (tile_C_VKQ_16 *) VKQ_C;

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start  = stride_k == WARP_SIZE ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
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
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    if (c::Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += tile_B::J) {
            if (ntiles == 1) {
                load_ldmatrix(Q_B[k0/tile_B::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
            } else {
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
                    load_ldmatrix(Q_B_16[k0/tile_B_16::J * ntiles/2 + t],
                        tile_Q + (j0 + t*tile_B_16::I)*stride_tile_Q + k0, stride_tile_Q);
                }
            }
        }
    }

    __syncthreads();

    // Preload mask and K data for first iteration when using cp_async with multiple stages:
    if constexpr (nstages > 1) {
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi-stage pipeline");
        constexpr bool use_cp_async = true;
        if (ncols2 > 1 || mask_h2) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, c::nbatch_fa, use_cp_async>
                (mask_h2 + kb0_start*c::nbatch_fa/2, tile_mask, stride_mask);
        }
        flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, c::nbatch_fa, use_cp_async>
            (K_h2 + int64_t(kb0_start)*c::nbatch_fa*stride_K, tile_K, nbatch_K2, stride_K);
    }

    // Iterate over ne11 == previous tokens:
    int kb0 = kb0_start;
    for (; kb0 < kb0_stop-1; ++kb0) {
        constexpr bool last_iter = false;
        flash_attn_ext_f16_iter<DKQ, DV, ncols1, ncols2, nwarps, ntiles, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, mask_h2, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    }
    { // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
        constexpr bool last_iter = true;
        flash_attn_ext_f16_iter<DKQ, DV, ncols1, ncols2, nwarps, ntiles, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, mask_h2, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    }

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if (nstages > 1 && nwarps*cols_per_warp > c::nbatch_fa) {
        __syncthreads();
    }

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

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchonization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            static_assert(ntiles == 1 || ntiles == 2, "ntiles > 2 not implemented");
            const int jc = ntiles == 1 ? 2*tile_C_VKQ::get_j(col/2) + col % 2 : tile_C_VKQ_16::get_i(col);
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

        if (ntiles == 1) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < DV/tile_C_VKQ::I; ++i) {
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
                for (int i = 0; i < DV/tile_C_VKQ_16::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_VKQ_16::ne; l0 += 2) {
                        VKQ_C_16[i*ntiles/2 + col/2].x[l0 + col % 2] *= KQ_max_scale_h2;
                    }
                }
            }
        }
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int nbatch_combine = c::get_nbatch_combine_device(ncols);
    constexpr int tile_stride    = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    if constexpr (ntiles == 1) {
        const int jc_cwmo = (threadIdx.x % (2*tile_C_VKQ::J)) / tile_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*tile_C_VKQ::J) + 2*tile_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*tile_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
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
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
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
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * WARP_SIZE * tile_stride/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
            }
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
            if (offset < WARP_SIZE) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
            }
        }

        __syncthreads();

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= WARP_SIZE || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * WARP_SIZE * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
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
    } else if (np > 1) {
        // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
        // Therefore, all other warps also need to execute a __syncthreads().
        // Otherwise the points at which warps synchronize with each other would become misaligned.
        __syncthreads();
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if (ntiles == 1) {
            const int jc_cwd = threadIdx.y*tile_B::I + tile_B::get_i(-1); // jc combine write data
#pragma unroll
            for (int k0 = 0; k0 < nbatch_combine; k0 += tile_B::J) {
                const tile_B B = get_transposed(VKQ_C[(k00 + k0)/tile_B::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < tile_B::ne; ++l) {
                    const int k = k0 + tile_B::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
                const int j0 = threadIdx.y*cols_per_warp + t*tile_C_VKQ_16::I;
#pragma unroll
                for (int k0 = 0; k0 < nbatch_combine; k0 += tile_C_VKQ_16::J) {
#pragma unroll
                    for (int l = 0; l < tile_C_VKQ_16::ne; ++l) {
                        const int j = j0 + tile_C_VKQ_16::get_i(l);
                        const int k = k0 + tile_C_VKQ_16::get_j(l);

                        tile_Q[j*tile_stride + k] = VKQ_C_16[(k00 + k0)/tile_C_VKQ_16::J * ntiles/2 + t].x[l];
                    }
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
                const int k0_start  = stride_k == WARP_SIZE ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
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

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        if (np > 1) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED(Q_f2); GGML_UNUSED(K_h2); GGML_UNUSED(V_h2);
    GGML_UNUSED(mask_h2); GGML_UNUSED(dstk); GGML_UNUSED(dstk_fixup);
    GGML_UNUSED(scale); GGML_UNUSED(slope); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(stride_Q1);
    GGML_UNUSED(stride_Q2); GGML_UNUSED(stride_K); GGML_UNUSED(stride_V); GGML_UNUSED(stride_mask);
    GGML_UNUSED(jt); GGML_UNUSED(kb0_start); GGML_UNUSED(kb0_stop);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, int ntiles, bool use_logit_softcap, bool mla>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE)

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256)) {
        NO_DEVICE_CODE;
        return;
    }
#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

    static_assert(!mla || DKQ >= DV, "MLA needs DKQ >= DV");

    typedef fattn_mma_f16_config<DKQ, DV> c;

    static_assert(FATTN_KQ_STRIDE % fattn_mma_f16_config<DKQ, DV>::nbatch_fa == 0, "bad nbatch_fa");

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half2);

    const int stride_V = mla ? stride_K : nb21 / sizeof(half2);

    const int iter_k = ne11 / FATTN_KQ_STRIDE;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

    constexpr int kb_niter = FATTN_KQ_STRIDE / c::nbatch_fa; // Number of kernel iterations per assigned KQ slice.

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = (blockIdx.x + 0)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
    const int kbc_stop = (blockIdx.x + 1)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
        const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); // head in units of ncols2
        const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k; // j index of current tile.

        const int head0 = zt * ncols2;

        const float2 * Q_f2    = (const float2 *) (Q + nb03*sequence + nb02* head0);
        const half2  * K_h2    = (const half2  *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
        const half2  * mask_h2 = ncols2 == 1 && !mask ? nullptr :
            (const half2  *) (mask + nb33*(sequence % ne33) + nb31*jt*ncols1);
        float2       * dstk    = ((float2 *) dst) + (sequence*ne01*ne02 + head0) * (DV/2);

        const half2 * V_h2 = mla ? K_h2 + (DKQ/2 - DV/2) : (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));
        const float * sinks_f = sinks ? (const float *) sinks + head0 : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

        const int kb0_start_kernel = kb0_start * kb_niter;
        int       kb0_stop_kernel  = kb0_stop  * kb_niter;

        if (KV_max) {
            kb0_stop_kernel = min(kb0_stop_kernel, KV_max[sequence*iter_j + jt] / c::nbatch_fa);
        }

        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, ntiles, use_logit_softcap, mla, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is working on the beginning of a tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, ntiles, use_logit_softcap, mla, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
    const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); // head in units of ncols2
    const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k; // j index of current tile.

    const int head0 = zt * ncols2;

    const float2 * Q_f2    = (const float2 *) (Q + nb03*sequence + nb02* head0);
    const half2  * K_h2    = (const half2  *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half2  * mask_h2 = ncols2 == 1 && !mask ? nullptr :
        (const half2  *) (mask + nb33*(sequence % ne33) + nb31*jt*ncols1);
    float2       * dstk    = ((float2 *) dst) + (sequence*ne01*ne02 + head0) * (DV/2);

    const half2 * V_h2 = mla ? K_h2 + (DKQ/2 - DV/2) : (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));
    const float * sinks_f = sinks ? (const float *) sinks + head0 : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    const int kb0_start_kernel = kb0_start * kb_niter;
    int       kb0_stop_kernel  = kb0_stop  * kb_niter;

    if (KV_max) {
        kb0_stop_kernel = min(kb0_stop_kernel, KV_max[sequence*iter_j + jt] / c::nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, ntiles, use_logit_softcap, mla, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h2, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask); GGML_UNUSED(sinks);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta);
    GGML_UNUSED(scale); GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03);
    GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
    GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13);
    GGML_UNUSED(nb21); GGML_UNUSED(nb22); GGML_UNUSED(nb23);
    GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(ne33);
    GGML_UNUSED(nb31); GGML_UNUSED(nb32); GGML_UNUSED(nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE)
}

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    typedef fattn_mma_f16_config<DKQ, DV> c;

    const int nstages = cp_async_available(cc) ? c::nstages_target : 0;

    constexpr int ncols         = ncols1 * ncols2;
    constexpr int ntiles        = ncols <= 8 ? 1 : 2; // Number of tiles per warp.
    constexpr int cols_per_warp = ntiles * tile_B::I;
    constexpr int nwarps_max_x  = ncols / cols_per_warp;
    constexpr int nwarps_max_y  = c::nbatch_fa / tile_A::I;
    constexpr int nwarps        = nwarps_max_x*nwarps_max_y <= c::nwarps_max ? nwarps_max_x*nwarps_max_y : c::nwarps_max;

    constexpr bool mla = DKQ == 576;

    const int nbatch_K2      = c::get_nbatch_K2_host     (cc, ncols);
    const int nbatch_V2      = c::get_nbatch_K2_host     (cc, ncols);
    const int nbatch_combine = c::get_nbatch_combine_host(cc, ncols);

    static_assert(DKQ   % tile_B::J     == 0, "bad DKQ");
    static_assert(DV    % tile_A::J     == 0, "bad DV");
    static_assert(ncols % cols_per_warp == 0, "bad ncols");

    const size_t nbytes_shared_KV_1stage = c::nbatch_fa         * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = c::nbatch_fa         *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (c::nbatch_fa/2 + 4)                    * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, c::Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, nwarps, ntiles, use_logit_softcap, mla>;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(fattn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, nwarps, ntiles, use_logit_softcap, mla>;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(fattn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    }

    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, FATTN_KQ_STRIDE, true, true, true);
}


#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,   8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  64)

// The number of viable configurations for Deepseek is very limited:
extern DECL_FATTN_MMA_F16_CASE(576, 512, 1, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 2, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 4, 16);
