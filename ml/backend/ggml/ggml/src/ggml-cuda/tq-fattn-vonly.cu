// V-only TQ fused flash-attention kernel.
// K is raw f16; V is TQ-packed with rms scales and codebook for
// warp-shuffle decode. Supports D ∈ {64, 128, 256, 512}.
//
// Template parameters:
//   D                  — head dimension (64, 128, 256, or 512)
//   USE_LOGIT_SOFTCAP  — apply logit_softcap * tanh(sum) before softmax.
//                        Required for Gemma 2/3 (logit_softcap=50.0).
//                        Catastrophic PPL regression if omitted for these models.
//
// Output is in the WHT-rotated V coordinate system; caller applies
// WHTUndo to produce the final attention output (mirrors the existing
// V-only routing in kvcache/turboquant.go that pairs DequantV with
// WHTUndo for the materialised path).
//
// Scope: decode-only (curQueryLen==1 gated by kvcache/turboquant.go), single-block KV,
// no outliers/asymmetric primary, no softcap for non-Gemma models (USE_LOGIT_SOFTCAP=false).
// ncols=1 always; each block handles one Q head.
// Indexed locs[] and parallel flash-decode splitting are both supported via template params.

#include "tq-fattn-vonly.cuh"
#include "tq-fattn-vec.cuh"   // tq_decode_N_shfl
#include "fattn-common.cuh"

#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)

namespace {

// D-independent constants (same for all head dimensions).
constexpr int ncols        = 1;    // decode-only; one Q-column per block
constexpr int nthreads     = 128;
constexpr int nwarps       = nthreads / WARP_SIZE;          // 4
constexpr int cpy_nb       = 16;                            // ggml_cuda_get_max_cpy_bytes() on Volta+
constexpr int cpy_ne       = cpy_nb / 4;                    // 4
constexpr int nthreads_KQ  = nthreads / cpy_nb;             // 8
constexpr int nthreads_V   = nthreads / cpy_nb;             // 8
constexpr int V_rows_per_thread = 2 * cpy_ne;               // 8
constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;   // 4

static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_KQ");
static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

}  // anonymous namespace

// SINGLE_SPLIT: true=write dst directly (nSplits==1), false=write to partial_buf (nSplits>1).
// INDEXED: true=locs[] maps cells to physical V slots, false=contiguous firstCell+offset.
// Both are selected at the launch site (host code) so device kernels have no runtime
// branches on nSplits or locs — avoids sm_120 miscompile of kernel-arg comparisons.
template<int D, bool USE_LOGIT_SOFTCAP, bool SINGLE_SPLIT, bool INDEXED>
__launch_bounds__(128, 2)
__global__ void tq_fattn_vec_vonly_kernel(
    const float   * __restrict__ Q,
    const __half  * __restrict__ K,
    const uint8_t * __restrict__ V_packed,
    const __half  * __restrict__ mask,
    float         * __restrict__ dst,
    float         * __restrict__ partial_buf,  // [nSplits * nTiles * (D+2)] f32; only used when !SINGLE_SPLIT
    const float   * __restrict__ v_scales,
    const float   * __restrict__ v_codebook,
    const int32_t * __restrict__ locs,         // NULL = contiguous; [nCells] i32 = physical V slots
    int     v_bits,
    int     v_packedBytes,
    float   scale,
    float   logit_softcap,
    int     firstCell,
    int     nCells,
    int     nSplits,
    int     nHeadsQ,
    int     nKVHeads,
    int     nSeq,
    int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    int32_t mask_ne0, int32_t mask_nb1)
{
    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 64");
#ifdef FLASH_ATTN_AVAILABLE
    // This kernel is decode-only (curQueryLen==1 gate in kvcache/turboquant.go).
    // ncols is always 1 at launch; use constexpr to restore compile-time constant
    // folding that was present at 91dfa2dd — sm_120 miscompiles the runtime gridDim.x
    // form introduced in 8ec5c0a2 for D=256 (different SASS register allocation).
    constexpr int ncols = 1;
    const int ic0      = blockIdx.x;
    const int sequence = blockIdx.z / nHeadsQ;
    const int head     = blockIdx.z % nHeadsQ;
    const int gqa_ratio = nHeadsQ / nKVHeads;
    const int head_kv   = head / gqa_ratio;

    // Flash-decode split: blockIdx.y selects which slice of cells this block owns.
    // For SINGLE_SPLIT=true (nSplits==1 always): no arithmetic on nSplits in device code —
    // sm_120 miscompiles division/ternary on runtime kernel-arg integers even when the
    // value is constant at the call site. Compute split params at compile time instead.
    int split_i, kv_start, nCells_local;
    if constexpr (SINGLE_SPLIT) {
        split_i = 0;
        kv_start = 0;
        nCells_local = nCells;
    } else {
        split_i = (int)blockIdx.y;
        const int cells_per_split = (nCells + nSplits - 1) / nSplits;
        kv_start = split_i * cells_per_split;
        nCells_local = min(cells_per_split, nCells - kv_start);
    }

    // Adjust pointers to this (sequence, head, token, split) tile.
    const float * Q_tile = (const float *)((const char *)Q
        + (int64_t)nb_q03*sequence + (int64_t)nb_q02*head + (int64_t)nb_q01*ic0);
    const __half * K_seq = (const __half *)((const char *)K
        + nb_k13*sequence + (int64_t)nb_k12*head_kv + (int64_t)(firstCell + kv_start)*nb_k11);
    // V base for contiguous mode; indexed mode computes physical slot per-cell via locs[].
    const uint8_t * V_packed_base = V_packed
        + (int64_t)(firstCell + kv_start) * nKVHeads * v_packedBytes
        + (int64_t)head_kv   * v_packedBytes;
    const float * v_scales_base = v_scales
        + (int64_t)(firstCell + kv_start) * nKVHeads + head_kv;
    // locs is only accessed in the INDEXED=true path (if constexpr below).
    if constexpr (!INDEXED) { (void)locs; }
    // Mask row for this Q column; kv_start shifts into the split's cell range.
    const __half * maskh = mask ? (mask + (int64_t)mask_nb1/2*ic0 + kv_start) : nullptr;

    const float v_cb_lane = v_codebook[threadIdx.x & ((1 << v_bits) - 1)];

    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    // Shared memory layout:
    //   s_Q[D]                — Q pre-scaled
    //   s_KQ[nthreads]        — per-cell softmax weight scratch (reused)
    //   s_VKQ_warp[nwarps][D] — per-warp partial VKQ
    extern __shared__ float s_mem_all[];
    float * s_Q        = s_mem_all;
    float * s_KQ       = s_mem_all + D;
    float * s_VKQ_warp = s_mem_all + D + nthreads;

    // Load Q into shared memory, pre-scaled.
    for (int i = tid; i < D; i += nthreads) {
        s_Q[i] = Q_tile[i] * scale;
    }
    __syncthreads();

    // Each KQ-group lane handles D/nthreads_KQ stride-nthreads_KQ elements of
    // Q·K (e.g. lane 0 handles d ∈ {0, 8, 16, ..., D-8} for nthreads_KQ=8).
    const int tid_kq = threadIdx.x % nthreads_KQ;
    constexpr int K_PER_LANE = D / nthreads_KQ;

    // Per-thread VKQ accumulator: each lane owns (D/2)/nthreads_V pairs.
    //   D= 64 → 4 float2
    //   D=128 → 8 float2
    //   D=256 → 16 float2
    //   D=512 → 32 float2
    float2 VKQ[(D/2)/nthreads_V];
#pragma unroll
    for (int i = 0; i < (D/2)/nthreads_V; ++i) {
        VKQ[i].x = 0.0f;
        VKQ[i].y = 0.0f;
    }
    float KQ_max = -FLT_MAX/2.0f;
    float KQ_sum = 0.0f;

    // Main KV loop: process nthreads cells per pass, restricted to this split's range.
    for (int k_VKQ_0 = 0; k_VKQ_0 < nCells_local; k_VKQ_0 += nthreads) {

        // ---- Compute KQ row for this pass ----
        float KQ_reg = -FLT_MAX/2.0f;
        float KQ_max_new = KQ_max;

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE
                           + (threadIdx.x & ~(nthreads_KQ-1))
                           + i_KQ_0;
            const int cell_rel = k_VKQ_0 + i_KQ;
            const bool in_range = (cell_rel < nCells_local);

            const __half * K_row = K_seq + (int64_t)cell_rel*(nb_k11/2);

            float sum = 0.0f;
#pragma unroll
            for (int ki = 0; ki < K_PER_LANE; ++ki) {
                const int d = tid_kq + ki * nthreads_KQ;
                const float k_val = in_range ? __half2float(K_row[d]) : 0.0f;
                sum += s_Q[d] * k_val;
            }
            sum = warp_reduce_sum<nthreads_KQ>(sum);

            if (mask && in_range) {
                sum += __half2float(maskh[cell_rel]);
            }
            if (!in_range) sum = -FLT_MAX/2.0f;

            // Apply logit softcap before softmax (required for Gemma 2/3).
            // Must be applied BEFORE FATTN_KQ_MAX_OFFSET so the cap compresses
            // the logit range first and the offset provides headroom for the
            // compressed value.
            if constexpr (USE_LOGIT_SOFTCAP) {
                sum = logit_softcap * tanhf(sum);
            }

            KQ_max_new = fmaxf(KQ_max_new, sum + FATTN_KQ_MAX_OFFSET);

            if ((threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                KQ_reg = sum;
            }
        }

        // Reduce max across the warp (full warp), then update soft-max state.
#pragma unroll
        for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
            KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, offset, WARP_SIZE));
        }
        const float KQ_max_scale = expf(KQ_max - KQ_max_new);
        KQ_max = KQ_max_new;
        KQ_reg = expf(KQ_reg - KQ_max);
        KQ_sum = KQ_sum * KQ_max_scale + KQ_reg;
        s_KQ[tid] = KQ_reg;

#pragma unroll
        for (int i = 0; i < (D/2)/nthreads_V; ++i) {
            VKQ[i].x *= KQ_max_scale;
            VKQ[i].y *= KQ_max_scale;
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif

        // ---- Decode V and accumulate into VKQ ----
#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0
                        + (threadIdx.x / nthreads_V);

            const float KQ_k = s_KQ[k];

            const int cell_rel    = k_VKQ_0 + k;
            const bool v_in_range = (cell_rel < nCells_local);
            // Clamp v_cell_addr to 0 for out-of-range lanes. tq_decode_N_shfl
            // is unconditional across the warp (needed for shfl convergence),
            // so an unclamped cell_rel would dereference past V_packed end.
            // v_rms=0 discards the decoded value for out-of-range lanes.
            const int v_safe_cell_addr = v_in_range ? cell_rel : 0;
            // if constexpr selects indexed vs contiguous at compile time — no runtime branch
            // on locs pointer in device code (avoids sm_120 miscompile of pointer null-checks).
            const uint8_t * v_row;
            float v_rms;
            if constexpr (INDEXED) {
                const int v_phys = v_in_range ? locs[kv_start + v_safe_cell_addr] : 0;
                v_row = V_packed + (int64_t)v_phys * nKVHeads * v_packedBytes
                                 + (int64_t)head_kv * v_packedBytes;
                v_rms = v_in_range ? v_scales[v_phys * nKVHeads + head_kv] : 0.0f;
            } else {
                v_row = V_packed_base + (int64_t)v_safe_cell_addr * nKVHeads * v_packedBytes;
                v_rms = v_in_range ? v_scales_base[cell_rel * nKVHeads] : 0.0f;
            }

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                const int base_elem = 2*i_VKQ_0
                    + (threadIdx.x % nthreads_V) * V_rows_per_thread;

                float v_dec[V_rows_per_thread];
                tq_decode_N_shfl<V_rows_per_thread>(v_row, v_cb_lane, v_rms,
                                                     base_elem, v_bits, nthreads_V, v_dec);

#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                    VKQ[i_VKQ_0/nthreads_V + i_VKQ_1].x += v_dec[2*i_VKQ_1]   * KQ_k;
                    VKQ[i_VKQ_0/nthreads_V + i_VKQ_1].y += v_dec[2*i_VKQ_1+1] * KQ_k;
                }
            }
        }
    } // end KV loop

    // ---- Reduce across warps and write final output ----
    // Reduce KQ_max across warps via shared memory.
    __shared__ float KQ_max_shared[WARP_SIZE];
    __shared__ float KQ_sum_shared[WARP_SIZE];
    if (threadIdx.y == 0) {
        KQ_max_shared[threadIdx.x] = -FLT_MAX/2.0f;
        KQ_sum_shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        KQ_max_shared[threadIdx.y] = KQ_max;
    }
    __syncthreads();

    float KQ_max_global = KQ_max_shared[0];
#pragma unroll
    for (int w = 1; w < nwarps; ++w) {
        KQ_max_global = fmaxf(KQ_max_global, KQ_max_shared[w]);
    }
    const float warp_scale = expf(KQ_max - KQ_max_global);
    KQ_sum *= warp_scale;
#pragma unroll
    for (int i = 0; i < (D/2)/nthreads_V; ++i) {
        VKQ[i].x *= warp_scale;
        VKQ[i].y *= warp_scale;
    }
    // Reduce KQ_sum across warp lanes before writing the per-warp contribution.
    KQ_sum = warp_reduce_sum(KQ_sum);
    if (threadIdx.x == 0) {
        KQ_sum_shared[threadIdx.y] = KQ_sum;
    }
    __syncthreads();
    float KQ_sum_global = 0.0f;
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        KQ_sum_global += KQ_sum_shared[w];
    }
    // Accumulate per-warp VKQ contributions into shared memory.
    for (int i = tid; i < nwarps * D; i += nthreads) {
        s_VKQ_warp[i] = 0.0f;
    }
    __syncthreads();

    // VKQ output: write each VKQ[idx] to the d position the decode loop
    // filled it from. See the original D=128 kernel's comment for the
    // d-formula derivation (the decode loop lays out V_rows_per_thread
    // halves per thread per step in a 4-pair stripe with stride 8).
#pragma unroll
    for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
#pragma unroll
        for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
            const int d_base = 2*i_VKQ_0
                + (threadIdx.x % nthreads_V) * V_rows_per_thread
                + 2*i_VKQ_1;
            const int idx = i_VKQ_0/nthreads_V + i_VKQ_1;
            atomicAdd(&s_VKQ_warp[threadIdx.y * D + d_base + 0], VKQ[idx].x);
            atomicAdd(&s_VKQ_warp[threadIdx.y * D + d_base + 1], VKQ[idx].y);
        }
    }
    __syncthreads();

    const float inv_sum = 1.0f / KQ_sum_global;
    const int64_t out_base = ((int64_t)sequence*ncols + ic0)*nHeadsQ*D + (int64_t)head*D;

    // if constexpr selects the output path at compile time — no runtime branch in device code.
    // SINGLE_SPLIT=true: write normalized output to dst (no partial_buf, no reduce kernel).
    // SINGLE_SPLIT=false: write unnormalized VKQ+metadata to partial_buf for the reduce kernel.
    if constexpr (SINGLE_SPLIT) {
        for (int d = tid; d < D; d += nthreads) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                v += s_VKQ_warp[w * D + d];
            }
            dst[out_base + d] = v * inv_sum;
        }
    } else {
        const int nTiles = (int)gridDim.z * ncols;
        const int tile   = (int)blockIdx.z * ncols + ic0;
        float * pbuf = partial_buf + ((int64_t)split_i * nTiles + tile) * (D + 2);
        for (int d = tid; d < D; d += nthreads) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                v += s_VKQ_warp[w * D + d];
            }
            pbuf[d] = v;
        }
        __syncthreads();
        if (tid == 0) {
            pbuf[D]   = KQ_max_global;
            pbuf[D+1] = KQ_sum_global;
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V_packed, mask, dst, partial_buf, v_scales, v_codebook, locs,
                     v_bits, v_packedBytes, scale, logit_softcap, firstCell, nCells, nSplits,
                     nHeadsQ, nKVHeads, nSeq,
                     nb_q01, nb_q02, nb_q03, nb_k11, nb_k12, nb_k13,
                     mask_ne0, mask_nb1);
    NO_DEVICE_CODE;
#endif
}

// ── Single-split (nSplits==1) non-indexed decode kernel ───────────────────────
// Exact 91dfa2dd kernel body: namespace-level constexpr ncols (see anonymous
// namespace above), no locs, no INDEXED template parameter.  Restores the ptxas
// code-gen that worked on Blackwell sm_120.  Routes: nSplits==1 && locs==nullptr.
// Indexed (locs!=nullptr) and nSplits>1 paths use the full kernel above.
template<int D, bool USE_LOGIT_SOFTCAP>
__launch_bounds__(128, 2)
__global__ void tq_fattn_vec_vonly_kernel_ns(
    const float   * __restrict__ Q,
    const __half  * __restrict__ K,
    const uint8_t * __restrict__ V_packed,
    const __half  * __restrict__ mask,
    float         * __restrict__ dst,
    const float   * __restrict__ v_scales,
    const float   * __restrict__ v_codebook,
    int     v_bits,
    int     v_packedBytes,
    float   scale,
    float   logit_softcap,
    int     firstCell,
    int     nCells,
    int     nHeadsQ,
    int     nKVHeads,
    int     nSeq,
    int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    int32_t mask_ne0, int32_t mask_nb1)
{
    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 64");
#ifdef FLASH_ATTN_AVAILABLE
    // ncols = 1 from anonymous namespace; ic0 = blockIdx.x = 0 for decode.
    const int ic0      = blockIdx.x;
    const int sequence = blockIdx.z / nHeadsQ;
    const int head     = blockIdx.z % nHeadsQ;
    const int gqa_ratio = nHeadsQ / nKVHeads;
    const int head_kv   = head / gqa_ratio;

    const int nCells_local = nCells;

    const float * Q_tile = (const float *)((const char *)Q
        + (int64_t)nb_q03*sequence + (int64_t)nb_q02*head + (int64_t)nb_q01*ic0);
    const __half * K_seq = (const __half *)((const char *)K
        + nb_k13*sequence + (int64_t)nb_k12*head_kv + (int64_t)firstCell*nb_k11);
    const uint8_t * V_packed_base = V_packed
        + (int64_t)firstCell * nKVHeads * v_packedBytes
        + (int64_t)head_kv   * v_packedBytes;
    const float * v_scales_base = v_scales
        + (int64_t)firstCell * nKVHeads + head_kv;
    const __half * maskh = mask ? (mask + (int64_t)mask_nb1/2*ic0) : nullptr;

    const float v_cb_lane = v_codebook[threadIdx.x & ((1 << v_bits) - 1)];
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    extern __shared__ float s_mem_all[];
    float * s_Q        = s_mem_all;
    float * s_KQ       = s_mem_all + D;
    float * s_VKQ_warp = s_mem_all + D + nthreads;

    for (int i = tid; i < D; i += nthreads) {
        s_Q[i] = Q_tile[i] * scale;
    }
    __syncthreads();

    const int tid_kq = threadIdx.x % nthreads_KQ;
    constexpr int K_PER_LANE = D / nthreads_KQ;

    float2 VKQ[(D/2)/nthreads_V];
#pragma unroll
    for (int i = 0; i < (D/2)/nthreads_V; ++i) {
        VKQ[i].x = 0.0f;
        VKQ[i].y = 0.0f;
    }
    float KQ_max = -FLT_MAX/2.0f;
    float KQ_sum = 0.0f;

    for (int k_VKQ_0 = 0; k_VKQ_0 < nCells_local; k_VKQ_0 += nthreads) {
        float KQ_reg = -FLT_MAX/2.0f;
        float KQ_max_new = KQ_max;

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE
                           + (threadIdx.x & ~(nthreads_KQ-1))
                           + i_KQ_0;
            const int cell_rel = k_VKQ_0 + i_KQ;
            const bool in_range = (cell_rel < nCells_local);
            const __half * K_row = K_seq + (int64_t)cell_rel*(nb_k11/2);
            float sum = 0.0f;
#pragma unroll
            for (int ki = 0; ki < K_PER_LANE; ++ki) {
                const int d = tid_kq + ki * nthreads_KQ;
                const float k_val = in_range ? __half2float(K_row[d]) : 0.0f;
                sum += s_Q[d] * k_val;
            }
            sum = warp_reduce_sum<nthreads_KQ>(sum);
            if (mask && in_range) {
                sum += __half2float(maskh[cell_rel]);
            }
            if (!in_range) sum = -FLT_MAX/2.0f;
            if constexpr (USE_LOGIT_SOFTCAP) {
                sum = logit_softcap * tanhf(sum);
            }
            KQ_max_new = fmaxf(KQ_max_new, sum + FATTN_KQ_MAX_OFFSET);
            if ((threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                KQ_reg = sum;
            }
        }

#pragma unroll
        for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
            KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, offset, WARP_SIZE));
        }
        const float KQ_max_scale = expf(KQ_max - KQ_max_new);
        KQ_max = KQ_max_new;
        KQ_reg = expf(KQ_reg - KQ_max);
        KQ_sum = KQ_sum * KQ_max_scale + KQ_reg;
        s_KQ[tid] = KQ_reg;

#pragma unroll
        for (int i = 0; i < (D/2)/nthreads_V; ++i) {
            VKQ[i].x *= KQ_max_scale;
            VKQ[i].y *= KQ_max_scale;
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (threadIdx.x / nthreads_V);
            const float KQ_k = s_KQ[k];
            const int cell_rel    = k_VKQ_0 + k;
            const bool v_in_range = (cell_rel < nCells_local);
            const int v_safe_cell_addr = v_in_range ? cell_rel : 0;
            // Contiguous path (no locs): V_packed_base + cell offset.
            const uint8_t * v_row = V_packed_base
                + (int64_t)v_safe_cell_addr * nKVHeads * v_packedBytes;
            const float v_rms = v_in_range ? v_scales_base[cell_rel * nKVHeads] : 0.0f;
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                const int base_elem = 2*i_VKQ_0
                    + (threadIdx.x % nthreads_V) * V_rows_per_thread;
                float v_dec[V_rows_per_thread];
                tq_decode_N_shfl<V_rows_per_thread>(v_row, v_cb_lane, v_rms,
                                                     base_elem, v_bits, nthreads_V, v_dec);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                    VKQ[i_VKQ_0/nthreads_V + i_VKQ_1].x += v_dec[2*i_VKQ_1]   * KQ_k;
                    VKQ[i_VKQ_0/nthreads_V + i_VKQ_1].y += v_dec[2*i_VKQ_1+1] * KQ_k;
                }
            }
        }
    }

    __shared__ float KQ_max_shared[WARP_SIZE];
    __shared__ float KQ_sum_shared[WARP_SIZE];
    if (threadIdx.y == 0) {
        KQ_max_shared[threadIdx.x] = -FLT_MAX/2.0f;
        KQ_sum_shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        KQ_max_shared[threadIdx.y] = KQ_max;
    }
    __syncthreads();

    float KQ_max_global = KQ_max_shared[0];
#pragma unroll
    for (int w = 1; w < nwarps; ++w) {
        KQ_max_global = fmaxf(KQ_max_global, KQ_max_shared[w]);
    }
    const float warp_scale = expf(KQ_max - KQ_max_global);
    KQ_sum *= warp_scale;
#pragma unroll
    for (int i = 0; i < (D/2)/nthreads_V; ++i) {
        VKQ[i].x *= warp_scale;
        VKQ[i].y *= warp_scale;
    }
    KQ_sum = warp_reduce_sum(KQ_sum);
    if (threadIdx.x == 0) {
        KQ_sum_shared[threadIdx.y] = KQ_sum;
    }
    __syncthreads();
    float KQ_sum_global = 0.0f;
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        KQ_sum_global += KQ_sum_shared[w];
    }
    for (int i = tid; i < nwarps * D; i += nthreads) {
        s_VKQ_warp[i] = 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
#pragma unroll
        for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
            const int d_base = 2*i_VKQ_0
                + (threadIdx.x % nthreads_V) * V_rows_per_thread
                + 2*i_VKQ_1;
            const int idx = i_VKQ_0/nthreads_V + i_VKQ_1;
            atomicAdd(&s_VKQ_warp[threadIdx.y * D + d_base + 0], VKQ[idx].x);
            atomicAdd(&s_VKQ_warp[threadIdx.y * D + d_base + 1], VKQ[idx].y);
        }
    }
    __syncthreads();

    const float inv_sum = 1.0f / KQ_sum_global;
    const int64_t out_base = ((int64_t)sequence*ncols + ic0)*nHeadsQ*D + (int64_t)head*D;
    for (int d = tid; d < D; d += nthreads) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            v += s_VKQ_warp[w * D + d];
        }
        dst[out_base + d] = v * inv_sum;
    }
#else
    GGML_UNUSED_VARS(Q, K, V_packed, mask, dst, v_scales, v_codebook,
                     v_bits, v_packedBytes, scale, logit_softcap, firstCell, nCells,
                     nHeadsQ, nKVHeads, nSeq,
                     nb_q01, nb_q02, nb_q03, nb_k11, nb_k12, nb_k13,
                     mask_ne0, mask_nb1);
    NO_DEVICE_CODE;
#endif
}

// ── Flash-decode reduce kernel ───────────────────────────────────────────────
// Combines nSplits partial results written by tq_fattn_vec_vonly_kernel into
// the final normalized output. Each thread handles one D element.
//
// partial_buf layout: [nSplits * nTiles * (D+2)] f32
//   slot [s * nTiles + tile][0..D-1] — unnormalized VKQ for split s, tile
//   slot [s * nTiles + tile][D]      — KQ_max for split s
//   slot [s * nTiles + tile][D+1]    — KQ_sum for split s
//
// Grid: (ncols, 1, nHeadsQ * nSeq) — same as the main kernel z-dim.
// Block: (D, 1, 1) — one thread per head-dim element.
template<int D>
__global__ void tq_fattn_vonly_reduce_kernel(
    float       * __restrict__ dst,
    const float * __restrict__ partial_buf,
    int nSplits, int nTiles, int nHeadsQ, int ncols)
{
#ifdef FLASH_ATTN_AVAILABLE
    const int ic0      = blockIdx.x;
    const int tile     = (int)blockIdx.z * ncols + ic0;
    const int sequence = blockIdx.z / nHeadsQ;
    const int head     = blockIdx.z % nHeadsQ;
    const int d        = threadIdx.x;   // each thread owns one D element

    // Find global max across all splits (redundant across threads but cheap —
    // nSplits ≤ 16 and the reads are broadcast from L2).
    float max_global = -FLT_MAX / 2.0f;
    for (int s = 0; s < nSplits; ++s) {
        const float m = partial_buf[((int64_t)s * nTiles + tile) * (D + 2) + D];
        max_global = fmaxf(max_global, m);
    }

    // Accumulate VKQ_d and KQ_sum, rescaling each split to the global max.
    float VKQ_d    = 0.0f;
    float sum_total = 0.0f;
    for (int s = 0; s < nSplits; ++s) {
        const float * pbase = partial_buf + ((int64_t)s * nTiles + tile) * (D + 2);
        const float m     = pbase[D];
        const float sum   = pbase[D + 1];
        const float scale = expf(m - max_global);
        sum_total += sum * scale;
        VKQ_d     += pbase[d] * scale;
    }

    const int64_t out_base = ((int64_t)sequence * ncols + ic0) * nHeadsQ * D
                           + (int64_t)head * D;
    dst[out_base + d] = VKQ_d / sum_total;
#else
    GGML_UNUSED_VARS(dst, partial_buf, nSplits, nTiles, nHeadsQ, ncols);
    NO_DEVICE_CODE;
#endif
}

// ── Per-D extern-C launchers ─────────────────────────────────────────────────
// Go dispatches to the right D at the kvcache layer (headDim is known).
// Each launcher selects the USE_LOGIT_SOFTCAP template based on whether
// logit_softcap is non-zero at runtime.

// TQ_VONLY_LAUNCH dispatches to the correct template variant based on runtime values.
// All branching is host-code only — no device-code branches on kernel arguments.
//
// nSplits==1 (ss=true): uses tq_fattn_vec_vonly_kernel_ns (nosplit) which omits
// partial_buf and nSplits from its signature — restoring the 91dfa2dd-era register
// budget and fixing sm_120 ptxas register-allocation miscompile.
//
// nSplits>1 (ss=false): uses full tq_fattn_vec_vonly_kernel with partial_buf/nSplits.
// _ns: 91dfa2dd-exact kernel — no locs, no INDEXED template, ncols from namespace.
// Handles nSplits==1 && locs==nullptr (the common case for decode).
#define TQ_VONLY_CALL_NS(D_L, SC) \
    tq_fattn_vec_vonly_kernel_ns<D_L, SC><<<blocks, threads, smem, stream>>>( \
        Q, K, V_packed, mask, dst, v_scales, v_codebook, \
        v_bits, v_packedBytes, scale, logit_softcap, firstCell, nCells, \
        nHeadsQ, nKVHeads, nSeq, \
        nb_q01, nb_q02, nb_q03, nb_k11, nb_k12, nb_k13, \
        mask_ne0, mask_nb1)
// Full kernel — nSplits>1 or indexed (locs!=nullptr).
// SINGLE_SPLIT=true for ss paths so no partial_buf arithmetic in device code.
#define TQ_VONLY_CALL_FULL_SS(D_L, SC, IX) \
    tq_fattn_vec_vonly_kernel<D_L, SC, true, IX><<<blocks, threads, smem, stream>>>( \
        Q, K, V_packed, mask, dst, nullptr, v_scales, v_codebook, locs, \
        v_bits, v_packedBytes, scale, logit_softcap, firstCell, nCells, 1, \
        nHeadsQ, nKVHeads, nSeq, \
        nb_q01, nb_q02, nb_q03, nb_k11, nb_k12, nb_k13, \
        mask_ne0, mask_nb1)
#define TQ_VONLY_CALL(D_L, SC, SS, IX) \
    tq_fattn_vec_vonly_kernel<D_L, SC, SS, IX><<<blocks, threads, smem, stream>>>( \
        Q, K, V_packed, mask, dst, partial_buf, v_scales, v_codebook, locs, \
        v_bits, v_packedBytes, scale, logit_softcap, firstCell, nCells, nSplits, \
        nHeadsQ, nKVHeads, nSeq, \
        nb_q01, nb_q02, nb_q03, nb_k11, nb_k12, nb_k13, \
        mask_ne0, mask_nb1)
#define TQ_VONLY_LAUNCH(D_VAL) \
    constexpr int D_L = D_VAL; \
    dim3 threads(WARP_SIZE, nwarps); \
    dim3 blocks(ncols, nSplits, nHeadsQ * nSeq); \
    constexpr size_t smem = (D_L + nthreads + nwarps * D_L) * sizeof(float); \
    const bool sc = logit_softcap != 0.0f; \
    const bool ss = nSplits == 1; \
    const bool ix = locs != nullptr; \
    if      ( sc &&  ss && !ix) { TQ_VONLY_CALL_NS(D_L, true);            } \
    else if ( sc &&  ss &&  ix) { TQ_VONLY_CALL_FULL_SS(D_L, true,  true);  } \
    else if ( sc && !ss &&  ix) { TQ_VONLY_CALL(D_L, true,  false, true);  } \
    else if ( sc && !ss && !ix) { TQ_VONLY_CALL(D_L, true,  false, false); } \
    else if (!sc &&  ss && !ix) { TQ_VONLY_CALL_NS(D_L, false);           } \
    else if (!sc &&  ss &&  ix) { TQ_VONLY_CALL_FULL_SS(D_L, false, true);  } \
    else if (!sc && !ss &&  ix) { TQ_VONLY_CALL(D_L, false, false, true);  } \
    else                        { TQ_VONLY_CALL(D_L, false, false, false); } \
    CUDA_CHECK(cudaGetLastError()); \
    if (nSplits > 1) { \
        const int nTiles_r = nHeadsQ * nSeq * ncols; \
        dim3 rblocks(ncols, 1, nHeadsQ * nSeq); \
        dim3 rthreads(D_L, 1, 1); \
        tq_fattn_vonly_reduce_kernel<D_L><<<rblocks, rthreads, 0, stream>>>( \
            dst, partial_buf, nSplits, nTiles_r, nHeadsQ, ncols); \
        CUDA_CHECK(cudaGetLastError()); \
    }

extern "C" void ggml_cuda_tq_fattn_vec_vonly_d64(
    cudaStream_t stream,
    const float * Q,             int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    const __half * K,            int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    const uint8_t * V_packed,    int32_t v_packedBytes,
    const __half * mask,         int32_t mask_ne0, int32_t mask_nb1,
    float * dst,
    float * partial_buf,
    const float * v_scales,
    const float * v_codebook,
    const int32_t * locs,
    int v_bits,
    float scale,
    float logit_softcap,
    int firstCell,
    int nCells,
    int nSplits,
    int nHeadsQ,
    int nKVHeads,
    int nSeq,
    int ncols)
{
    TQ_VONLY_LAUNCH(64)
}

extern "C" void ggml_cuda_tq_fattn_vec_vonly_d128(
    cudaStream_t stream,
    const float * Q,             int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    const __half * K,            int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    const uint8_t * V_packed,    int32_t v_packedBytes,
    const __half * mask,         int32_t mask_ne0, int32_t mask_nb1,
    float * dst,
    float * partial_buf,
    const float * v_scales,
    const float * v_codebook,
    const int32_t * locs,
    int v_bits,
    float scale,
    float logit_softcap,
    int firstCell,
    int nCells,
    int nSplits,
    int nHeadsQ,
    int nKVHeads,
    int nSeq,
    int ncols)
{
    TQ_VONLY_LAUNCH(128)
}

extern "C" void ggml_cuda_tq_fattn_vec_vonly_d256(
    cudaStream_t stream,
    const float * Q,             int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    const __half * K,            int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    const uint8_t * V_packed,    int32_t v_packedBytes,
    const __half * mask,         int32_t mask_ne0, int32_t mask_nb1,
    float * dst,
    float * partial_buf,
    const float * v_scales,
    const float * v_codebook,
    const int32_t * locs,
    int v_bits,
    float scale,
    float logit_softcap,
    int firstCell,
    int nCells,
    int nSplits,
    int nHeadsQ,
    int nKVHeads,
    int nSeq,
    int ncols)
{
    TQ_VONLY_LAUNCH(256)
}

extern "C" void ggml_cuda_tq_fattn_vec_vonly_d512(
    cudaStream_t stream,
    const float * Q,             int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    const __half * K,            int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    const uint8_t * V_packed,    int32_t v_packedBytes,
    const __half * mask,         int32_t mask_ne0, int32_t mask_nb1,
    float * dst,
    float * partial_buf,
    const float * v_scales,
    const float * v_codebook,
    const int32_t * locs,
    int v_bits,
    float scale,
    float logit_softcap,
    int firstCell,
    int nCells,
    int nSplits,
    int nHeadsQ,
    int nKVHeads,
    int nSeq,
    int ncols)
{
    TQ_VONLY_LAUNCH(512)
}

#undef TQ_VONLY_LAUNCH

#else  // !FLASH_ATTN_AVAILABLE / sm < 600

#define TQ_VONLY_STUB(name) \
extern "C" void name( \
    cudaStream_t, const float *, int32_t, int32_t, int64_t, \
    const __half *, int32_t, int32_t, int64_t, \
    const uint8_t *, int32_t, \
    const __half *, int32_t, int32_t, \
    float *, float *, const float *, const float *, const int32_t *, int, float, float, \
    int, int, int, int, int, int, int) { \
    GGML_ABORT(#name ": not available on this device"); \
}

TQ_VONLY_STUB(ggml_cuda_tq_fattn_vec_vonly_d64)
TQ_VONLY_STUB(ggml_cuda_tq_fattn_vec_vonly_d128)
TQ_VONLY_STUB(ggml_cuda_tq_fattn_vec_vonly_d256)
TQ_VONLY_STUB(ggml_cuda_tq_fattn_vec_vonly_d512)

#undef TQ_VONLY_STUB

#endif
