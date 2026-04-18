#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// TurboQuant inline decode: extract a N-bit Lloyd-Max index from a packed byte row
// and return codebook[idx] * rms_scale.  Handles cross-byte boundaries for bits=2,3.
static __device__ __forceinline__ float tq_decode_elem(
    const uint8_t * packed_row, const float * codebook, float rms_scale, int elem, int bits)
{
    const int bit_pos  = elem * bits;
    const int byte_idx = bit_pos >> 3;
    const int shift    = bit_pos & 7;
    const int mask_val = (1 << bits) - 1;
    int idx = ((int)(packed_row[byte_idx] >> shift)) & mask_val;
    if (shift + bits > 8) {
        idx |= ((int)(packed_row[byte_idx + 1] << (8 - shift))) & mask_val;
    }
    return codebook[idx] * rms_scale;
}

// --------------------------------------------------------------------
// Hardcoded warp-shuffle TQ decode: combine packed bytes into a single
// integer, shift to align, then extract N indices with compile-time
// offsets.  Eliminates per-element multiply, byte_idx computation, and
// boundary-crossing branches.
//
// cb_lane: codebook[threadIdx.x % (1<<bits)], loaded once per kernel.
// shfl_w:  shuffle width (power of 2, >= 1<<bits).
// --------------------------------------------------------------------

#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
// 3-bit, 4 elements.  start_elem is a multiple of 4.
// 12 bits needed from 2 bytes; bit offset within the first byte is 0 or 4.
static __device__ __forceinline__ void tq_decode_4_3bit(
    const uint8_t * __restrict__ packed_row,
    float cb_lane, float rms, int start_elem, int shfl_w,
    float * __restrict__ out)
{
    const int byte_off = (start_elem * 3) >> 3;
    const int bit_off  = (start_elem * 3) & 7;   // 0 or 4
    const uint32_t w = (uint32_t)packed_row[byte_off] | ((uint32_t)packed_row[byte_off + 1] << 8);
    const uint32_t s = w >> bit_off;              // align element 0 to bit 0
    out[0] = __shfl_sync(0xFFFFFFFF, cb_lane, (s >> 0) & 7, shfl_w) * rms;
    out[1] = __shfl_sync(0xFFFFFFFF, cb_lane, (s >> 3) & 7, shfl_w) * rms;
    out[2] = __shfl_sync(0xFFFFFFFF, cb_lane, (s >> 6) & 7, shfl_w) * rms;
    out[3] = __shfl_sync(0xFFFFFFFF, cb_lane, (s >> 9) & 7, shfl_w) * rms;
}

// 3-bit, 8 elements.  start_elem is a multiple of 8 (Volta+ path).
// 24 bits from 3 bytes; bit offset is always 0.
static __device__ __forceinline__ void tq_decode_8_3bit(
    const uint8_t * __restrict__ packed_row,
    float cb_lane, float rms, int start_elem, int shfl_w,
    float * __restrict__ out)
{
    const int byte_off = (start_elem * 3) >> 3;
    const uint32_t w = (uint32_t)packed_row[byte_off]
                     | ((uint32_t)packed_row[byte_off + 1] << 8)
                     | ((uint32_t)packed_row[byte_off + 2] << 16);
    out[0] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  0) & 7, shfl_w) * rms;
    out[1] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  3) & 7, shfl_w) * rms;
    out[2] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  6) & 7, shfl_w) * rms;
    out[3] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  9) & 7, shfl_w) * rms;
    out[4] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 12) & 7, shfl_w) * rms;
    out[5] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 15) & 7, shfl_w) * rms;
    out[6] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 18) & 7, shfl_w) * rms;
    out[7] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 21) & 7, shfl_w) * rms;
}

// 2-bit, 4 elements.  start_elem is a multiple of 4.
// 8 bits = 1 byte, always byte-aligned.
static __device__ __forceinline__ void tq_decode_4_2bit(
    const uint8_t * __restrict__ packed_row,
    float cb_lane, float rms, int start_elem, int shfl_w,
    float * __restrict__ out)
{
    const uint32_t b = packed_row[start_elem >> 2];
    out[0] = __shfl_sync(0xFFFFFFFF, cb_lane, (b >> 0) & 3, shfl_w) * rms;
    out[1] = __shfl_sync(0xFFFFFFFF, cb_lane, (b >> 2) & 3, shfl_w) * rms;
    out[2] = __shfl_sync(0xFFFFFFFF, cb_lane, (b >> 4) & 3, shfl_w) * rms;
    out[3] = __shfl_sync(0xFFFFFFFF, cb_lane, (b >> 6) & 3, shfl_w) * rms;
}

// 2-bit, 8 elements.  start_elem is a multiple of 8.
// 16 bits = 2 bytes.
static __device__ __forceinline__ void tq_decode_8_2bit(
    const uint8_t * __restrict__ packed_row,
    float cb_lane, float rms, int start_elem, int shfl_w,
    float * __restrict__ out)
{
    const int byte_off = start_elem >> 2;
    const uint32_t w = (uint32_t)packed_row[byte_off] | ((uint32_t)packed_row[byte_off + 1] << 8);
    out[0] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  0) & 3, shfl_w) * rms;
    out[1] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  2) & 3, shfl_w) * rms;
    out[2] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  4) & 3, shfl_w) * rms;
    out[3] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  6) & 3, shfl_w) * rms;
    out[4] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  8) & 3, shfl_w) * rms;
    out[5] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 10) & 3, shfl_w) * rms;
    out[6] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 12) & 3, shfl_w) * rms;
    out[7] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 14) & 3, shfl_w) * rms;
}
#else
// Stubs for sm < 600 (no __shfl_sync).  Never executed — the launcher
// asserts compute capability >= 6.0.  These only exist so that the
// __device__ dispatch template below can compile for Maxwell targets.
static __device__ __forceinline__ void tq_decode_4_3bit(
    const uint8_t * __restrict__, float, float, int, int,
    float * __restrict__ out)
{ out[0] = out[1] = out[2] = out[3] = 0.0f; }
static __device__ __forceinline__ void tq_decode_8_3bit(
    const uint8_t * __restrict__, float, float, int, int,
    float * __restrict__ out)
{ out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = 0.0f; }
static __device__ __forceinline__ void tq_decode_4_2bit(
    const uint8_t * __restrict__, float, float, int, int,
    float * __restrict__ out)
{ out[0] = out[1] = out[2] = out[3] = 0.0f; }
static __device__ __forceinline__ void tq_decode_8_2bit(
    const uint8_t * __restrict__, float, float, int, int,
    float * __restrict__ out)
{ out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = 0.0f; }
#endif

// Dispatch: decode N elements (N=4 on Pascal, N=8 on Volta+).
template<int N>
static __device__ __forceinline__ void tq_decode_N_shfl(
    const uint8_t * __restrict__ packed_row,
    float cb_lane, float rms,
    int start_elem, int bits, int shfl_w,
    float * __restrict__ out)
{
    if constexpr (N == 4) {
        if (bits == 3) { tq_decode_4_3bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
        else           { tq_decode_4_2bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
    } else {
        if (bits == 3) { tq_decode_8_3bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
        else           { tq_decode_8_2bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
    }
}

// Compute Q·K dot product where K is TQ-compressed.
// Uses warp-shuffle for codebook lookup (1-cycle register-to-register).
template<int D, int nthreads, int cpy_ne>
static __device__ __forceinline__ float tq_vec_dot_KQ(
    const uint8_t * packed_row,
    float           cb_lane,
    float           rms_scale,
    const float2  * Q_f,
    int             bits)
{
    float sum = 0.0f;
    const int tid_kq = threadIdx.x % nthreads;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        // Decode 2*cpy_ne K elements at once.
        float k_dec[2*cpy_ne];
        tq_decode_N_shfl<2*cpy_ne>(packed_row, cb_lane, rms_scale,
                                    2*(k_KQ_0 + tid_kq*cpy_ne), bits, nthreads, k_dec);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const float2 q_pair = Q_f[k_KQ_0/nthreads + k_KQ_1];
            sum += q_pair.x * k_dec[2*k_KQ_1] + q_pair.y * k_dec[2*k_KQ_1 + 1];
        }
    }
    return sum;
}

// TurboQuant fused flash-attention kernel.
//
// Q:        [D, nTokensQ, nHeadsQ, nSeq] f32  (after SDPA Permute(0,2,1,3))
// K_packed: [packedBytes*nKVHeads, capacity] i8 (encode result; base = full buffer)
// V:        [D, nCells, nKVHeads, nSeq] f16  when !V_PACKED (after FA-branch Permute(0,2,1,3))
//            [v_packedBytes*nKVHeads, capacity] i8 when V_PACKED (packed buffer, no permute)
// mask:     [nCells, nTokensQ] f16 or NULL
// scales:   [nKVHeads, capacity] f32         (K scales; base = full buffer)
// codebook: [1<<bits] f32                    (K codebook)
//
// V_PACKED == false: existing K-only fused path (V is f16, src[6] == NULL)
// V_PACKED == true:  new K+V fused path (V is packed i8, src[6] = v_scales, src[7] = v_codebook)
//
// Phase 1: D == 128, bits runtime param, gridDim.y == 1 (no multi-block).
template<int D, int ncols, bool use_logit_softcap, bool V_PACKED>
__launch_bounds__(128, 2)
static __global__ void tq_flash_attn_ext_vec(
    const char    * __restrict__ Q,
    const uint8_t * __restrict__ K_packed,
    const char    * __restrict__ V,
    const char    * __restrict__ mask,
    float         * __restrict__ dst,
    const float   * __restrict__ scales,
    const float   * __restrict__ codebook,
    float   scale,
    float   logit_softcap,
    int     bits,
    int     firstCell,
    int     nCells,
    int     nKVHeads,
    int     packedBytes,
    // Q geometry
    int32_t ne00,
    uint3   ne01,       // init_fastdiv_values(nTokensQ); .z == nTokensQ
    int32_t ne02,       // nHeadsQ
    int32_t ne03,       // nSeq
    int32_t nb01,       // Q stride: bytes between consecutive tokens
    int32_t nb02,       // Q stride: bytes between consecutive heads
    int64_t nb03,       // Q stride: bytes between consecutive sequences
    // V geometry (after permute: [D, nCells, nKVHeads]) — only used when !V_PACKED
    int32_t nb21,       // bytes between V cells  (= D*nKVHeads*sizeof(half))
    int32_t nb22,       // bytes between V heads  (= D*sizeof(half))
    int64_t nb23,       // bytes between V seqs
    // mask geometry
    int32_t ne31,       // nCells (mask row width)
    int32_t nb31,       // mask stride: bytes between token rows
    // V packed params (only used when V_PACKED == true)
    const float   * __restrict__ v_scales,    // [nKVHeads, capacity] f32
    const float   * __restrict__ v_codebook,  // [1<<v_bits] f32
    int     v_bits,
    int     v_packedBytes
)
{
#ifdef FLASH_ATTN_AVAILABLE
    // Skip logit_softcap variants for unsupported D values (mirrors original kernel guard).
    if (use_logit_softcap && D != 128 && D != 256) {
        GGML_UNUSED_VARS(Q, K_packed, V, mask, dst, scales, codebook,
            scale, logit_softcap, bits, firstCell, nCells, nKVHeads, packedBytes,
            ne00, ne01, ne02, ne03, nb01, nb02, nb03,
            nb21, nb22, nb23, ne31, nb31,
            v_scales, v_codebook, v_bits, v_packedBytes);
        NO_DEVICE_CODE;
        return;
    }

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();  // 16 on Volta+
    constexpr int cpy_ne = cpy_nb / 4;                     // 4

    constexpr int nthreads    = 128;
    constexpr int nthreads_KQ = nthreads / cpy_nb;         // 8
    constexpr int nthreads_V  = nthreads / cpy_nb;         // 8

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_KQ");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");
    static_assert(D % (2*WARP_SIZE) == 0,        "D not divisible by 64");

    constexpr int V_rows_per_thread = 2 * cpy_ne;                  // 8
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;      // 4

    // dequantize_V is only needed for the f16 V path
    [[maybe_unused]] constexpr dequantize_V_t dequantize_V =
        V_PACKED ? (dequantize_V_t)nullptr
                 : get_dequantize_V<GGML_TYPE_F16, float, V_rows_per_thread>();

    const int ic0     = blockIdx.x * ncols;
    const int sequence = blockIdx.z / ne02;
    const int head     = blockIdx.z % ne02;
    const int gqa_ratio = ne02 / nKVHeads;
    const int head_kv   = head / gqa_ratio;

    // Advance base pointers.
    Q        += (int64_t)nb03*sequence + nb02*head + nb01*ic0;
    K_packed += (int64_t)firstCell * nKVHeads * packedBytes + (int64_t)head_kv * packedBytes;
    scales   += (int64_t)firstCell * nKVHeads + head_kv;

    // V pointer setup: f16 path uses stride-based addressing; packed path uses cell-index addressing.
    const uint8_t * V_packed_base = nullptr;
    const float   * v_scales_base = nullptr;
    if constexpr (V_PACKED) {
        V_packed_base = (const uint8_t *)V
            + (int64_t)firstCell * nKVHeads * v_packedBytes
            + (int64_t)head_kv * v_packedBytes;
        v_scales_base = v_scales
            + (int64_t)firstCell * nKVHeads + head_kv;
    } else {
        V += (int64_t)nb23*sequence + (int64_t)nb22*head_kv;
    }

    const half * maskh = mask ? (const half *)(mask + (int64_t)nb31*ic0) : nullptr;

    // Load one codebook entry per lane for warp-shuffle lookups.
    // For 3-bit (8 entries): lanes 0-7 hold codebook[0-7], lanes 8-15 repeat, etc.
    // __shfl_sync with width=8 handles the wrap-around.
    const float k_cb_lane = codebook[threadIdx.x & ((1 << bits) - 1)];
    [[maybe_unused]] const float v_cb_lane = V_PACKED
        ? v_codebook[threadIdx.x & ((1 << v_bits) - 1)]
        : 0.0f;

    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols * D;
    constexpr int ne_combine = nwarps * V_cols_per_iter * D;

    float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    __shared__ float KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Load Q into registers (float2 per half-pair, scale applied).
    float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}};
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float2 * Q_j = (const float2 *)(Q + j*nb01);
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
            const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
            if (ncols == 1 || ic0 + j < (int)ne01.z) {
                ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
            }
        }
        // Apply attention scale.
#pragma unroll
        for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
            Q_reg[j][k].x *= scale;
            Q_reg[j][k].y *= scale;
        }
    }

    // Main KV loop — single block (gridDim.y == 1).
    for (int k_VKQ_0 = 0; k_VKQ_0 < nCells; k_VKQ_0 += nthreads,
             V += (V_PACKED ? 0 : (int64_t)nthreads * nb21),
             maskh += (maskh ? nthreads : 0)) {

        float KQ_reg[ncols];
        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE
                           + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1)))
                           + i_KQ_0;
            const int cell_rel = k_VKQ_0 + i_KQ;
            const bool in_range = (cell_rel < nCells);

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                // Always execute the dot product to keep all lanes convergent
                // for warp shuffles.  rms_scale=0 for out-of-range cells.
                const uint8_t * packed_row = K_packed + (int64_t)cell_rel * nKVHeads * packedBytes;
                const float     rms_scale  = in_range ? scales[cell_rel * nKVHeads] : 0.0f;
                float sum = tq_vec_dot_KQ<D, nthreads_KQ, cpy_ne>(
                    packed_row, k_cb_lane, rms_scale, Q_reg[j], bits);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap * tanhf(sum);
                }

                if (maskh && (ncols == 1 || ic0 + j < (int)ne01.z)) {
                    sum += __half2float(maskh[j*ne31 + i_KQ]);
                }

                if (!in_range) {
                    sum = -FLT_MAX/2.0f;
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == (uint32_t)i_KQ_0) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0
                        + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }

            if constexpr (V_PACKED) {
                // Decode V from packed buffer inline — warp-shuffle batch decode.
                // Always execute decodes to keep all lanes convergent for shuffles.
                // v_rms=0 for out-of-range cells produces zero contributions.
                const int cell_rel = k_VKQ_0 + k;
                const uint8_t * v_row = V_packed_base
                    + (int64_t)cell_rel * nKVHeads * v_packedBytes;
                const float v_rms = (cell_rel < nCells)
                    ? v_scales_base[cell_rel * nKVHeads] : 0.0f;

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    const int base_elem = 2*i_VKQ_0
                        + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)
                          * V_rows_per_thread;

                    // Batch-decode V elements using shuffle codebook.
                    float v_dec[V_rows_per_thread];
                    tq_decode_N_shfl<V_rows_per_thread>(v_row, v_cb_lane, v_rms,
                                                         base_elem, v_bits, nthreads_V, v_dec);

#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += v_dec[2*i_VKQ_1]   * KQ_k[j];
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += v_dec[2*i_VKQ_1+1] * KQ_k[j];
                        }
                    }
                }
            } else {
                // Original f16 V path (unchanged).
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    float2 tmp[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x * KQ_k[j];
                            VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y * KQ_k[j];
                        }
                    }
                }
            }
        }
    } // end KV loop

    // --- Reduce across warps and write output ---

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= (int)ne01.z) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

        float2 * VKQ_tmp = (float2 *)KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0
                + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,
                                &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4,
                                &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }
        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                dst_val /= KQ_sum[j_VKQ];
                // Output layout: [D, nHeadsQ, nTokensQ, nSeq] — matches ggml_flash_attn_ext layout.
                dst[(((int64_t)sequence*(int)ne01.z + ic0 + j_VKQ)*ne02 + head)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q, K_packed, V, mask, dst, scales, codebook,
        scale, logit_softcap, bits, firstCell, nCells, nKVHeads, packedBytes,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        nb21, nb22, nb23, ne31, nb31,
        v_scales, v_codebook, v_bits, v_packedBytes);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
