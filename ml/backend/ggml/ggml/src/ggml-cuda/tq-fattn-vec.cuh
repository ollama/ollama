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

// 4-bit, 4 elements.  start_elem is a multiple of 4.
// 16 bits = 2 bytes, always byte-aligned (start_elem*4 bits always on a byte boundary).
// Uses shfl_w=16 (two KQ-groups): 16-entry codebook fits exactly in a 16-lane subgroup
// because codebook[thread%16] cycles 0..15 within each 16-lane aligned half-warp.
static __device__ __forceinline__ void tq_decode_4_4bit(
    const uint8_t * __restrict__ packed_row,
    float cb_lane, float rms, int start_elem, int shfl_w,
    float * __restrict__ out)
{
    const int byte_off = start_elem >> 1;  // start_elem * 4 / 8
    const uint32_t w = (uint32_t)packed_row[byte_off] | ((uint32_t)packed_row[byte_off + 1] << 8);
    out[0] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  0) & 0xF, 16) * rms;
    out[1] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  4) & 0xF, 16) * rms;
    out[2] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  8) & 0xF, 16) * rms;
    out[3] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 12) & 0xF, 16) * rms;
}

// 4-bit, 8 elements.  start_elem is a multiple of 8 (Volta+ path).
// 32 bits = 4 bytes.
static __device__ __forceinline__ void tq_decode_8_4bit(
    const uint8_t * __restrict__ packed_row,
    float cb_lane, float rms, int start_elem, int shfl_w,
    float * __restrict__ out)
{
    const int byte_off = start_elem >> 1;
    const uint32_t w = (uint32_t)packed_row[byte_off]
                     | ((uint32_t)packed_row[byte_off + 1] << 8)
                     | ((uint32_t)packed_row[byte_off + 2] << 16)
                     | ((uint32_t)packed_row[byte_off + 3] << 24);
    out[0] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  0) & 0xF, 16) * rms;
    out[1] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  4) & 0xF, 16) * rms;
    out[2] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >>  8) & 0xF, 16) * rms;
    out[3] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 12) & 0xF, 16) * rms;
    out[4] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 16) & 0xF, 16) * rms;
    out[5] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 20) & 0xF, 16) * rms;
    out[6] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 24) & 0xF, 16) * rms;
    out[7] = __shfl_sync(0xFFFFFFFF, cb_lane, (w >> 28) & 0xF, 16) * rms;
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
static __device__ __forceinline__ void tq_decode_4_4bit(
    const uint8_t * __restrict__, float, float, int, int,
    float * __restrict__ out)
{ out[0] = out[1] = out[2] = out[3] = 0.0f; }
static __device__ __forceinline__ void tq_decode_8_4bit(
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
        if      (bits == 4) { tq_decode_4_4bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
        else if (bits == 3) { tq_decode_4_3bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
        else                { tq_decode_4_2bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
    } else {
        if      (bits == 4) { tq_decode_8_4bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
        else if (bits == 3) { tq_decode_8_3bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
        else                { tq_decode_8_2bit(packed_row, cb_lane, rms, start_elem, shfl_w, out); }
    }
}

// Fetch Q[d] (a single float at head-dim position d ∈ [0, D)) from the
// per-lane Q_reg storage, routing across the 8-lane KQ group via a warp
// shuffle. Q_reg layout (see kernel body): for each i0 ∈ {0,32,64,96}
// in pair units, the 8 lanes of a KQ group each hold 4 consecutive pairs
// Q[i0 + tid_kq*4 .. i0 + tid_kq*4 + 3] at slots [i0/8 .. i0/8+3].
//
// Given d, derive the owning lane and slot and shuffle out the scalar.
// cpy_ne=4, nthreads_KQ=8 are fixed at template instantiation time.
template<int D, int nthreads_KQ, int cpy_ne>
static __device__ __forceinline__ float shuffle_Q_at(
    const float2 * __restrict__ Q_reg_j, int d)
{
    const int p     = d >> 1;
    const int chunk = p / (nthreads_KQ * cpy_ne);               // p / 32
    const int lane  = (p / cpy_ne) % nthreads_KQ;               // (p/4) & 7
    const int slot  = chunk * cpy_ne + (p & (cpy_ne - 1));      // chunk*4 + p%4
    const unsigned tgt_lane = (threadIdx.x & ~(nthreads_KQ - 1)) | lane;
    const float qx = __shfl_sync(0xFFFFFFFF, Q_reg_j[slot].x, tgt_lane, WARP_SIZE);
    const float qy = __shfl_sync(0xFFFFFFFF, Q_reg_j[slot].y, tgt_lane, WARP_SIZE);
    return (d & 1) ? qy : qx;
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
// Phase 1: D == 128, bits runtime param.
// Multi-block KV-split (parallel_blocks > 1): when gridDim.y > 1, blocks share
// each (Q-tile, head, sequence) and use interleaved K-cell striping. Each block
// writes its partial VKQ + (KQ_max, KQ_sum) meta into dst / dst_meta workspace
// slots indexed by blockIdx.y; flash_attn_combine_results combines them.
template<int D, int ncols, bool use_logit_softcap, bool V_PACKED, bool HAS_OUTLIERS>
__launch_bounds__(128, 2)
static __global__ void tq_flash_attn_ext_vec(
    const char    * __restrict__ Q,
    const uint8_t * __restrict__ K_packed,
    const char    * __restrict__ V,
    const char    * __restrict__ mask,
    float         * __restrict__ dst,
    float2        * __restrict__ dst_meta,
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
    int     v_packedBytes,
    // Asymmetric primary quantization params
    const float   * __restrict__ zeros,       // [nKVHeads, capacity] f32 (NULL if symmetric)
    int     asymmetric,
    // QJL residual sketch params
    const uint8_t * __restrict__ qjl_packed,  // QJL sign bits (NULL if no QJL)
    const float   * __restrict__ qjl_norm,    // [nKVHeads, capacity] f32 (NULL if no QJL)
    const float   * __restrict__ qjl_projection, // [qjlRows, headDim] f32 (NULL if no QJL)
    int     qjl_rows,
    int     qjl_packedBytes,
    // Outlier-split dual-stream decode params (only used when HAS_OUTLIERS).
    // When HAS_OUTLIERS, the codebook array is concatenated as
    //   [regular (1<<bits), outlier (1<<outlier_bits)]
    // and outlier_count > 0.
    const uint8_t * __restrict__ outlier_packed,  // [outlierPackedBytes*nKVHeads, capacity] i8
    const float   * __restrict__ outlier_scales,  // [nKVHeads, capacity] f32
    const int8_t  * __restrict__ outlier_indices, // [outlierCount*nKVHeads, capacity] i8
    const float   * __restrict__ outlier_zeros,   // [nKVHeads, capacity] f32 (NULL if !asymmetric)
    int     outlier_bits,
    int     outlier_count,
    int     outlier_packedBytes
)
{
#ifdef FLASH_ATTN_AVAILABLE
    // Skip logit_softcap variants for unsupported D values (mirrors original kernel guard).
    if (use_logit_softcap && D != 64 && D != 128 && D != 256) {
        GGML_UNUSED_VARS(Q, K_packed, V, mask, dst, dst_meta, scales, codebook,
            scale, logit_softcap, bits, firstCell, nCells, nKVHeads, packedBytes,
            ne00, ne01, ne02, ne03, nb01, nb02, nb03,
            nb21, nb22, nb23, ne31, nb31,
            v_scales, v_codebook, v_bits, v_packedBytes,
            zeros, asymmetric, qjl_packed, qjl_norm, qjl_projection, qjl_rows, qjl_packedBytes,
            outlier_packed, outlier_scales, outlier_indices, outlier_zeros,
            outlier_bits, outlier_count, outlier_packedBytes);
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

    Q        += (int64_t)nb03*sequence + nb02*head + nb01*ic0;
    K_packed += (int64_t)firstCell * nKVHeads * packedBytes + (int64_t)head_kv * packedBytes;
    scales   += (int64_t)firstCell * nKVHeads + head_kv;

    if (zeros) {
        zeros += (int64_t)firstCell * nKVHeads + head_kv;
    }
    if (qjl_rows > 0) {
        qjl_packed += (int64_t)firstCell * nKVHeads * qjl_packedBytes + (int64_t)head_kv * qjl_packedBytes;
        qjl_norm   += (int64_t)firstCell * nKVHeads + head_kv;
    }

    // Outlier-split base-pointer adjustments. All outlier tensors are indexed
    // with firstCell as the leading cell offset (matches encode-side writes).
    if constexpr (HAS_OUTLIERS) {
        outlier_packed  += (int64_t)firstCell * nKVHeads * outlier_packedBytes + (int64_t)head_kv * outlier_packedBytes;
        outlier_scales  += (int64_t)firstCell * nKVHeads + head_kv;
        outlier_indices += (int64_t)firstCell * nKVHeads * outlier_count + (int64_t)head_kv * outlier_count;
        if (outlier_zeros) {
            outlier_zeros += (int64_t)firstCell * nKVHeads + head_kv;
        }
    }

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

    // Multi-block KV-split: when gridDim.y > 1, each block handles an interleaved
    // subset of K cells starting at blockIdx.y * nthreads, striding by gridDim.y *
    // nthreads. Bump V (f16 path) and maskh by this block's starting offset so the
    // existing per-iteration increments (k_VKQ_0 += gridDim.y*nthreads) track them.
    // Mirrors stock CUDA FA's pattern at fattn-vec.cuh:240-245.
    if constexpr (!V_PACKED) {
        V += (int64_t)blockIdx.y * nthreads * nb21;
    }

    const half * maskh = mask ? (const half *)(mask + (int64_t)nb31*ic0) : nullptr;
    if (maskh) {
        maskh += blockIdx.y * nthreads;
    }

    // Load one codebook entry per lane for warp-shuffle lookups.
    // For 3-bit (8 entries): lanes 0-7 hold codebook[0-7], lanes 8-15 repeat, etc.
    // __shfl_sync with width=8 handles the wrap-around.
    const float k_cb_lane = codebook[threadIdx.x & ((1 << bits) - 1)];
    [[maybe_unused]] const float v_cb_lane = V_PACKED
        ? v_codebook[threadIdx.x & ((1 << v_bits) - 1)]
        : 0.0f;
    // Outlier codebook is concatenated onto `codebook` after the first (1<<bits)
    // regular entries (see NewTQCompressedKManager). Lane `t` reads outlier
    // codebook entry at index `t & ((1<<outlier_bits)-1)`.
    [[maybe_unused]] const float o_cb_lane = HAS_OUTLIERS
        ? codebook[(1 << bits) + (threadIdx.x & ((1 << outlier_bits) - 1))]
        : 0.0f;

    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols * D;
    constexpr int ne_combine = nwarps * V_cols_per_iter * D;

    float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    extern __shared__ float s_mem_all[];
    float * KQ = s_mem_all;
    float * s_Q_fixed = s_mem_all + (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    float * s_dot_q_fixed = s_Q_fixed + ncols * D;

    const int tid_global = threadIdx.y * WARP_SIZE + threadIdx.x;

    // Load Q into shared memory for stable divergent access in the outlier path.
    // Pre-scale on load so consumers can read s_Q_fixed directly.
    for (int i = tid_global; i < ncols * D; i += nthreads) {
        const int head_q = i / D;
        const int elem   = i % D;
        if (head_q < ncols) {
            const float * Q_ptr = (const float *)(Q + head_q * nb01);
            s_Q_fixed[i] = Q_ptr[elem] * scale;
        }
    }
    __syncthreads();

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Precompute constants for KQ groups and thread identification.
    constexpr int num_kq_groups = nthreads / nthreads_KQ; // 128 / 8 = 16
    const int kq_group_id = threadIdx.y * (WARP_SIZE / nthreads_KQ) + (threadIdx.x / nthreads_KQ);
    const int tid_kq = threadIdx.x % nthreads_KQ;

    // Load Q into registers from shared memory (stable divergent access).
    // s_Q_fixed is already pre-scaled, no additional multiply needed.
    float2 Q_reg[ncols][(D/2)/nthreads_KQ];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
            const int i = k * nthreads_KQ + tid_kq;
            // Q_reg[j][k] stores elements (2*i, 2*i+1)
            // s_Q_fixed is pre-scaled at load time.
            Q_reg[j][k].x = s_Q_fixed[j * D + 2*i];
            Q_reg[j][k].y = s_Q_fixed[j * D + 2*i + 1];
        }
    }

    // Precompute per-query sums of Q elements (for asymmetric zero contribution).
    float sum_q[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        float local_sum = 0.0f;
#pragma unroll
        for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
            local_sum += Q_reg[j][k].x + Q_reg[j][k].y;
        }
        sum_q[j] = warp_reduce_sum<nthreads_KQ>(local_sum);
    }

    // Precompute dot_q[j] = Σ_e Q[e] * G_j[e] for QJL residual sketch.
    // Partition j across KQ groups to avoid redundant work.

    if (qjl_rows > 0) {
        for (int i = tid_global; i < ncols * qjl_rows; i += nthreads) {
            s_dot_q_fixed[i] = 0.0f;
        }
        __syncthreads();

        for (int qj = 0; qj < ncols; ++qj) {
            for (int i = kq_group_id; i < qjl_rows; i += num_kq_groups) {
                float partial = 0.0f;
#pragma unroll
                for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                    const int base_elem = 2*(k*nthreads_KQ + tid_kq);
                    const float qx = Q_reg[qj][k].x;
                    const float qy = Q_reg[qj][k].y;
                    const float px = qjl_projection[i * D + base_elem];
                    const float py = qjl_projection[i * D + base_elem + 1];
                    const float prod_x = qx * px;
                    const float prod_y = qy * py;
                    partial += prod_x;
                    partial += prod_y;
                }
                partial = warp_reduce_sum<nthreads_KQ>(partial);
                if (tid_kq == 0) {
                    s_dot_q_fixed[qj * 256 + i] = partial;
                }
            }
        }
        __syncthreads();
    }

    // Main KV loop. When gridDim.y > 1, this block handles every gridDim.y'th
    // chunk of K cells starting at blockIdx.y. Initial K_packed/V/maskh offsets
    // for blockIdx.y are applied above; the per-iteration increment uses gridDim.y
    // so successive iterations skip over slices owned by sibling blocks.
    for (int k_VKQ_0 = blockIdx.y * nthreads; k_VKQ_0 < nCells; k_VKQ_0 += gridDim.y * nthreads,
             V += (V_PACKED ? 0 : (int64_t)gridDim.y * nthreads * nb21),
             maskh += (maskh ? gridDim.y * nthreads : 0)) {

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

            // Outlier-split: build per-lane register bitmap from outlier_indices.
            // sum_q_outl[j] accumulates Σ_d∈outlier Q[d] across the
            // outlier slots, for asymmetric zero re-addition later.
            [[maybe_unused]] float sum_q_outl[ncols];
            [[maybe_unused]] const int8_t * o_idx_cell_saved = nullptr;
            [[maybe_unused]] uint32_t bmap[(D + 31) / 32] = {0};
            if constexpr (HAS_OUTLIERS) {
#pragma unroll
                for (int j = 0; j < ncols; ++j) { sum_q_outl[j] = 0.0f; }

                const int8_t * o_idx_cell = in_range
                    ? outlier_indices + (int64_t)cell_rel * nKVHeads * outlier_count
                    : nullptr;
                o_idx_cell_saved = o_idx_cell;
                if (in_range) {
#pragma unroll
                    for (int s = 0; s < 32; ++s) {              // outlier_count ≤ 32 for D=128 presets
                        if (s >= outlier_count) break;
                        const int pos = (int)o_idx_cell[s];
                        if (pos >= 0 && pos < D) {
                            bmap[pos >> 5] |= (1u << (pos & 31));
                        }
                    }
                }

                // Precompute Σ_s Q[outlier_pos[s]] per query for asymmetric fixup.
                // Must be unconditional across the warp because shuffle_Q_at
                // uses a full-mask __shfl_sync — divergent lane groups in the
                // same warp (different cell_rel, hence different in_range)
                // would break warp convergence. For out-of-range cells we
                // load pos=0 and multiply by zero-scale at the use site; the
                // contribution is mathematically harmless because such cells
                // are masked to -FLT_MAX downstream.
#pragma unroll
                for (int s = 0; s < 32; ++s) {
                    if (s >= outlier_count) break;
                    const int pos = (in_range && o_idx_cell) ? (int)o_idx_cell[s] : 0;
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        sum_q_outl[j] += s_Q_fixed[j * D + pos];
                    }
                }
#ifndef GGML_USE_HIP
                __syncwarp();
#endif
            }

            // ----------------------------------------------------------------
            // Q-tile decode amortization (the prefill perf fix).
            //
            // The original kernel decoded K (and outliers) inside the `j`
            // (ncols) loop, so each cell got decoded ncols × per Q-tile.
            // Hoist the decode out of `j`: build a per-lane register cache
            // of K values, then the `j` loop is a pure FMA against
            // s_Q_fixed[j*D + d]. Same hoist for the outlier sub-stream.
            //
            // Storage:
            //   k_lane[D/nthreads_KQ]      — regular-stream decoded K at strided d's
            //   o_val_lane[outlier_count/nthreads_KQ_max] — outlier-stream decoded K
            //   o_pos_lane[outlier_count/nthreads_KQ_max] — outlier head-dim positions
            // ----------------------------------------------------------------

            const uint8_t * packed_row = K_packed + (int64_t)cell_rel * nKVHeads * packedBytes;
            const float     rms_scale  = in_range ? scales[cell_rel * nKVHeads] : 0.0f;

            // Regular-stream decode: hoisted out of `j`.
            constexpr int K_PER_LANE = D / nthreads_KQ;          // 16 for D=128, 8 for D=64, 32 for D=256
            float k_lane[K_PER_LANE];
            // Per-lane outlier-channel mask & rank-below-d, only meaningful
            // when HAS_OUTLIERS. For the regular stream, k_lane[ki] holds the
            // dequantized regular K at head-dim position d=tid_kq+ki*nthreads_KQ;
            // when the position is itself an outlier, the entry is forced to 0
            // so the unconditional FMA in the j-loop produces the correct
            // (zero) contribution from regular-stream K.
            if constexpr (HAS_OUTLIERS) {
                #pragma unroll
                for (int ki = 0; ki < K_PER_LANE; ++ki) {
                    const int d = tid_kq + ki * nthreads_KQ;
                    const bool is_outl = (bmap[d >> 5] >> (d & 31)) & 1u;
                    int outl_below = 0;
                    #pragma unroll
                    for (int w = 0; w < (D + 31) / 32; ++w) {
                        const int wbit = w * 32;
                        if (wbit + 31 < d) {
                            outl_below += __popc(bmap[w]);
                        } else if (wbit < d) {
                            const uint32_t mask = (d - wbit) < 32
                                ? ((1u << (d - wbit)) - 1u) : ~0u;
                            outl_below += __popc(bmap[w] & mask);
                        }
                    }
                    const int r_raw = d - outl_below;
                    const int r = is_outl ? 0 : r_raw;
                    const float k_val = tq_decode_elem(
                        packed_row, codebook, rms_scale, r, bits);
                    k_lane[ki] = is_outl ? 0.0f : k_val;
                }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_PER_LANE; ++ki) {
                    const int d = tid_kq + ki * nthreads_KQ;
                    k_lane[ki] = tq_decode_elem(packed_row, codebook, rms_scale, d, bits);
                }
            }

            // Outlier-stream decode: hoisted out of `j`. Each lane handles
            // 32/nthreads_KQ outlier slots (outlier_count ≤ 32 by construction).
            constexpr int O_PER_LANE = 32 / nthreads_KQ;         // 4 (assumes nthreads_KQ=8)
            [[maybe_unused]] float o_val_lane[O_PER_LANE];
            [[maybe_unused]] int   o_pos_lane[O_PER_LANE];
            if constexpr (HAS_OUTLIERS) {
                const uint8_t * o_packed_row = outlier_packed
                    + (int64_t)cell_rel * nKVHeads * outlier_packedBytes;
                const float o_rms = in_range
                    ? outlier_scales[cell_rel * nKVHeads] : 0.0f;
                const float * outlier_codebook_ptr = codebook + (1 << bits);
                #pragma unroll
                for (int s_it = 0; s_it < O_PER_LANE; ++s_it) {
                    const int s = s_it * nthreads_KQ + tid_kq;
                    const bool s_valid = (s < outlier_count);
                    o_val_lane[s_it] = s_valid
                        ? tq_decode_elem(o_packed_row, outlier_codebook_ptr,
                                         o_rms, s, outlier_bits)
                        : 0.0f;
                    o_pos_lane[s_it] = (s_valid && in_range && o_idx_cell_saved)
                        ? (int)o_idx_cell_saved[s] : 0;
                }
            }

            // Per-cell scalars hoisted out of `j` too.
            const float reg_zero_val = (HAS_OUTLIERS
                ? (zeros && in_range ? zeros[cell_rel * nKVHeads] : 0.0f)
                : (asymmetric && zeros && in_range ? zeros[cell_rel * nKVHeads] : 0.0f));
            [[maybe_unused]] const float out_zero_val =
                (HAS_OUTLIERS && outlier_zeros && in_range)
                    ? outlier_zeros[cell_rel * nKVHeads] : 0.0f;
            const float qjl_norm_val = (qjl_rows > 0 && qjl_norm && in_range)
                ? qjl_norm[cell_rel * nKVHeads] : 0.0f;
            const uint8_t * cell_qjl = (qjl_rows > 0 && qjl_packed)
                ? qjl_packed + (int64_t)cell_rel * nKVHeads * qjl_packedBytes
                : nullptr;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;

                if constexpr (HAS_OUTLIERS) {
                    // Regular stream: pure FMA with cached K decode.
                    float reg_sum = 0.0f;
                    #pragma unroll
                    for (int ki = 0; ki < K_PER_LANE; ++ki) {
                        const int d = tid_kq + ki * nthreads_KQ;
                        reg_sum += s_Q_fixed[j * D + d] * k_lane[ki];
                    }
                    // Outlier stream: pure FMA with cached outlier decode.
                    float out_sum = 0.0f;
                    #pragma unroll
                    for (int s_it = 0; s_it < O_PER_LANE; ++s_it) {
                        out_sum += s_Q_fixed[j * D + o_pos_lane[s_it]] * o_val_lane[s_it];
                    }
                    sum = reg_sum + out_sum;
                } else {
                    // Non-outlier path: pure FMA with cached K decode.
                    float reg_sum = 0.0f;
                    #pragma unroll
                    for (int ki = 0; ki < K_PER_LANE; ++ki) {
                        const int d = tid_kq + ki * nthreads_KQ;
                        reg_sum += s_Q_fixed[j * D + d] * k_lane[ki];
                    }
                    sum = reg_sum;
                }
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                // Asymmetric primary quantization: add zero * sum(Q) to the dot product.
                if constexpr (HAS_OUTLIERS) {
                    // Dual-stream zeros: regular zero applies to Σ_{d∈reg} Q[d]
                    // = sum_q[j] − sum_q_outl[j]; outlier zero applies to sum_q_outl[j].
                    if (asymmetric && (zeros || outlier_zeros)) {
#ifdef GGML_USE_HIP
                        // HIP compiler reorders sum_q_outl[j] reads past subsequent
                        // warp-reduce shuffles, reading a stale accumulator on RDNA3.
                        // A compiler fence prevents the reordering at zero HW cost.
                        asm volatile("" ::: "memory");
#endif
                        sum += reg_zero_val * (sum_q[j] - sum_q_outl[j])
                             + out_zero_val * sum_q_outl[j];
                    }
                } else if (asymmetric && zeros) {
                    sum += reg_zero_val * sum_q[j];
                }

                // QJL residual sketch: add norm * scale * Σ_i sign_i * dot_q[i] to the dot product.
                if (qjl_rows > 0 && qjl_packed && qjl_norm) {
                    const float qjl_scale = 1.2533141373155001f / qjl_rows;
                    float sign_dot = 0.0f;
                    for (int i = tid_kq; i < qjl_rows; i += nthreads_KQ) {
                        int sign = ((cell_qjl[i >> 3] >> (i & 7)) & 1) ? 1.0f : -1.0f;
                        sign_dot += (float)sign * s_dot_q_fixed[j * 256 + i];
                    }
                    sign_dot = warp_reduce_sum<nthreads_KQ>(sign_dot);
                    sum += qjl_norm_val * qjl_scale * sign_dot;
                }

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
                // f16 V path.
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
                // gridDim.y == 1: produce final result (divide by KQ_sum, write to
                // [D, head, col, seq] dst). gridDim.y > 1: write un-normalised
                // partial VKQ to workspace at slot blockIdx.y; the combine kernel
                // (flash_attn_combine_results) does the cross-block normalisation.
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                // Output layout: [D, gridDim.y, nHeadsQ, nTokensQ, nSeq] when gridDim.y
                // > 1 (workspace); [D, nHeadsQ, nTokensQ, nSeq] when gridDim.y == 1
                // (final). Matches stock CUDA FA layout (fattn-vec.cuh:482) and
                // flash_attn_combine_results' VKQ_parts addressing.
                dst[((((int64_t)sequence*(int)ne01.z + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }

            // Meta write for the combine kernel. Only when gridDim.y > 1; one thread
            // per (j_VKQ, blockIdx.y) is sufficient since KQ_max[j_VKQ] / KQ_sum[j_VKQ]
            // are the cross-warp reduced values for this block's K slice.
            if (gridDim.y != 1 && tid == 0) {
                const int64_t meta_offset =
                    (((int64_t)sequence*(int)ne01.z + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y;
                dst_meta[meta_offset] = make_float2(KQ_max[j_VKQ], KQ_sum[j_VKQ]);
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q, K_packed, V, mask, dst, dst_meta, scales, codebook,
        scale, logit_softcap, bits, firstCell, nCells, nKVHeads, packedBytes,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        nb21, nb22, nb23, ne31, nb31,
        v_scales, v_codebook, v_bits, v_packedBytes,
        zeros, asymmetric, qjl_packed, qjl_norm, qjl_projection, qjl_rows, qjl_packedBytes,
        outlier_packed, outlier_scales, outlier_indices, outlier_zeros,
        outlier_bits, outlier_count, outlier_packedBytes);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
