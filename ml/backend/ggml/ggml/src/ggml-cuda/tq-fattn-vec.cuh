#pragma once

#include "common.cuh"
#include "cp-async.cuh"
#include "fattn-common.cuh"

// TurboQuant inline decode: extract a N-bit Lloyd-Max index from a packed byte row
// and return codebook[idx] * rms_scale.  Handles cross-byte boundaries for bits=2,3.
//
// Branchless 16-bit-window read instead of the prior `if (shift+bits > 8)`
// conditional: avoids the class of nvcc miscompile previously documented for
// sm_120 (see feedback_builtin_ctz_unreliable_in_cuda_kernels.md). Reading
// packed_row[byte_idx+1] unconditionally is safe because regularPackedBytes /
// outlierPackedBytes are rounded up to 4-byte alignment in
// turboquant_compressed.go; the per-head region has at least one byte of
// padding past the last bit, and any value in the high byte is masked away
// by `& mask_val` in the non-straddle case.
static __device__ __forceinline__ float tq_decode_elem(
    const uint8_t * packed_row, const float * codebook, float rms_scale, int elem, int bits)
{
    const int bit_pos  = elem * bits;
    const int byte_idx = bit_pos >> 3;
    const int shift    = bit_pos & 7;
    const int mask_val = (1 << bits) - 1;
    // Guard second-byte read: only needed when bits straddle a byte boundary.
    // An unconditional read past the last data byte causes smem OOB for the
    // final ring slot when byte_idx+1 == packedBytes and pb_aligned is tight.
    uint32_t window = (uint32_t)packed_row[byte_idx];
    if (shift + bits > 8) {
        window |= ((uint32_t)packed_row[byte_idx + 1] << 8);
    }
    const int idx = (int)((window >> shift) & (uint32_t)mask_val);
    return codebook[idx] * rms_scale;
}

// Template version: if constexpr elides the second-byte load for byte-aligned
// widths (bits=2, bits=4).  Only use at sites where bits is compile-time known.
// For straddling widths (bits=3, outlier_bits=5) the load is preserved.
// 8 % bits != 0 is the straddle predicate: {3,5,6,7} straddle; {2,4} do not.
template<int bits>
static __device__ __forceinline__ float tq_decode_elem(
    const uint8_t * packed_row, const float * codebook, float rms_scale, int elem)
{
    const int bit_pos  = elem * bits;
    const int byte_idx = bit_pos >> 3;
    const int shift    = bit_pos & 7;
    const int mask_val = (1 << bits) - 1;
    uint32_t window = (uint32_t)packed_row[byte_idx];
    if constexpr (8 % bits != 0) {
        // Only read the second byte when the bits straddle a byte boundary.
        // Without the guard, the last element whose 3 bits fit entirely in the
        // final byte (e.g. r=94, r=95 for packedBytes=36) reads one byte past
        // the per-cell data, which is an smem OOB for the last ring slot.
        if (shift + bits > 8) {
            window |= ((uint32_t)packed_row[byte_idx + 1] << 8);
        }
    }
    const int idx = (int)((window >> shift) & (uint32_t)mask_val);
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
//
// Codebook pre-scaling (when safe): in the V_PACKED decode loop, each
// 8-lane subgroup of a warp handles a different cell with a different
// rms (V loop: k = sgitg*WARP_SIZE + k0 + tid/nthreads_V, nthreads_V=8).
// Hoisting `cb_lane * rms` ahead of __shfl_sync is only correct when the
// shuffle source lane is in the SAME 8-lane subgroup as the destination
// — otherwise the destination gets the source's rms. shfl_w == 8 bounds
// the shuffle to one subgroup, making pre-scale safe at 2-bit and 3-bit.
// At 4-bit the codebook has 16 entries that don't fit in 8 lanes, so
// shfl_w is hardcoded to 16 — that spans two cells with different rms
// and pre-scale is INCORRECT. The 4-bit helper must keep `* rms` after
// the shuffle. See feedback_simd_shuffle_prescale_rms.md.
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
// Pre-scale OK: caller passes shfl_w=8, shuffle source is in same cell-uniform subgroup.
static __device__ __forceinline__ void tq_decode_8_3bit(
    const uint8_t * __restrict__ packed_row,
    float cb_lane, float rms, int start_elem, int shfl_w,
    float * __restrict__ out)
{
    const int byte_off = (start_elem * 3) >> 3;
    const uint32_t w = (uint32_t)packed_row[byte_off]
                     | ((uint32_t)packed_row[byte_off + 1] << 8)
                     | ((uint32_t)packed_row[byte_off + 2] << 16);
    const float cb_s = cb_lane * rms;
    out[0] = __shfl_sync(0xFFFFFFFF, cb_s, (w >>  0) & 7, shfl_w);
    out[1] = __shfl_sync(0xFFFFFFFF, cb_s, (w >>  3) & 7, shfl_w);
    out[2] = __shfl_sync(0xFFFFFFFF, cb_s, (w >>  6) & 7, shfl_w);
    out[3] = __shfl_sync(0xFFFFFFFF, cb_s, (w >>  9) & 7, shfl_w);
    out[4] = __shfl_sync(0xFFFFFFFF, cb_s, (w >> 12) & 7, shfl_w);
    out[5] = __shfl_sync(0xFFFFFFFF, cb_s, (w >> 15) & 7, shfl_w);
    out[6] = __shfl_sync(0xFFFFFFFF, cb_s, (w >> 18) & 7, shfl_w);
    out[7] = __shfl_sync(0xFFFFFFFF, cb_s, (w >> 21) & 7, shfl_w);
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
// DO NOT pre-scale: shfl_w hardcoded to 16 (codebook has 16 entries that
// don't fit in 8 lanes). 16-lane shuffle subgroup spans two cells with
// different rms, so pre-scaled `cb_lane * rms` from the source lane carries
// the wrong rms to the destination. `* rms` must run on the destination
// after the shuffle.
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
// Pre-scale OK: caller passes shfl_w=8, shuffle source is in same cell-uniform subgroup.
static __device__ __forceinline__ void tq_decode_8_2bit(
    const uint8_t * __restrict__ packed_row,
    float cb_lane, float rms, int start_elem, int shfl_w,
    float * __restrict__ out)
{
    const int byte_off = start_elem >> 2;
    const uint32_t w = (uint32_t)packed_row[byte_off] | ((uint32_t)packed_row[byte_off + 1] << 8);
    const float cb_s = cb_lane * rms;
    out[0] = __shfl_sync(0xFFFFFFFF, cb_s, (w >>  0) & 3, shfl_w);
    out[1] = __shfl_sync(0xFFFFFFFF, cb_s, (w >>  2) & 3, shfl_w);
    out[2] = __shfl_sync(0xFFFFFFFF, cb_s, (w >>  4) & 3, shfl_w);
    out[3] = __shfl_sync(0xFFFFFFFF, cb_s, (w >>  6) & 3, shfl_w);
    out[4] = __shfl_sync(0xFFFFFFFF, cb_s, (w >>  8) & 3, shfl_w);
    out[5] = __shfl_sync(0xFFFFFFFF, cb_s, (w >> 10) & 3, shfl_w);
    out[6] = __shfl_sync(0xFFFFFFFF, cb_s, (w >> 12) & 3, shfl_w);
    out[7] = __shfl_sync(0xFFFFFFFF, cb_s, (w >> 14) & 3, shfl_w);
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
    // Outlier-split dual-stream decode params (only used when HAS_OUTLIERS).
    // When HAS_OUTLIERS, the codebook array is concatenated as
    //   [regular (1<<bits), outlier (1<<outlier_bits)]
    // and outlier_count > 0.
    const uint8_t * __restrict__ outlier_packed,  // [outlierPackedBytes*nKVHeads, capacity] i8
    const float   * __restrict__ outlier_scales,  // [nKVHeads, capacity] f32
    const int16_t * __restrict__ outlier_indices, // [outlierCount*nKVHeads, capacity] i16
    const float   * __restrict__ outlier_zeros,   // [nKVHeads, capacity] f32 (NULL if !asymmetric)
    int     outlier_bits,
    int     outlier_count,
    int     outlier_packedBytes,
    // Indexed-addressing locs. NULL = contiguous (cache cell = firstCell+c).
    // [nCells] i32 = physical cache slot for dense cell c. Block-uniform branch.
    const int32_t * __restrict__ locs
)
{
#ifdef FLASH_ATTN_AVAILABLE
    // Skip logit_softcap variants for unsupported D values (mirrors original kernel guard).
    if (use_logit_softcap && D != 64 && D != 128 && D != 256 && D != 512) {
        GGML_UNUSED_VARS(Q, K_packed, V, mask, dst, dst_meta, scales, codebook,
            scale, logit_softcap, bits, firstCell, nCells, nKVHeads, packedBytes,
            ne00, ne01, ne02, ne03, nb01, nb02, nb03,
            nb21, nb22, nb23, ne31, nb31,
            v_scales, v_codebook, v_bits, v_packedBytes,
            zeros, asymmetric,
            outlier_packed, outlier_scales, outlier_indices, outlier_zeros,
            outlier_bits, outlier_count, outlier_packedBytes, locs);
        NO_DEVICE_CODE;
        return;
    }

    // Indexed addressing — when locs != null, each dense cell c maps to
    // physical slot locs[c]. Base pointers below stay un-offset by firstCell
    // (which is irrelevant in indexed mode); the per-iteration phys_cell
    // formula picks up the right slot.
    const bool indexed = (locs != nullptr);

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

    // Phase A: GQA-shared K+V decode. Grid is now per-KV-head per chunk:
    //   gridDim.z = nSeq * nKVHeads * num_chunks
    // where num_chunks = ceil(gqa_ratio / GQA_TILE). Each block decodes
    // K (and V_PACKED) once per cell and accumulates against up to GQA_TILE
    // query heads in the group.
    //
    // GQA_TILE is conditional on ncols:
    //   - ncols == 1 (decode): GQA_TILE=4 — full GQA amortization for the
    //     bandwidth-bound decode step. Per-thread state ≈ ncols=2's, which
    //     compiles under cc 6.1's 255-reg cap.
    //   - ncols  > 1 (prefill): GQA_TILE=1 — keeps current behavior because
    //     the gqa-expanded smem (s_Q_fixed + KQ scratch) would exceed Pascal's
    //     48 KiB/block cap at D=256, ncols=8.
    constexpr int GQA_TILE = (ncols == 1) ? 4 : 1;
    const int gqa_ratio   = ne02 / nKVHeads;
    const int num_chunks  = (gqa_ratio + GQA_TILE - 1) / GQA_TILE;

    const int sequence    = blockIdx.z / (nKVHeads * num_chunks);
    const int rest_z      = blockIdx.z % (nKVHeads * num_chunks);
    const int head_kv     = rest_z / num_chunks;
    const int chunk_idx   = rest_z % num_chunks;
    const int head_q_base = head_kv * gqa_ratio + chunk_idx * GQA_TILE;
    const int g_max = (gqa_ratio - chunk_idx * GQA_TILE) < GQA_TILE
        ? (gqa_ratio - chunk_idx * GQA_TILE) : GQA_TILE;

    // In contiguous mode K/V/scales/zeros/outlier_* are pre-offset by
    // firstCell. In indexed mode that offset is folded into locs[c] instead.
    const int cell_base = indexed ? 0 : firstCell;

    // Q is positioned at the first query head of this chunk; per-g loads
    // offset by g * nb02 inside the Q-load loop.
    Q        += (int64_t)nb03*sequence + nb02*head_q_base + nb01*ic0;
    K_packed += (int64_t)cell_base * nKVHeads * packedBytes + (int64_t)head_kv * packedBytes;
    scales   += (int64_t)cell_base * nKVHeads + head_kv;

    if (zeros) {
        zeros += (int64_t)cell_base * nKVHeads + head_kv;
    }

    // Outlier-split base-pointer adjustments. All outlier tensors are indexed
    // with cell_base as the leading cell offset (matches encode-side writes).
    if constexpr (HAS_OUTLIERS) {
        outlier_packed  += (int64_t)cell_base * nKVHeads * outlier_packedBytes + (int64_t)head_kv * outlier_packedBytes;
        outlier_scales  += (int64_t)cell_base * nKVHeads + head_kv;
        outlier_indices += (int64_t)cell_base * nKVHeads * outlier_count + (int64_t)head_kv * outlier_count;
        if (outlier_zeros) {
            outlier_zeros += (int64_t)cell_base * nKVHeads + head_kv;
        }
    }

    // V pointer setup: f16 path uses stride-based addressing; packed path uses cell-index addressing.
    const uint8_t * V_packed_base = nullptr;
    const float   * v_scales_base = nullptr;
    if constexpr (V_PACKED) {
        V_packed_base = (const uint8_t *)V
            + (int64_t)cell_base * nKVHeads * v_packedBytes
            + (int64_t)head_kv * v_packedBytes;
        v_scales_base = v_scales
            + (int64_t)cell_base * nKVHeads + head_kv;
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
    [[maybe_unused]] const float k_cb_lane = codebook[threadIdx.x & ((1 << bits) - 1)];
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

    // GQA-share: VKQ holds gqa_tile output rows per cell stripe per Q token.
    float2 VKQ[ncols][GQA_TILE][(D/2)/nthreads_V] = {};

    extern __shared__ float s_mem_all[];
    float * KQ = s_mem_all;
    float * s_Q_fixed = s_mem_all + (ne_KQ > ne_combine ? ne_KQ : ne_combine);

    const int tid_global = threadIdx.y * WARP_SIZE + threadIdx.x;

    // Load Q into shared memory for gqa_tile * ncols query heads/tokens.
    // s_Q_fixed layout: (g, j, elem) → index = (g * ncols + j) * D + elem.
    // Pre-scale on load so consumers can read s_Q_fixed directly. Inactive
    // slots (g >= g_max) get zero so downstream FMAs contribute 0.
    for (int i = tid_global; i < GQA_TILE * ncols * D; i += nthreads) {
        const int g    = i / (ncols * D);
        const int rest = i % (ncols * D);
        const int j    = rest / D;
        const int elem = rest % D;
        if (g < g_max) {
            const float * Q_ptr = (const float *)(Q + (int64_t)g * nb02 + (int64_t)j * nb01);
            s_Q_fixed[i] = Q_ptr[elem] * scale;
        } else {
            s_Q_fixed[i] = 0.0f;
        }
    }
    __syncthreads();

    // Phase B: K_packed cp.async ring for sm 80+ (Ampere+).
    // Two-deep ring of nthreads cells; each thread stages its own cell of
    // packedBytes bytes into ring[slot][tid*pb_aligned..tid*pb_aligned+packedBytes).
    // 16-byte cp.async chunks plus a 4/8/12-byte sync tail (packedBytes ∈
    // {12,16,20,24,36,48,56,84,112} across ship presets). Indexed mode
    // (locs != nullptr) falls through to the direct-load path: per-lane
    // cell_addr depends on locs[c] which would force a two-stage prefetch.
    //
    // Hoisted above the Q_reg/sum_q compute so that on Ampere+ the initial
    // cp.async prefetch is issued before Q_reg/sum_q is computed, letting
    // K-ring loads overlap with that scalar compute.
    // Pascal (no CP_ASYNC_AVAILABLE) sees this entire block as empty, so
    // its execution path is unchanged.
#ifdef CP_ASYNC_AVAILABLE
    // use_kasync must mirror the launcher's gate exactly: cp.async needs
    // 16-byte aligned src addresses, which requires per-cell stride
    // (nKVHeads * X_packedBytes) to be a multiple of 16 for every active
    // ring. If any ring fails the check, fall through to the #else
    // direct-load path. See launcher's k_stride_ok/o_stride_ok/v_stride_ok.
    const bool k_stride_ok = ((int64_t)nKVHeads * packedBytes & 15) == 0;
    const bool o_stride_ok = !HAS_OUTLIERS
        || (((int64_t)nKVHeads * outlier_packedBytes & 15) == 0);
    const bool v_stride_ok = !V_PACKED
        || (((int64_t)nKVHeads * v_packedBytes & 15) == 0);
    // The scalar path is the sole compute/dst producer on all arches.
    [[maybe_unused]] const bool use_kasync = !indexed && k_stride_ok && o_stride_ok && v_stride_ok;
    // pb_aligned must equal the launcher's value. Formula: 16-aligned size
    // that fits packedBytes + worst-case 16-byte alignment slack. See the
    // launcher's compute_aligned() lambda for the rationale. The
    // worst-case slack is 16 - gcd(packedBytes, 16); gcd computed via
    // `pb & -pb` (lowest set bit) capped at 16 — portable across host
    // compilers and avoids __builtin_ctz, which produced wrong runtime
    // values for this exact site on nvcc/Windows (the kernel collapsed
    // pb_aligned to packedBytes itself; observed at runtime via memcheck
    // as a 16-byte __shared__ write at stride packedBytes instead of
    // stride pb_aligned).
    const int  pb_gcd16_raw = packedBytes & -packedBytes;
    const int  pb_gcd16 = pb_gcd16_raw > 16 ? 16 : pb_gcd16_raw;
    const int  pb_aligned = (packedBytes & 15)
        ? (((packedBytes + (16 - pb_gcd16)) + 15) & ~15)
        : packedBytes;
    const int  k_chunks16 = pb_aligned >> 4;
    // align_off = (head_kv * packedBytes) % 16 is the per-block constant byte
    // offset between the per-head base in K_packed and the nearest 16-byte
    // boundary BELOW it. The cp.async src is rounded down by this amount;
    // the decoder reads from smem at offset + align_off.
    const int  k_align_off = ((int)head_kv * packedBytes) & 15;
    uint8_t * k_ring_base = (uint8_t *)(s_Q_fixed + GQA_TILE * ncols * D);
    uint8_t * k_ring[2] = {
        k_ring_base,
        k_ring_base + nthreads * pb_aligned
    };
    int k_slot = 0;

    auto k_async_issue = [&] __device__ (int slot, int k_start) {
        const int cell_n   = k_start + (int)tid;
        const int src_cell = (cell_n < nCells) ? cell_n : 0;
        // K_packed already includes head_kv * packedBytes (see kernel-start
        // offset). Subtract k_align_off to land on a 16-byte boundary.
        const uint8_t * src = (K_packed - k_align_off)
                            + (int64_t)src_cell * nKVHeads * packedBytes;
        uint8_t * dst = k_ring[slot] + tid * pb_aligned;
        const unsigned int dst_s = ggml_cuda_cvta_generic_to_shared(dst);
        for (int b = 0; b < k_chunks16; ++b) {
            cp_async_cg_16<64>(dst_s + b*16, src + b*16);
        }
    };

    // Phase B step 2: parallel outlier_packed ring. Sized only when
    // HAS_OUTLIERS is true (compile-time elided otherwise). One cp.async
    // batch covers both rings since cp_async_wait_all is coarse.
    // Same 16-byte alignment treatment as K (see comment above k_async_issue).
    [[maybe_unused]] const int ob_gcd16_raw = HAS_OUTLIERS
        ? (outlier_packedBytes & -outlier_packedBytes) : 0;
    [[maybe_unused]] const int ob_gcd16 = ob_gcd16_raw > 16 ? 16 : ob_gcd16_raw;
    [[maybe_unused]] const int ob_aligned = HAS_OUTLIERS
        ? ((outlier_packedBytes & 15)
            ? (((outlier_packedBytes + (16 - ob_gcd16)) + 15) & ~15)
            : outlier_packedBytes)
        : 0;
    [[maybe_unused]] const int o_chunks16 = HAS_OUTLIERS ? (ob_aligned >> 4) : 0;
    [[maybe_unused]] const int o_align_off = HAS_OUTLIERS
        ? (((int)head_kv * outlier_packedBytes) & 15) : 0;
    [[maybe_unused]] uint8_t * o_ring_base = HAS_OUTLIERS
        ? (k_ring_base + (size_t)2 * nthreads * pb_aligned)
        : nullptr;
    [[maybe_unused]] uint8_t * o_ring[2] = {
        o_ring_base,
        o_ring_base ? o_ring_base + (size_t)nthreads * ob_aligned : nullptr
    };

    auto o_async_issue = [&] __device__ (int slot, int k_start) {
        if constexpr (!HAS_OUTLIERS) return;
        const int cell_n   = k_start + (int)tid;
        const int src_cell = (cell_n < nCells) ? cell_n : 0;
        const uint8_t * src = (outlier_packed - o_align_off)
                            + (int64_t)src_cell * nKVHeads * outlier_packedBytes;
        uint8_t * dst = o_ring[slot] + tid * ob_aligned;
        const unsigned int dst_s = ggml_cuda_cvta_generic_to_shared(dst);
        for (int b = 0; b < o_chunks16; ++b) {
            cp_async_cg_16<64>(dst_s + b*16, src + b*16);
        }
    };

    // Phase B step 3: parallel V_packed ring. Sized only when V_PACKED is
    // true (compile-time elided otherwise). V is consumed in the same
    // outer iter as K (V loop runs after the K loop within each iter), so
    // staging into the same iter-N slot as K/outlier is correct.
    // Same 16-byte alignment treatment as K (see comment above k_async_issue).
    [[maybe_unused]] const int vp_gcd16_raw = V_PACKED
        ? (v_packedBytes & -v_packedBytes) : 0;
    [[maybe_unused]] const int vp_gcd16 = vp_gcd16_raw > 16 ? 16 : vp_gcd16_raw;
    [[maybe_unused]] const int vp_aligned = V_PACKED
        ? ((v_packedBytes & 15)
            ? (((v_packedBytes + (16 - vp_gcd16)) + 15) & ~15)
            : v_packedBytes)
        : 0;
    [[maybe_unused]] const int v_chunks16 = V_PACKED ? (vp_aligned >> 4) : 0;
    [[maybe_unused]] const int v_align_off = V_PACKED
        ? (((int)head_kv * v_packedBytes) & 15) : 0;
    [[maybe_unused]] uint8_t * v_ring_base = V_PACKED
        ? (k_ring_base
           + (size_t)2 * nthreads * pb_aligned
           + (HAS_OUTLIERS ? (size_t)2 * nthreads * ob_aligned : 0))
        : nullptr;
    [[maybe_unused]] uint8_t * v_ring[2] = {
        v_ring_base,
        v_ring_base ? v_ring_base + (size_t)nthreads * vp_aligned : nullptr
    };

    auto v_async_issue = [&] __device__ (int slot, int k_start) {
        if constexpr (!V_PACKED) return;
        const int cell_n   = k_start + (int)tid;
        const int src_cell = (cell_n < nCells) ? cell_n : 0;
        const uint8_t * src = (V_packed_base - v_align_off)
                            + (int64_t)src_cell * nKVHeads * v_packedBytes;
        uint8_t * dst = v_ring[slot] + tid * vp_aligned;
        const unsigned int dst_s = ggml_cuda_cvta_generic_to_shared(dst);
        for (int b = 0; b < v_chunks16; ++b) {
            cp_async_cg_16<64>(dst_s + b*16, src + b*16);
        }
    };

    if (use_kasync) {
        k_async_issue(0, blockIdx.y * nthreads);
        if constexpr (HAS_OUTLIERS) {
            o_async_issue(0, blockIdx.y * nthreads);
        }
        if constexpr (V_PACKED) {
            v_async_issue(0, blockIdx.y * nthreads);
        }
        cp_async_wait_all();
        __syncthreads();
    }
#else
    // Pascal / Turing (no CP_ASYNC_AVAILABLE): no Phase B rings exist;
    // all K/outlier/V loads below use the direct scalar path.
#endif

    // Scalar running max/sum trackers (online softmax).
    float KQ_max[ncols][GQA_TILE];
    float KQ_sum[ncols][GQA_TILE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int g = 0; g < GQA_TILE; ++g) {
            KQ_max[j][g] = -FLT_MAX/2.0f;
            KQ_sum[j][g] = 0.0f;
        }
    }

    // tid_kq is the lane index within a KQ group (group size = nthreads_KQ).
    const int tid_kq = threadIdx.x % nthreads_KQ;

    // Load Q into registers from shared memory (stable divergent access).
    // s_Q_fixed is already pre-scaled, no additional multiply needed.
    float2 Q_reg[ncols][GQA_TILE][(D/2)/nthreads_KQ];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int g = 0; g < GQA_TILE; ++g) {
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                const int i = k * nthreads_KQ + tid_kq;
                const int s_idx = ((g * ncols) + j) * D + 2*i;
                Q_reg[j][g][k].x = s_Q_fixed[s_idx];
                Q_reg[j][g][k].y = s_Q_fixed[s_idx + 1];
            }
        }
    }

    // Per-(j,g) sum of Q elements (asymmetric zero contribution).
    float sum_q[ncols][GQA_TILE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int g = 0; g < GQA_TILE; ++g) {
            float local_sum = 0.0f;
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                local_sum += Q_reg[j][g][k].x + Q_reg[j][g][k].y;
            }
            sum_q[j][g] = warp_reduce_sum<nthreads_KQ>(local_sum);
        }
    }

    // Main KV loop. When gridDim.y > 1, this block handles every gridDim.y'th
    // chunk of K cells starting at blockIdx.y. Initial K_packed/V/maskh offsets
    // for blockIdx.y are applied above; the per-iteration increment uses gridDim.y
    // so successive iterations skip over slices owned by sibling blocks.
    //
    // In indexed mode we force gridDim.y=1 in the launcher and read V at
    // absolute slot locs[c] every iteration, so the V += stride increment must
    // not fire (V stays at its sequence/head base).
    for (int k_VKQ_0 = blockIdx.y * nthreads; k_VKQ_0 < nCells; k_VKQ_0 += gridDim.y * nthreads,
             V += ((V_PACKED || indexed) ? 0 : (int64_t)gridDim.y * nthreads * nb21),
             maskh += (maskh ? gridDim.y * nthreads : 0)) {

        // Phase B: issue cp.async for the next outer-iter's K_packed slice into
        // the alternate ring slot. Compute below (K decode + softmax + V) runs
        // while the load is in flight; the matching cp_async_wait_all at the
        // end of the iter completes the overlap.
#ifdef CP_ASYNC_AVAILABLE
        const int  k_VKQ_next = k_VKQ_0 + (int)gridDim.y * nthreads;
        const bool kasync_prefetch = use_kasync && k_VKQ_next < nCells;
        const int  k_slot_next = 1 - k_slot;
        if (kasync_prefetch) {
            k_async_issue(k_slot_next, k_VKQ_next);
            if constexpr (HAS_OUTLIERS) {
                o_async_issue(k_slot_next, k_VKQ_next);
            }
            if constexpr (V_PACKED) {
                v_async_issue(k_slot_next, k_VKQ_next);
            }
        }
#endif

        // Scalar K decode + Q·K + online softmax + V accumulate.
        float KQ_reg[ncols][GQA_TILE];
        float KQ_max_new[ncols][GQA_TILE];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int g = 0; g < GQA_TILE; ++g) {
                KQ_max_new[j][g] = KQ_max[j][g];
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE
                           + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1)))
                           + i_KQ_0;
            const int cell_rel = k_VKQ_0 + i_KQ;
            const bool in_range = (cell_rel < nCells);
            // Physical cache slot for this dense cell — locs[c] when indexed,
            // cell_rel otherwise. Used for K/V/scales/zeros/outlier addressing
            // off the firstCell-adjusted bases (or off the raw bases in
            // indexed mode where cell_base was set to 0).
            const int cell_addr = (indexed && in_range) ? (int)locs[cell_rel] : cell_rel;

            // Outlier-split: build per-lane register bitmap from outlier_indices.
            // sum_q_outl[j][g] accumulates Σ_d∈outlier Q[(j,g)][d] across the
            // outlier slots, for asymmetric zero re-addition later.
            [[maybe_unused]] float sum_q_outl[ncols][GQA_TILE];
            [[maybe_unused]] const int16_t * o_idx_cell_saved = nullptr;
            [[maybe_unused]] uint32_t bmap[(D + 31) / 32] = {0};
            if constexpr (HAS_OUTLIERS) {
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
#pragma unroll
                    for (int g = 0; g < GQA_TILE; ++g) {
                        sum_q_outl[j][g] = 0.0f;
                    }
                }

                const int16_t * o_idx_cell = in_range
                    ? outlier_indices + (int64_t)cell_addr * nKVHeads * outlier_count
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

                // Precompute Σ_s Q[outlier_pos[s]] per (j, g) for asymmetric
                // fixup. Must be unconditional across the warp because the
                // s_Q_fixed reads share an index across lanes; divergent
                // lane groups in the same warp (different cell_rel, hence
                // different in_range) would still execute the read safely
                // since s_Q_fixed is small smem.
#pragma unroll
                for (int s = 0; s < 32; ++s) {
                    if (s >= outlier_count) break;
                    const int pos = (in_range && o_idx_cell) ? (int)o_idx_cell[s] : 0;
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
#pragma unroll
                        for (int g = 0; g < GQA_TILE; ++g) {
                            sum_q_outl[j][g] += s_Q_fixed[(g * ncols + j) * D + pos];
                        }
                    }
                }
#ifndef GGML_USE_HIP
                __syncwarp();
#endif
            }

            // ----------------------------------------------------------------
            // Q-tile decode amortization.
            //
            // K (and outliers) are decoded once per Q-tile and held in a
            // per-lane register cache; the inner `j` (ncols) loop becomes a
            // pure FMA against s_Q_fixed[j*D + d]. Same pattern for the
            // outlier sub-stream.
            //
            // Storage:
            //   k_lane[D/nthreads_KQ]      — regular-stream decoded K at strided d's
            //   o_val_lane[outlier_count/nthreads_KQ_max] — outlier-stream decoded K
            //   o_pos_lane[outlier_count/nthreads_KQ_max] — outlier head-dim positions
            // ----------------------------------------------------------------

            // Clamp cell_addr to 0 for out-of-range lanes. The K decode below
            // is unconditional across the warp (needed for shfl convergence);
            // gating only the result via in_range still issues the load, so
            // an unclamped cell_addr would dereference past K_packed end.
            // Cell 0 is always valid; the dequantized k_val is discarded by
            // softmax masking downstream for out-of-range cells.
            const int safe_cell_addr = in_range ? cell_addr : 0;
            // Phase B: on sm 80+ non-indexed mode, K bytes for this iter's
            // nthreads cells were cp.async-staged into k_ring[k_slot] (one
            // cell per tid at offset tid*pb_aligned). The decoder reads
            // packed_row[byte_off] which transparently retargets to smem.
#ifdef CP_ASYNC_AVAILABLE
            const uint8_t * packed_row = use_kasync
                ? (k_ring[k_slot] + (int64_t)i_KQ * pb_aligned + k_align_off)
                : (K_packed + (int64_t)safe_cell_addr * nKVHeads * packedBytes);
#else
            const uint8_t * packed_row = K_packed + (int64_t)safe_cell_addr * nKVHeads * packedBytes;
#endif
            const float     rms_scale  = in_range ? scales[cell_addr * nKVHeads] : 0.0f;

            // Regular-stream decode: hoisted out of `j`.
            constexpr int K_PER_LANE = D / nthreads_KQ;          // 16 for D=128, 8 for D=64, 32 for D=256
            float k_lane[K_PER_LANE];
            // Per-lane outlier-channel mask & rank-below-d, only meaningful
            // when HAS_OUTLIERS. For the regular stream, k_lane[ki] holds the
            // dequantized regular K at head-dim position d=tid_kq+ki*nthreads_KQ;
            // when the position is itself an outlier, the entry is forced to 0
            // so the unconditional FMA in the j-loop produces the correct
            // (zero) contribution from regular-stream K.
            // Hoist bits dispatch outside ki-loop: each arm sees a compile-time
            // constant, letting tq_decode_elem<N> elide the second-byte load for
            // bits=4 and bits=2.  bits is uniform per kernel launch; the branch
            // costs one SETP per kernel, not one per iteration.
            if (bits == 4) {
                if constexpr (HAS_OUTLIERS) {
                    constexpr int N_WORDS = (D + 31) / 32;
                    int prefix[N_WORDS + 1];
                    prefix[0] = 0;
                    #pragma unroll
                    for (int w = 0; w < N_WORDS; ++w) {
                        prefix[w + 1] = prefix[w] + __popc(bmap[w]);
                    }
                    #pragma unroll
                    for (int ki = 0; ki < K_PER_LANE; ++ki) {
                        const int d = tid_kq + ki * nthreads_KQ;
                        const bool is_outl = (bmap[d >> 5] >> (d & 31)) & 1u;
                        const int w_d   = d >> 5;
                        const int bit_d = d & 31;
                        const int outl_below = prefix[w_d]
                            + (bit_d ? __popc(bmap[w_d] & ((1u << bit_d) - 1u)) : 0);
                        const int r_raw = d - outl_below;
                        const int r = is_outl ? 0 : r_raw;
                        k_lane[ki] = is_outl ? 0.0f
                            : tq_decode_elem<4>(packed_row, codebook, rms_scale, r);
                    }
                } else {
                    #pragma unroll
                    for (int ki = 0; ki < K_PER_LANE; ++ki) {
                        const int d = tid_kq + ki * nthreads_KQ;
                        k_lane[ki] = tq_decode_elem<4>(packed_row, codebook, rms_scale, d);
                    }
                }
            } else if (bits == 2) {
                if constexpr (HAS_OUTLIERS) {
                    constexpr int N_WORDS = (D + 31) / 32;
                    int prefix[N_WORDS + 1];
                    prefix[0] = 0;
                    #pragma unroll
                    for (int w = 0; w < N_WORDS; ++w) {
                        prefix[w + 1] = prefix[w] + __popc(bmap[w]);
                    }
                    #pragma unroll
                    for (int ki = 0; ki < K_PER_LANE; ++ki) {
                        const int d = tid_kq + ki * nthreads_KQ;
                        const bool is_outl = (bmap[d >> 5] >> (d & 31)) & 1u;
                        const int w_d   = d >> 5;
                        const int bit_d = d & 31;
                        const int outl_below = prefix[w_d]
                            + (bit_d ? __popc(bmap[w_d] & ((1u << bit_d) - 1u)) : 0);
                        const int r_raw = d - outl_below;
                        const int r = is_outl ? 0 : r_raw;
                        k_lane[ki] = is_outl ? 0.0f
                            : tq_decode_elem<2>(packed_row, codebook, rms_scale, r);
                    }
                } else {
                    #pragma unroll
                    for (int ki = 0; ki < K_PER_LANE; ++ki) {
                        const int d = tid_kq + ki * nthreads_KQ;
                        k_lane[ki] = tq_decode_elem<2>(packed_row, codebook, rms_scale, d);
                    }
                }
            } else { // bits == 3
                if constexpr (HAS_OUTLIERS) {
                    constexpr int N_WORDS = (D + 31) / 32;
                    int prefix[N_WORDS + 1];
                    prefix[0] = 0;
                    #pragma unroll
                    for (int w = 0; w < N_WORDS; ++w) {
                        prefix[w + 1] = prefix[w] + __popc(bmap[w]);
                    }
                    #pragma unroll
                    for (int ki = 0; ki < K_PER_LANE; ++ki) {
                        const int d = tid_kq + ki * nthreads_KQ;
                        const bool is_outl = (bmap[d >> 5] >> (d & 31)) & 1u;
                        const int w_d   = d >> 5;
                        const int bit_d = d & 31;
                        const int outl_below = prefix[w_d]
                            + (bit_d ? __popc(bmap[w_d] & ((1u << bit_d) - 1u)) : 0);
                        const int r_raw = d - outl_below;
                        const int r = is_outl ? 0 : r_raw;
                        k_lane[ki] = is_outl ? 0.0f
                            : tq_decode_elem<3>(packed_row, codebook, rms_scale, r);
                    }
                } else {
                    #pragma unroll
                    for (int ki = 0; ki < K_PER_LANE; ++ki) {
                        const int d = tid_kq + ki * nthreads_KQ;
                        k_lane[ki] = tq_decode_elem<3>(packed_row, codebook, rms_scale, d);
                    }
                }
            }

            // Outlier-stream decode: hoisted out of `j`. Each lane handles
            // 32/nthreads_KQ outlier slots (outlier_count ≤ 32 by construction).
            constexpr int O_PER_LANE = 32 / nthreads_KQ;         // 4 (assumes nthreads_KQ=8)
            [[maybe_unused]] float o_val_lane[O_PER_LANE];
            [[maybe_unused]] int   o_pos_lane[O_PER_LANE];
            if constexpr (HAS_OUTLIERS) {
                // Same clamp as the regular K_packed pointer above: the
                // tq_decode_elem call in the outlier loop is unconditional, so
                // an unclamped cell_addr would OOB on out-of-range lanes.
                // Phase B step 2: on sm 80+ non-indexed mode, outlier_packed
                // bytes for this iter were cp.async-staged into o_ring[k_slot].
#ifdef CP_ASYNC_AVAILABLE
                const uint8_t * o_packed_row = use_kasync
                    ? (o_ring[k_slot] + (int64_t)i_KQ * ob_aligned + o_align_off)
                    : (outlier_packed + (int64_t)safe_cell_addr * nKVHeads * outlier_packedBytes);
#else
                const uint8_t * o_packed_row = outlier_packed
                    + (int64_t)safe_cell_addr * nKVHeads * outlier_packedBytes;
#endif
                const float o_rms = in_range
                    ? outlier_scales[cell_addr * nKVHeads] : 0.0f;
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
                ? (zeros && in_range ? zeros[cell_addr * nKVHeads] : 0.0f)
                : (asymmetric && zeros && in_range ? zeros[cell_addr * nKVHeads] : 0.0f));
            [[maybe_unused]] const float out_zero_val =
                (HAS_OUTLIERS && outlier_zeros && in_range)
                    ? outlier_zeros[cell_addr * nKVHeads] : 0.0f;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int g = 0; g < GQA_TILE; ++g) {
                    float sum;

                    if constexpr (HAS_OUTLIERS) {
                        // Regular stream: pure FMA with cached K decode.
                        float reg_sum = 0.0f;
                        #pragma unroll
                        for (int ki = 0; ki < K_PER_LANE; ++ki) {
                            const int d = tid_kq + ki * nthreads_KQ;
                            reg_sum += s_Q_fixed[(g * ncols + j) * D + d] * k_lane[ki];
                        }
                        // Outlier stream: pure FMA with cached outlier decode.
                        float out_sum = 0.0f;
                        #pragma unroll
                        for (int s_it = 0; s_it < O_PER_LANE; ++s_it) {
                            out_sum += s_Q_fixed[(g * ncols + j) * D + o_pos_lane[s_it]] * o_val_lane[s_it];
                        }
                        sum = reg_sum + out_sum;
                    } else {
                        // Non-outlier path: pure FMA with cached K decode.
                        float reg_sum = 0.0f;
                        #pragma unroll
                        for (int ki = 0; ki < K_PER_LANE; ++ki) {
                            const int d = tid_kq + ki * nthreads_KQ;
                            reg_sum += s_Q_fixed[(g * ncols + j) * D + d] * k_lane[ki];
                        }
                        sum = reg_sum;
                    }
                    sum = warp_reduce_sum<nthreads_KQ>(sum);

                    // Asymmetric primary quantization: add zero * sum(Q) to the dot product.
                    if constexpr (HAS_OUTLIERS) {
                        if (asymmetric && (zeros || outlier_zeros)) {
#ifdef GGML_USE_HIP
                            asm volatile("" ::: "memory");
#endif
                            sum += reg_zero_val * (sum_q[j][g] - sum_q_outl[j][g])
                                 + out_zero_val * sum_q_outl[j][g];
                        }
                    } else if (asymmetric && zeros) {
                        sum += reg_zero_val * sum_q[j][g];
                    }

                    if (use_logit_softcap) {
                        sum = logit_softcap * tanhf(sum);
                    }

                    // Guard mask read with in_range: out-of-range threads (cell_addr
                    // >= nCells, last partial block) would read past mask end. Omitting
                    // the mask for those lanes is safe — sum is overwritten to -FLT_MAX/2
                    // by the in_range check below regardless of the mask value.
                    if (in_range && maskh && (ncols == 1 || ic0 + j < (int)ne01.z)) {
                        sum += __half2float(maskh[j*ne31 + i_KQ]);
                    }

                    // Zero out lanes for inactive g slots so they don't contribute.
                    if (!in_range || g >= g_max) {
                        sum = -FLT_MAX/2.0f;
                    }

                    KQ_max_new[j][g] = fmaxf(KQ_max_new[j][g], sum + FATTN_KQ_MAX_OFFSET);

                    if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == (uint32_t)i_KQ_0) {
                        KQ_reg[j][g] = sum;
                    }
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int g = 0; g < GQA_TILE; ++g) {
#pragma unroll
                for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                    KQ_max_new[j][g] = fmaxf(KQ_max_new[j][g],
                        __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j][g], offset, WARP_SIZE));
                }
                const float KQ_max_scale = expf(KQ_max[j][g] - KQ_max_new[j][g]);
                KQ_max[j][g] = KQ_max_new[j][g];

                KQ_reg[j][g] = expf(KQ_reg[j][g] - KQ_max[j][g]);
                KQ_sum[j][g] = KQ_sum[j][g]*KQ_max_scale + KQ_reg[j][g];
                // Smem layout: KQ[(j * GQA_TILE + g) * nthreads + tid]
                KQ[(j * GQA_TILE + g) * nthreads + tid] = KQ_reg[j][g];

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                    VKQ[j][g][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                    VKQ[j][g][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
                }
            }
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0
                        + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

            float KQ_k[ncols][GQA_TILE];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int g = 0; g < GQA_TILE; ++g) {
                    KQ_k[j][g] = KQ[(j * GQA_TILE + g) * nthreads + k];
                }
            }

            if constexpr (V_PACKED) {
                // Decode V from packed buffer inline — warp-shuffle batch decode.
                // Always execute decodes to keep all lanes convergent for shuffles.
                // v_rms=0 for out-of-range cells produces zero contributions.
                const int cell_rel = k_VKQ_0 + k;
                const bool   v_in_range = (cell_rel < nCells);
                const int    v_cell_addr = (indexed && v_in_range) ? (int)locs[cell_rel] : cell_rel;
                // Clamp v_cell_addr to 0 for out-of-range lanes (matches the K
                // decode clamp above). tq_decode_N_shfl below is unconditional
                // to keep all lanes convergent for the shuffles, so an
                // unclamped v_cell_addr would dereference past V_packed end.
                // v_rms=0 still discards the decoded value mathematically.
                const int v_safe_cell_addr = v_in_range ? v_cell_addr : 0;
                // Phase B step 3: on sm 80+ non-indexed mode, V_packed bytes
                // for this iter were cp.async-staged into v_ring[k_slot].
                // k is the warp-local cell index in [0, nthreads).
#ifdef CP_ASYNC_AVAILABLE
                const uint8_t * v_row = use_kasync
                    ? (v_ring[k_slot] + (int64_t)k * vp_aligned + v_align_off)
                    : (V_packed_base + (int64_t)v_safe_cell_addr * nKVHeads * v_packedBytes);
#else
                const uint8_t * v_row = V_packed_base
                    + (int64_t)v_safe_cell_addr * nKVHeads * v_packedBytes;
#endif
                const float v_rms = v_in_range
                    ? v_scales_base[v_cell_addr * nKVHeads] : 0.0f;

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
#pragma unroll
                            for (int g = 0; g < GQA_TILE; ++g) {
                                VKQ[j][g][i_VKQ_0/nthreads_V + i_VKQ_1].x += v_dec[2*i_VKQ_1]   * KQ_k[j][g];
                                VKQ[j][g][i_VKQ_0/nthreads_V + i_VKQ_1].y += v_dec[2*i_VKQ_1+1] * KQ_k[j][g];
                            }
                        }
                    }
                }
            } else {
                // f16 V path. Clamp to valid range: for out-of-range k,
                // KQ_k[j]==0 so the contribution is 0*finite=0. Without
                // clamping, V + k*nb21 is an OOB read that may hit NaN from
                // prior GPU allocations, and 0.0f*NaN == NaN in IEEE 754.
                //
                // In indexed mode the dense cell maps to physical slot
                // locs[k_VKQ_0+k] in the f16 V cache.
                const int v_dense   = k_VKQ_0 + k;
                const bool v_valid  = v_dense < nCells;
                const int v_addr    = indexed ? (v_valid ? (int)locs[v_dense] : 0) : (v_valid ? k : 0);
#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                    float2 tmp[V_rows_per_thread/2];
                    dequantize_V(V + (int64_t)v_addr*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                        for (int j = 0; j < ncols; ++j) {
#pragma unroll
                            for (int g = 0; g < GQA_TILE; ++g) {
                                VKQ[j][g][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x * KQ_k[j][g];
                                VKQ[j][g][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y * KQ_k[j][g];
                            }
                        }
                    }
                }
            }
        }
        // Phase B: wait for next-iter's K/outlier/V cp.async to complete,
        // then swap ring slots. cp_async_wait_all is coarse — it waits for
        // every in-flight cp.async issued by this thread. That is fine
        // today because all cp.async issues happen at the top of the iter
        // (k_async_issue + o_async_issue + v_async_issue), but any future
        // staging added inside the loop body must coordinate with this
        // wait point. The wait is no-op when no prefetch was issued (last
        // iter or indexed mode).
#ifdef CP_ASYNC_AVAILABLE
        if (kasync_prefetch) {
            cp_async_wait_all();
            __syncthreads();
            k_slot = k_slot_next;
        }
#endif
    } // end KV loop

    // --- Reduce across warps and write output ---

    // GQA-share output: serialize cross-warp reduction over g_VKQ. KQ smem
    // buffer (sized for V combine) is reused per-g iteration; the KQ_max /
    // KQ_sum shared buffers are likewise reused.
    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];

    // Scalar reduce-across-warps + dst write.
#pragma unroll
    for (int g_VKQ = 0; g_VKQ < GQA_TILE; ++g_VKQ) {
        if (g_VKQ >= g_max) break;

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
                KQ_max_shared[j][threadIdx.y] = KQ_max[j][g_VKQ];
            }
        }
        __syncthreads();

        const int head_q = head_q_base + g_VKQ;

#pragma unroll
        for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
            if (ncols > 1 && ic0 + j_VKQ >= (int)ne01.z) {
                break;
            }

            float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
            kqmax_new = warp_reduce_max(kqmax_new);
            const float kqmax_scale = expf(KQ_max[j_VKQ][g_VKQ] - kqmax_new);
            KQ_max[j_VKQ][g_VKQ] = kqmax_new;

            float2 * VKQ_tmp = (float2 *)KQ + threadIdx.y*(V_cols_per_iter*D/2)
                + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j_VKQ][g_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
                VKQ[j_VKQ][g_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                const int i_VKQ = i_VKQ_0
                    + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);
                ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,
                                    &VKQ[j_VKQ][g_VKQ][i_VKQ_0/nthreads_V]);
                ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4,
                                    &VKQ[j_VKQ][g_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
            }

            KQ_sum[j_VKQ][g_VKQ] *= kqmax_scale;
            KQ_sum[j_VKQ][g_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ][g_VKQ]);
            if (threadIdx.x == 0) {
                KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ][g_VKQ];
            }
            __syncthreads();

            if (nthreads <= D || tid < D) {
                KQ_sum[j_VKQ][g_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
                KQ_sum[j_VKQ][g_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ][g_VKQ]);

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
                    if (gridDim.y == 1) {
                        dst_val /= KQ_sum[j_VKQ][g_VKQ];
                    }
                    dst[((((int64_t)sequence*(int)ne01.z + ic0 + j_VKQ)*ne02 + head_q)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
                }

                if (gridDim.y != 1 && tid == 0) {
                    const int64_t meta_offset =
                        (((int64_t)sequence*(int)ne01.z + ic0 + j_VKQ)*ne02 + head_q)*gridDim.y + blockIdx.y;
                    dst_meta[meta_offset] = make_float2(KQ_max[j_VKQ][g_VKQ], KQ_sum[j_VKQ][g_VKQ]);
                }
            }

            if (j_VKQ < ncols-1) {
                __syncthreads();
            }
        }

        // Sync before reusing KQ smem (V combine) and KQ_max/sum_shared for next g_VKQ.
        if (g_VKQ + 1 < g_max) {
            __syncthreads();
        }
    }

#else
    GGML_UNUSED_VARS(Q, K_packed, V, mask, dst, dst_meta, scales, codebook,
        scale, logit_softcap, bits, firstCell, nCells, nKVHeads, packedBytes,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        nb21, nb22, nb23, ne31, nb31,
        v_scales, v_codebook, v_bits, v_packedBytes,
        zeros, asymmetric,
        outlier_packed, outlier_scales, outlier_indices, outlier_zeros,
        outlier_bits, outlier_count, outlier_packedBytes);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// Host-side launch wrapper. Included here (rather than tq-fattn.cu) so that
// tq-fattn-konly-outlier.cu can share it without duplicating the function body.
template<int D, int ncols, bool use_logit_softcap, bool V_PACKED, bool HAS_OUTLIERS>
void tq_fattn_vec_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                         float scale, float logit_softcap,
                         int bits, int firstCell, int nCells, int nKVHeads, int packedBytes,
                         int v_bits, int v_packedBytes,
                         const float * v_scales_ptr, const float * v_codebook_ptr,
                         const float   * zeros_ptr,
                         int             asymmetric,
                         const uint8_t * outlier_packed_ptr,
                         const float   * outlier_scales_ptr,
                         const int16_t * outlier_indices_ptr,
                         const float   * outlier_zeros_ptr,
                         int             outlier_bits,
                         int             outlier_count,
                         int             outlier_packed_bytes,
                         const int32_t * locs_ptr)
{
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K_p  = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * scales   = dst->src[4];
    const ggml_tensor * codebook = dst->src[5];

    GGML_ASSERT(Q->ne[0] == D);
    GGML_ASSERT(Q->type  == GGML_TYPE_F32);
    if constexpr (!V_PACKED) {
        GGML_ASSERT(V->type  == GGML_TYPE_F16);
    } else {
        GGML_ASSERT(V->type  == GGML_TYPE_I8);
    }

    const int nTokensQ = (int)Q->ne[1];
    const int nHeadsQ  = (int)Q->ne[2];
    const int nSeq     = (int)Q->ne[3];

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    const uint3 ne01 = init_fastdiv_values((uint64_t)nTokensQ);

    const int ntiles_x = (nTokensQ + ncols - 1) / ncols;
    dim3 threads(WARP_SIZE, 4);

    // Match the kernel's GQA_TILE: 4 for decode (ncols==1), 1 otherwise.
    constexpr int GQA_TILE = (ncols == 1) ? 4 : 1;
    const int gqa_ratio  = nHeadsQ / nKVHeads;
    const int num_chunks = (gqa_ratio + GQA_TILE - 1) / GQA_TILE;

    constexpr int nthreads = 128;
    // ne_KQ scales with GQA_TILE — kernel stores softmax probs per (j,g) at
    // KQ[(j*GQA_TILE+g)*nthreads+tid].
    constexpr int ne_KQ      = ncols * GQA_TILE * nthreads;
    constexpr int ne_combine = 16 * D;
    // s_Q_fixed holds gqa_tile * ncols * D pre-scaled Q values.
    constexpr size_t smem_base = ((ne_KQ > ne_combine ? ne_KQ : ne_combine)
                                + GQA_TILE * ncols * D) * sizeof(float);

    // Phase B: K_packed cp.async ring on sm 80+ (Ampere+). Adds two slots of
    // nthreads * pb_aligned bytes for double-buffered K decode loads. Indexed
    // mode falls through to direct loads inside the kernel; the ring slots
    // are still allocated (kernel-time gate is per-arch via __CUDA_ARCH__).
    //
    // Alignment: cp.async.cg requires 16-byte aligned src AND dst. The cell
    // stride (nKVHeads * packedBytes) is 16-aligned for ship presets, but
    // the per-head base (head_kv * packedBytes) is not 16-aligned when
    // packedBytes is not a multiple of 16 (e.g., tq3 D=128 packedBytes=36
    // → head_kv * 4 mod 16). We round the cp.async src DOWN to a 16-byte
    // boundary by subtracting align_off = (head_kv * packedBytes) % 16 in
    // the kernel, and enlarge each per-cell smem slot by exactly enough to
    // absorb the worst-case leading slack. The decoder reads packed_row
    // from smem at offset + align_off so the byte-level decode picks up
    // the correct data starting past the slack.
    //
    // Worst-case align_off = 16 - gcd(packedBytes, 16). For packedBytes
    // already 16-aligned, no slack is needed. Otherwise gcd computed via
    // largest power of 2 dividing packedBytes = 1 << ctz(packedBytes).
    // gcd(pb, 16) = lowest set bit of pb (= pb & -pb), capped at 16.
    // Portable across host compilers — avoids __builtin_ctz which is
    // GCC/Clang only and breaks MSVC host-compile.
    auto compute_aligned = [](int pb) {
        if (pb <= 0 || (pb & 15) == 0) return pb;
        const int gcd16_raw = pb & -pb;
        const int gcd16 = gcd16_raw > 16 ? 16 : gcd16_raw;
        const int worst_align = 16 - gcd16;
        return ((pb + worst_align) + 15) & ~15;
    };
    const int dev_id = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[dev_id].cc;
    const int pb_aligned   = compute_aligned(packedBytes);
    const int ob_aligned   = HAS_OUTLIERS ? compute_aligned(outlier_packed_bytes) : 0;
    const int vp_aligned   = V_PACKED     ? compute_aligned(v_packedBytes)        : 0;

    // Per-ring cp.async gate. cp.async.cg requires 16-byte aligned src
    // addresses, which means the per-cell stride (nKVHeads * X_packedBytes)
    // must be a multiple of 16. For ship presets this is true except when
    // nKVHeads == 1 (e.g., gemma3:1b D=256 tq3: nKVHeads=1, packedBytes=84,
    // stride=84). In that case per-cell increments cycle through non-16
    // alignments and cp.async would fault. Disable the ring for that case;
    // the kernel falls through to direct loads via use_kasync = false
    // semantics. The three rings gate independently — outlier and V may
    // still pipeline even when K can't (rare in practice).
    // Ring smem is part of the kernel's per-ARCH layout: when compiled with
    // CP_ASYNC_AVAILABLE (sm_80+) the kernel anchors the k/o/v rings right
    // after smem_base regardless of whether cp.async is actually *used* this
    // launch. The per-ring 16-byte stride check (nKVHeads * X_packedBytes
    // multiple of 16) gates only cp.async USAGE — that gate lives kernel-side
    // (`use_kasync`, this file ~line 568). It must NOT gate the ALLOCATION
    // here: when the stride check fails (e.g. gemma3 D=256, nKVHeads=1,
    // packedBytes=84, stride=84) the kernel still reserves the ring region, so
    // zero-sizing it leaves the kernel reading ring smem the launcher never
    // allocated (observed as a __shared__ OOB on sm_120 for forced K+V fused).
    // Allocate whenever the arch has cp.async; the kernel falls back to direct
    // loads internally. Pascal/Turing (no CP_ASYNC_AVAILABLE) reserve nothing,
    // so cp_async_available(cc)==false → no ring bytes → unchanged.
    const bool rings_allocated = cp_async_available(cc);

    const size_t k_ring_bytes = rings_allocated ? (size_t)2 * nthreads * pb_aligned : 0;
    const size_t o_ring_bytes = (rings_allocated && HAS_OUTLIERS)
        ? (size_t)2 * nthreads * ob_aligned : 0;
    const size_t v_ring_bytes = (rings_allocated && V_PACKED)
        ? (size_t)2 * nthreads * vp_aligned : 0;

    const size_t smem = smem_base + k_ring_bytes + o_ring_bytes + v_ring_bytes;

    const int ntiles_total = ntiles_x * nKVHeads * num_chunks * nSeq;
    const int ntiles_KQ = std::max(1, (nCells + nthreads - 1) / nthreads);

    auto kernel_ptr = tq_flash_attn_ext_vec<D, ncols, use_logit_softcap, V_PACKED, HAS_OUTLIERS>;

    // Phase B: raise the per-block dynamic smem cap when cp.async rings push
    // us past the default. The launcher's `smem` already accounts for the
    // three rings (k/o/v) when cp_async_available(cc); without bumping the
    // cap, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns 0 on archs
    // where the default cap is ≤ smem (observed on Blackwell sm 100 with tq4
    // at D=128: 46 KiB > 48 KiB default-conservative cap). Setting the
    // attribute is idempotent across launches; no harm calling every time.
    if (cp_async_available(cc) && (k_ring_bytes + o_ring_bytes + v_ring_bytes) > 0) {
        const cudaError_t attr_err = cudaFuncSetAttribute(
            kernel_ptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem);
        if (attr_err != cudaSuccess) {
            // Hardware can't grant the requested smem (cap < smem). Fall back
            // by clearing the error; the occupancy query below will return 0
            // and the assert will fire with a clearer diagnostic context.
            (void)cudaGetLastError();
        }
    }

    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kernel_ptr,
        (int)(threads.x * threads.y * threads.z), smem));
    GGML_ASSERT(max_blocks_per_sm > 0);

    const int nsm    = ggml_cuda_info().devices[dev_id].nsm;
    const int blocks_per_wave = nsm * max_blocks_per_sm;

    int parallel_blocks = 1;
    // Indexed (locs != null) mode requires a single KV block — multi-block KV
    // splitting assumes contiguous cell ranges per block, which doesn't hold
    // when each cell maps to an independent physical slot via locs[c].
    if (locs_ptr == nullptr && ntiles_total < blocks_per_wave && ntiles_KQ > 1) {
        int nwaves_best = 1;
        int eff_best    = (100 * ntiles_total) / blocks_per_wave;
        for (int test = std::min(max_blocks_per_sm, ntiles_KQ); test <= ntiles_KQ; ++test) {
            const int nblocks_total = ntiles_total * test;
            const int nwaves        = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int eff_pct       = nwaves > 0 ? (100 * nblocks_total) / (nwaves * blocks_per_wave) : 0;
            if (eff_best >= 95 && nwaves > nwaves_best) { break; }
            if (eff_pct > eff_best) {
                nwaves_best     = nwaves;
                eff_best        = eff_pct;
                parallel_blocks = test;
            }
        }
    }

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);
    if (parallel_blocks > 1) {
        const size_t kqv_n    = (size_t)D * nHeadsQ * nTokensQ * nSeq;
        const size_t kqv_rows = (size_t)nHeadsQ * nTokensQ * nSeq;
        dst_tmp.alloc((size_t)parallel_blocks * kqv_n);
        dst_tmp_meta.alloc((size_t)parallel_blocks * kqv_rows);
    }

    dim3 blocks(ntiles_x, parallel_blocks, nKVHeads * num_chunks * nSeq);

    float  * kernel_dst      = (parallel_blocks > 1) ? dst_tmp.ptr      : (float  *)dst->data;
    float2 * kernel_dst_meta = (parallel_blocks > 1) ? dst_tmp_meta.ptr : (float2 *)nullptr;

    kernel_ptr<<<blocks, threads, smem, ctx.stream()>>>(
        (const char    *)Q->data,
        (const uint8_t *)K_p->data,
        (const char    *)V->data,
        mask ? (const char *)mask->data : nullptr,
        kernel_dst,
        kernel_dst_meta,
        (const float *)scales->data,
        (const float *)codebook->data,
        scale, logit_softcap, bits, firstCell, nCells, nKVHeads, packedBytes,
        (int32_t)Q->ne[0],
        ne01,
        (int32_t)Q->ne[2],
        (int32_t)Q->ne[3],
        (int32_t)Q->nb[1],
        (int32_t)Q->nb[2],
        (int64_t)Q->nb[3],
        (int32_t)V->nb[1],
        (int32_t)V->nb[2],
        (int64_t)V->nb[3],
        mask ? (int32_t)mask->ne[0] : 0,
        mask ? (int32_t)mask->nb[1] : 0,
        v_scales_ptr, v_codebook_ptr, v_bits, v_packedBytes,
        zeros_ptr, asymmetric,
        outlier_packed_ptr, outlier_scales_ptr, outlier_indices_ptr, outlier_zeros_ptr,
        outlier_bits, outlier_count, outlier_packed_bytes,
        locs_ptr
    );
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks > 1) {
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine((unsigned)nTokensQ, (unsigned)nHeadsQ, (unsigned)nSeq);
        const size_t nbytes_shared_combine = (size_t)parallel_blocks * sizeof(float2);
        flash_attn_combine_results<D>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, ctx.stream()>>>(
                dst_tmp.ptr, dst_tmp_meta.ptr, (float *)dst->data, parallel_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}
