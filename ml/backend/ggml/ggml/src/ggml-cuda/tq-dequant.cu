#include "tq-dequant.cuh"
#include <cuda_fp16.h>

// Optimized TQ dequant kernel: warp-shuffle codebook + hardcoded bit extraction.
//
// Grid: (nCells, numKVHeads).  Block: 128 threads.
// For D=128 and 128 threads: each thread decodes exactly 1 element.
// Codebook lookup via __shfl_sync eliminates global memory reads.
// Output is written as f16.
//
// This kernel is the "separate dequant" path — paired with the stock f16
// flash attention kernel, it avoids injecting decode ALU into the
// bandwidth-bound FA loop.

#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
__global__ void tq_dequant_multihead_kernel(
    const uint8_t *packed,    // [(firstCell+c)*numKVHeads+h]*packed_bytes
    const float   *scales,    // [(firstCell+c)*numKVHeads+h]
    const float   *codebook,  // [codebook_len]
    uint16_t      *output,    // [nCells * numKVHeads * headDim] f16
    int            headDim,
    int            numKVHeads,
    int            bits,
    int            packed_bytes,
    int            codebook_len,
    int            firstCell
) {
    int c    = blockIdx.x;  // cell index within [0, nCells)
    int h    = blockIdx.y;  // head index within [0, numKVHeads)
    int cell = firstCell + c;

    int slot = cell * numKVHeads + h;
    float scale = scales[slot];
    const uint8_t *cell_packed = packed + (size_t)slot * packed_bytes;
    __half *cell_out = (__half *)(output + ((size_t)c * numKVHeads + h) * headDim);

    // Load one codebook entry per lane for warp-shuffle lookup.
    // For 3-bit (8 entries): lanes 0-7 hold codebook[0-7], repeated every 8 lanes.
    // For 2-bit (4 entries): lanes 0-3 hold codebook[0-3], repeated.
    const int cb_mask = (1 << bits) - 1;
    const float cb_lane = codebook[threadIdx.x & cb_mask];

    for (int elem = threadIdx.x; elem < headDim; elem += blockDim.x) {
        // Generic bit extraction (handles any alignment).
        const int bit_offset = elem * bits;
        const int byte_idx   = bit_offset >> 3;
        const int shift      = bit_offset & 7;

        int idx = (cell_packed[byte_idx] >> shift) & cb_mask;
        if (shift + bits > 8) {
            idx |= (cell_packed[byte_idx + 1] << (8 - shift)) & cb_mask;
        }

        // Codebook lookup via warp shuffle: zero global memory latency.
        // Width = warpSize (32) works because cb_lane is periodic with period (1<<bits).
        float val = __shfl_sync(0xFFFFFFFF, cb_lane, idx) * scale;
        cell_out[elem] = __float2half_rn(val);
    }
}
#else
// Stub for sm < 600 (no __shfl_sync).  Never executed — the launcher
// asserts compute capability >= 6.0.  Only exists so the kernel launch
// in ggml_cuda_tq_dequant compiles for Maxwell targets.
__global__ void tq_dequant_multihead_kernel(
    const uint8_t *, const float *, const float *, uint16_t *,
    int, int, int, int, int, int) {}
#endif

// ── outlier-split dequant ──────────────────────────────────────────────────
// Paired with tq_encode_kernel_outlier. Reads the regular packed sub-block and
// the outlier packed sub-block from two separate per-layer tensors, decodes
// each with its own codebook/scale, and writes a single [headDim] f16 K
// vector per (cell, head) to the output tensor.
//
// The outlier_indices tensor holds the top-K channel positions the encoder
// selected. For each output position elem, the kernel scans outlier_indices
// to determine whether elem is an outlier (and if so, which outlier slot k)
// or a regular channel (and if so, its contiguous regular slot r = elem minus
// the number of outlier indices less than elem). This scan is O(outlier_count)
// per thread, which is cheap: at outlier_count=32 it's 32 compares per thread,
// all in registers.
#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
__global__ void tq_dequant_multihead_kernel_outlier(
    const uint8_t *reg_packed,
    const float   *reg_scales,
    const float   *reg_codebook,
    const uint8_t *out_packed,
    const float   *out_scales,
    const uint8_t *out_indices,
    const float   *out_codebook,
    uint16_t      *output,
    int headDim,
    int numKVHeads,
    int bits,
    int reg_packed_bytes,
    int outlier_bits,
    int outlier_count,
    int out_packed_bytes,
    int firstCell
) {
    int c    = blockIdx.x;
    int h    = blockIdx.y;
    int cell = firstCell + c;
    int slot = cell * numKVHeads + h;

    float regScale = reg_scales[slot];
    float outScale = out_scales[slot];
    const uint8_t *cell_reg  = reg_packed  + (size_t)slot * reg_packed_bytes;
    const uint8_t *cell_outl = out_packed  + (size_t)slot * out_packed_bytes;
    const uint8_t *cell_idx  = out_indices + (size_t)slot * outlier_count;
    __half *cell_out = (__half *)(output + ((size_t)c * numKVHeads + h) * headDim);

    // Shared memory layout:
    //   s_outl_slot[headDim]           int8_t — outlier slot k or -1 if not outlier
    //   s_mask[headDim / 32 (min 4)]  uint32  — outlier bitmap (bit i = channel i outlier)
    //
    // Per-element `regular_slot` is computed at decode time via popcount over the
    // mask bits below `elem` — O(1) for headDim <= 256 (up to 8 popc ops). This
    // replaces the per-element O(outlier_count) classification scan.
    // Setup is fully parallel: no serial prefix sum, no idle threads.
    const int mask_words = (headDim + 31) >> 5;  // e.g. 4 for headDim=128, 8 for 256
    extern __shared__ char s_mem_dq[];
    int8_t   *s_outl_slot = (int8_t   *)s_mem_dq;
    uint32_t *s_mask      = (uint32_t *)(s_outl_slot + headDim);

    // Step A: init s_outl_slot to -1 and s_mask to 0 (parallel over all threads).
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        s_outl_slot[i] = -1;
    }
    for (int w = threadIdx.x; w < mask_words; w += blockDim.x) {
        s_mask[w] = 0u;
    }
    __syncthreads();

    // Step B: threads 0..outlier_count-1 each register one outlier — write its
    // slot into s_outl_slot AND set its bit in s_mask. atomicOr needed because
    // two outliers can land in the same word.
    if ((int)threadIdx.x < outlier_count) {
        int pos = (int)cell_idx[threadIdx.x];
        s_outl_slot[pos] = (int8_t)threadIdx.x;
        atomicOr(&s_mask[pos >> 5], 1u << (pos & 31));
    }
    __syncthreads();

    // Warp-shuffle register-resident codebooks.
    const int cb_mask  = (1 << bits) - 1;
    const int ocb_mask = (1 << outlier_bits) - 1;
    const float cb_lane_reg = reg_codebook[threadIdx.x & cb_mask];
    const float cb_lane_out = out_codebook[threadIdx.x & ocb_mask];

    for (int elem = threadIdx.x; elem < headDim; elem += blockDim.x) {
        int outlier_slot = (int)s_outl_slot[elem];

        // regular_slot = elem - popcount(outlier_mask bits below elem).
        // Sum popcount of fully-covered 32-bit chunks, then partial chunk.
        int outliers_below = 0;
        const int full_words = elem >> 5;
        #pragma unroll
        for (int w = 0; w < 8; w++) {   // hard-unrolled; loop body is a no-op past mask_words
            if (w < full_words && w < mask_words) {
                outliers_below += __popc(s_mask[w]);
            }
        }
        if (full_words < mask_words) {
            uint32_t partial_bits = (1u << (elem & 31)) - 1u;
            outliers_below += __popc(s_mask[full_words] & partial_bits);
        }
        int regular_slot = elem - outliers_below;

        // Both sub-block decodes must execute unconditionally because
        // __shfl_sync with mask 0xFFFFFFFF requires every lane in the warp
        // to be at the same instruction. Putting a shuffle inside a divergent
        // if-branch is undefined behavior (observed as all-zero decodes for
        // some (cell, head) blocks on multi-head models). Compute both values
        // up front, then select.
        int reg_bit_offset = regular_slot * bits;
        int reg_byte_idx   = reg_bit_offset >> 3;
        int reg_shift      = reg_bit_offset & 7;
        int reg_idx = (cell_reg[reg_byte_idx] >> reg_shift) & cb_mask;
        if (reg_shift + bits > 8) {
            reg_idx |= (cell_reg[reg_byte_idx + 1] << (8 - reg_shift)) & cb_mask;
        }
        float reg_val = __shfl_sync(0xFFFFFFFF, cb_lane_reg, reg_idx) * regScale;

        int out_slot_safe = (outlier_slot >= 0) ? outlier_slot : 0;
        int out_bit_offset = out_slot_safe * outlier_bits;
        int out_byte_idx   = out_bit_offset >> 3;
        int out_shift      = out_bit_offset & 7;
        int out_idx = (cell_outl[out_byte_idx] >> out_shift) & ocb_mask;
        if (out_shift + outlier_bits > 8) {
            out_idx |= (cell_outl[out_byte_idx + 1] << (8 - out_shift)) & ocb_mask;
        }
        float out_val = __shfl_sync(0xFFFFFFFF, cb_lane_out, out_idx) * outScale;

        float val = (outlier_slot >= 0) ? out_val : reg_val;
        cell_out[elem] = __float2half_rn(val);
    }
}
#else
__global__ void tq_dequant_multihead_kernel_outlier(
    const uint8_t *, const float *, const float *,
    const uint8_t *, const float *, const uint8_t *, const float *,
    uint16_t *, int, int, int, int, int, int, int, int) {}
#endif

void ggml_cuda_tq_dequant(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc >= 600 &&
                "TurboQuant dequant requires compute capability 6.0+ (Pascal or newer)");
    const struct ggml_tensor * encode_result = dst->src[0]; // view of packed
    const struct ggml_tensor * scales        = dst->src[1];
    const struct ggml_tensor * codebook      = dst->src[2];

    const int headDim      = (int)dst->ne[0];
    const int numKVHeads   = (int)dst->ne[1];
    const int nCells       = (int)dst->ne[2];
    const int bits         = (int)((const int32_t *)dst->op_params)[0];
    const int firstCell    = (int)((const int32_t *)dst->op_params)[1];
    const int outlierBits  = (int)((const int32_t *)dst->op_params)[2];
    const int outlierCount = (int)((const int32_t *)dst->op_params)[3];

    dim3 grid(nCells, numKVHeads);
    int block_size = 128;
    if (headDim < block_size) block_size = headDim;

    if (outlierCount > 0 && outlierBits > 0 && outlierCount < headDim) {
        // Outlier-split dequant: read both sub-blocks.
        const struct ggml_tensor * outlier_packed   = dst->src[3];
        const struct ggml_tensor * outlier_scales   = dst->src[4];
        const struct ggml_tensor * outlier_indices  = dst->src[5];
        const struct ggml_tensor * outlier_codebook = dst->src[6];

        const int regular_count        = headDim - outlierCount;
        // Per-head stride for both packed tensors is padded up to a 4-byte
        // multiple so atomicOr-on-word stays aligned in the encode kernel.
        // The Go-side ggmlTQCompressedK.regularPackedBytes() applies the
        // same padding, so the kernel-visible layout matches the allocator.
        const int reg_packed_raw       = (regular_count * bits + 7) / 8;
        const int reg_packed_bytes     = (reg_packed_raw + 3) & ~3;
        const int out_packed_raw       = (outlierCount * outlierBits + 7) / 8;
        const int out_packed_bytes     = (out_packed_raw + 3) & ~3;

        // Shared memory: s_outl_slot (headDim * int8) + s_mask (ceil(headDim/32) * u32).
        const int mask_words = (headDim + 31) >> 5;
        size_t smem = (size_t)headDim * sizeof(int8_t)
                    + (size_t)mask_words * sizeof(uint32_t);

        tq_dequant_multihead_kernel_outlier<<<grid, block_size, smem, ctx.stream()>>>(
            (const uint8_t *)encode_result->data,
            (const float   *)scales->data,
            (const float   *)codebook->data,
            (const uint8_t *)outlier_packed->data,
            (const float   *)outlier_scales->data,
            (const uint8_t *)outlier_indices->data,
            (const float   *)outlier_codebook->data,
            (uint16_t      *)dst->data,
            headDim, numKVHeads, bits, reg_packed_bytes,
            outlierBits, outlierCount, out_packed_bytes, firstCell
        );
        return;
    }

    const int packed_bytes  = (headDim * bits + 7) / 8;
    const int codebook_len  = (int)codebook->ne[0];

    tq_dequant_multihead_kernel<<<grid, block_size, 0, ctx.stream()>>>(
        (const uint8_t *)encode_result->data,
        (const float   *)scales->data,
        (const float   *)codebook->data,
        (uint16_t      *)dst->data,
        headDim, numKVHeads, bits, packed_bytes, codebook_len, firstCell
    );
}

// V dequant with fused rotation undo: decode packed V to shared memory
// (rotated domain), then multiply by R [headDim × headDim] to produce
// unrotated f16 output.  Eliminates the per-layer mulmat op in SDPA.
//
// Grid: (nCells, numKVHeads).  Block: 128 (= headDim for D=128).
// Shared memory: headDim floats = 512 bytes.
#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
__global__ void tq_dequant_v_rotated_kernel(
    const uint8_t * __restrict__ packed,
    const float   * __restrict__ scales,
    const float   * __restrict__ codebook,
    const float   * __restrict__ rotation,  // R [headDim, headDim] row-major
    uint16_t      * __restrict__ output,
    int headDim, int numKVHeads, int bits, int packed_bytes,
    int codebook_len, int firstCell)
{
    extern __shared__ float s_rotV[];  // headDim floats

    int c    = blockIdx.x;
    int h    = blockIdx.y;
    int cell = firstCell + c;
    int slot = cell * numKVHeads + h;
    float scale = scales[slot];
    const uint8_t *cell_packed = packed + (size_t)slot * packed_bytes;

    // Phase 1: decode one element per thread into shared memory (rotated domain).
    const int cb_mask = (1 << bits) - 1;
    const float cb_lane = codebook[threadIdx.x & cb_mask];

    int elem = threadIdx.x;
    int bit_offset = elem * bits;
    int byte_idx   = bit_offset >> 3;
    int shift      = bit_offset & 7;
    int idx = (cell_packed[byte_idx] >> shift) & cb_mask;
    if (shift + bits > 8) {
        idx |= (cell_packed[byte_idx + 1] << (8 - shift)) & cb_mask;
    }
    s_rotV[elem] = __shfl_sync(0xFFFFFFFF, cb_lane, idx) * scale;
    __syncthreads();

    // Phase 2: each thread computes one output element = dot(R[elem,:], s_rotV).
    // R is in L2 (64KB, fits in P40's 3MB L2; read-only, broadcast across blocks).
    const float *R_row = rotation + elem * headDim;
    float sum = 0.0f;
    for (int j = 0; j < headDim; j++) {
        sum += R_row[j] * s_rotV[j];
    }

    __half *cell_out = (__half *)(output + ((size_t)c * numKVHeads + h) * headDim);
    cell_out[elem] = __float2half_rn(sum);
}
#else
// Stub for sm < 600 (no __shfl_sync).  Not currently launched, but kept
// so future call sites compile for Maxwell targets without special casing.
__global__ void tq_dequant_v_rotated_kernel(
    const uint8_t * __restrict__, const float * __restrict__,
    const float   * __restrict__, const float * __restrict__,
    uint16_t      * __restrict__,
    int, int, int, int, int, int) {}
#endif

// Combined K+V dequant: two back-to-back kernel launches in a single GGML op.
// Output: [headDim, numKVHeads, nCells, 2] f16 — K at ne[3]=0, V at ne[3]=1.
// When src[6] (v_rotation) is non-NULL, V is dequanted with rotation fused.
void ggml_cuda_tq_dequant_kv(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc >= 600 &&
                "TurboQuant dequant requires compute capability 6.0+ (Pascal or newer)");
    const struct ggml_tensor * k_encode   = dst->src[0];
    const struct ggml_tensor * k_scales   = dst->src[1];
    const struct ggml_tensor * k_cb       = dst->src[2];
    const struct ggml_tensor * v_encode   = dst->src[3];
    const struct ggml_tensor * v_scales   = dst->src[4];
    const struct ggml_tensor * v_cb       = dst->src[5];
    const struct ggml_tensor * v_rotation = dst->src[6];  // NULL = no rotation fusion

    const int headDim    = (int)dst->ne[0];
    const int numKVHeads = (int)dst->ne[1];
    const int nCells     = (int)dst->ne[2];

    int32_t k_bits, v_bits, firstCell;
    memcpy(&k_bits,    (const int32_t *)dst->op_params + 0, sizeof(int32_t));
    memcpy(&v_bits,    (const int32_t *)dst->op_params + 1, sizeof(int32_t));
    memcpy(&firstCell, (const int32_t *)dst->op_params + 2, sizeof(int32_t));

    const int k_packed_bytes = (headDim * k_bits + 7) / 8;
    const int v_packed_bytes = (headDim * v_bits + 7) / 8;

    dim3 grid(nCells, numKVHeads);
    int block_size = 128;
    if (headDim < block_size) {
        block_size = headDim;
    }

    const size_t plane_size = (size_t)headDim * numKVHeads * nCells;
    uint16_t * out_base = (uint16_t *)dst->data;

    cudaStream_t stream = ctx.stream();

    // K dequant → first plane (offset 0) — always unrotated
    tq_dequant_multihead_kernel<<<grid, block_size, 0, stream>>>(
        (const uint8_t *)k_encode->data,
        (const float   *)k_scales->data,
        (const float   *)k_cb->data,
        out_base,
        headDim, numKVHeads, k_bits, k_packed_bytes, (int)k_cb->ne[0], firstCell
    );

    // V dequant → second plane (offset plane_size). Plain dequant only — the
    // rotation undo (R @ attn_out) is handled by SDPA via mulmat, which is
    // dramatically faster than the per-cell matmul the fused kernel did.
    (void)v_rotation;
    tq_dequant_multihead_kernel<<<grid, block_size, 0, stream>>>(
        (const uint8_t *)v_encode->data,
        (const float   *)v_scales->data,
        (const float   *)v_cb->data,
        out_base + plane_size,
        headDim, numKVHeads, v_bits, v_packed_bytes, (int)v_cb->ne[0], firstCell
    );
}
