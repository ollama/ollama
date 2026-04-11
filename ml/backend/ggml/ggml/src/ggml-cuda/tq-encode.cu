#include "tq-encode.cuh"
#include <cuda_fp16.h>
#include <math.h>

#define TQ_ENCODE_BLOCK_SIZE 128

// ── kernel ──────────────────────────────────────────────────────────────────

__global__ void tq_encode_kernel(
    const void      *k,            // f16 or f32, ggml layout [headDim, numKVHeads, batchSize]
    const float     *rotation,     // [headDim, headDim] R^T stored row-major
    uint8_t         *packed_out,   // [(c*numKVHeads+h)*packedBytes] interleaved
    float           *scales_out,   // [c*numKVHeads+h] interleaved
    int              firstCell,    // first cache cell index (cell = firstCell + batch)
    const float     *boundaries,   // [numLevels-1]
    int              headDim,
    int              numKVHeads,
    int              bits,
    int              numBoundaries, // = (1<<bits) - 1
    int              kIsF32        // non-zero when k is float32 (vs float16)
) {
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int cell  = firstCell + batch;

    // Shared memory layout:
    //   s_k[headDim]       – K values as f32
    //   s_rot[headDim]     – rotated values
    //   s_reduce[BLOCK]    – warp reduction scratch
    //   s_idx[headDim]     – quantized indices (uint8)
    extern __shared__ char s_mem[];
    float   *s_k      = (float *)s_mem;
    float   *s_rot    = s_k + headDim;
    float   *s_reduce = s_rot + headDim;
    uint8_t *s_idx    = (uint8_t *)(s_reduce + blockDim.x);

    // ── Step 1: Load K[batch, head] into shared memory as f32 ────────────────
    // K layout: element (dim=d, head=h, batch=b) at b*numKVHeads*headDim + h*headDim + d
    int base_k = batch * numKVHeads * headDim + head * headDim;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        if (kIsF32) {
            s_k[d] = ((const float *)k)[base_k + d];
        } else {
            s_k[d] = __half2float(__ushort_as_half(((const uint16_t *)k)[base_k + d]));
        }
    }
    __syncthreads();

    // ── Step 2: Rotation matmul: rotated[i] = Σ_j rotation[i*headDim+j] * s_k[j] ──
    // rotation stores Q^T row-major (rotTensor[i][j] = Q^T[i][j]).
    // This computes rotated = Q^T @ k.
    // Q is also rotated as Q^T @ q (via ggml_mul_mat(rotTensor, q)),
    // so attention = (Q^T q)^T (Q^T k) = q^T Q Q^T k = q^T k.
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < headDim; j++) {
            sum += rotation[i * headDim + j] * s_k[j];
        }
        s_rot[i] = sum;
    }
    __syncthreads();

    // ── Step 3: RMS scale = sqrt(mean(rotated²)) ─────────────────────────────
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        local_sq += s_rot[i] * s_rot[i];
    }
    s_reduce[threadIdx.x] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        __syncthreads();
    }

    float scale = 0.0f;
    if (threadIdx.x == 0) {
        float sum_sq = s_reduce[0];
        if (sum_sq > 1e-12f)
            scale = sqrtf(sum_sq / (float)headDim);
        scales_out[cell * numKVHeads + head] = scale;
        s_reduce[0] = scale;  // broadcast via shared mem
    }
    __syncthreads();
    scale = s_reduce[0];

    // ── Step 4: Quantize each element via boundary binary search ─────────────
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float v = (scale > 0.0f) ? (s_rot[i] / scale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numBoundaries; b++) {
            if (v >= boundaries[b]) idx++;
        }
        s_idx[i] = (uint8_t)idx;
    }
    __syncthreads();

    // ── Step 5: Pack bits LSB-first into output (parallel via atomicOr) ─────
    {
        int packed_bytes = (headDim * bits + 7) / 8;
        uint8_t *out = packed_out + (cell * numKVHeads + head) * packed_bytes;
        uint8_t bitmask = (uint8_t)((1 << bits) - 1);

        // Zero output buffer in parallel.
        for (int p = threadIdx.x; p < packed_bytes; p += blockDim.x) {
            out[p] = 0;
        }
        __syncthreads();

        // Each thread packs its own elements using atomicOr on 4-byte aligned words.
        for (int elem = threadIdx.x; elem < headDim; elem += blockDim.x) {
            int bit_offset = elem * bits;
            int byte_idx   = bit_offset >> 3;
            int shift      = bit_offset & 7;
            uint8_t v = s_idx[elem] & bitmask;
            // Pack into 4-byte aligned word: position byte within 32-bit word.
            atomicOr((unsigned int *)(out + (byte_idx & ~3)),
                     (unsigned int)(v << shift) << ((byte_idx & 3) * 8));
            if (shift + bits > 8) {
                int byte_idx2 = byte_idx + 1;
                atomicOr((unsigned int *)(out + (byte_idx2 & ~3)),
                         (unsigned int)(v >> (8 - shift)) << ((byte_idx2 & 3) * 8));
            }
        }
    }
}

// ── outlier-split kernel ─────────────────────────────────────────────────────
//
// Implements the TurboQuant paper's actual experimental configuration
// (arXiv 2504.19874 Sec 4.3): split channels into a top-K outlier set and a
// regular set, each encoded with its own RMS scale and codebook at different
// bit widths. Outlier selection happens in ROTATED space (single rotation
// matmul shared between both sub-blocks), not in the original space like the
// CPU reference — this is the only paper-realistic algorithm we can run
// cheaply on the GPU without a second rotation pass. For the near-orthogonal
// rotations produced by QR on a Gaussian matrix, the top-K in rotated space
// is a close proxy for the top-K in original space.

__global__ void tq_encode_kernel_outlier(
    const void      *k,                  // f16 or f32, [headDim, numKVHeads, batchSize]
    const float     *rotation,           // [headDim, headDim] R^T row-major
    uint8_t         *packed_out,         // regular packed [regularPackedBytes*numKVHeads*cells]
    float           *scales_out,         // regular scales [numKVHeads*cells]
    uint8_t         *outlier_packed,     // outlier packed [outlierPackedBytes*numKVHeads*cells]
    float           *outlier_scales,     // outlier scales [numKVHeads*cells]
    uint8_t         *outlier_indices,    // outlier channel idx [outlierCount*numKVHeads*cells] (interpreted as uint8 so positions 0..255 fit)
    int              firstCell,
    const float     *boundaries,         // regular boundaries [(1<<bits)-1]
    const float     *outlier_boundaries, // outlier boundaries [(1<<outlierBits)-1]
    int              headDim,
    int              numKVHeads,
    int              bits,
    int              numBoundaries,
    int              outlierBits,
    int              outlierCount,
    int              numOutlierBoundaries,
    int              kIsF32
) {
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int cell  = firstCell + batch;

    // Shared memory layout (laid out contiguously — launch passes total size):
    //   s_k[headDim]              - f32 K values
    //   s_rot[headDim]             - f32 rotated values
    //   s_reduce[blockDim.x]       - reduction scratch (float)
    //   s_idx[headDim]             - regular quantized indices (uint8)
    //   s_is_outlier[headDim]      - 1 = outlier, 0 = regular (uint8)
    //   s_reg_pos[headDim]         - regular position index map (int, up to headDim entries)
    //   s_outl_pos[outlierCount]   - outlier channel positions (int)
    //   s_outl_val[outlierCount]   - outlier rotated values (float)
    //   s_outl_idx[outlierCount]   - outlier quantized indices (uint8)

    extern __shared__ char s_mem[];
    float   *s_k          = (float *)s_mem;
    float   *s_rot        = s_k + headDim;
    float   *s_reduce     = s_rot + headDim;
    uint8_t *s_idx        = (uint8_t *)(s_reduce + blockDim.x);
    uint8_t *s_is_outlier = s_idx + headDim;
    int     *s_reg_pos    = (int *)(s_is_outlier + headDim);
    int     *s_outl_pos   = s_reg_pos + headDim;
    float   *s_outl_val   = (float *)(s_outl_pos + outlierCount);
    uint8_t *s_outl_idx   = (uint8_t *)(s_outl_val + outlierCount);

    // Step 1: Load K into s_k as f32.
    int base_k = batch * numKVHeads * headDim + head * headDim;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        if (kIsF32) {
            s_k[d] = ((const float *)k)[base_k + d];
        } else {
            s_k[d] = __half2float(__ushort_as_half(((const uint16_t *)k)[base_k + d]));
        }
    }
    __syncthreads();

    // Step 2: Rotate: s_rot[i] = sum_j rotation[i*headDim+j] * s_k[j] = (R^T @ k)[i].
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < headDim; j++) {
            sum += rotation[i * headDim + j] * s_k[j];
        }
        s_rot[i] = sum;
    }
    __syncthreads();

    // Step 3: Clear outlier mask.
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        s_is_outlier[i] = 0;
    }
    __syncthreads();

    // Step 4: Top-K outlier selection (serial on thread 0). O(K * headDim)
    // comparisons. At outlierCount=32, headDim=128 that's ~4k ops on one
    // thread — negligible next to the rotation matmul (16k ops) that ran
    // in parallel above. Also builds s_reg_pos as the ordered list of
    // regular (non-outlier) channel positions, and s_outl_pos for outliers.
    if (threadIdx.x == 0) {
        for (int r = 0; r < outlierCount; r++) {
            float best_val = -1.0f;
            int   best_idx = 0;
            for (int i = 0; i < headDim; i++) {
                if (s_is_outlier[i]) continue;
                float a = fabsf(s_rot[i]);
                if (a > best_val) {
                    best_val = a;
                    best_idx = i;
                }
            }
            s_is_outlier[best_idx] = 1;
            s_outl_pos[r]          = best_idx;
            s_outl_val[r]          = s_rot[best_idx];
        }
        // Build regular position map after outlier selection is complete.
        int pos = 0;
        for (int i = 0; i < headDim; i++) {
            if (!s_is_outlier[i]) {
                s_reg_pos[pos++] = i;
            }
        }
    }
    __syncthreads();

    int regularCount = headDim - outlierCount;

    // Step 5: Per-sub-block RMS scales via parallel reduction.
    // Each thread accumulates its dims' squared contributions into two
    // locals; we reduce regular and outlier sums back to back using
    // s_reduce as scratch.
    float local_sq_reg = 0.0f;
    float local_sq_out = 0.0f;
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float v = s_rot[i];
        float sq = v * v;
        if (s_is_outlier[i]) {
            local_sq_out += sq;
        } else {
            local_sq_reg += sq;
        }
    }
    s_reduce[threadIdx.x] = local_sq_reg;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        __syncthreads();
    }
    float regScale = 0.0f;
    if (threadIdx.x == 0) {
        float sum_sq = s_reduce[0];
        if (sum_sq > 1e-12f && regularCount > 0) {
            regScale = sqrtf(sum_sq / (float)regularCount);
        }
        scales_out[cell * numKVHeads + head] = regScale;
        s_reduce[0] = regScale;
    }
    __syncthreads();
    regScale = s_reduce[0];
    __syncthreads();

    s_reduce[threadIdx.x] = local_sq_out;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        __syncthreads();
    }
    float outScale = 0.0f;
    if (threadIdx.x == 0) {
        float sum_sq = s_reduce[0];
        if (sum_sq > 1e-12f && outlierCount > 0) {
            outScale = sqrtf(sum_sq / (float)outlierCount);
        }
        outlier_scales[cell * numKVHeads + head] = outScale;
        s_reduce[0] = outScale;
    }
    __syncthreads();
    outScale = s_reduce[0];

    // Step 6: Quantize regular channels. s_idx[r] stores the code at the
    // CONTIGUOUS regular position r, not the original channel index, so the
    // dequant kernel can read them directly into the packed bit stream.
    for (int r = threadIdx.x; r < regularCount; r += blockDim.x) {
        int orig = s_reg_pos[r];
        float v = (regScale > 0.0f) ? (s_rot[orig] / regScale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numBoundaries; b++) {
            if (v >= boundaries[b]) idx++;
        }
        s_idx[r] = (uint8_t)idx;
    }
    __syncthreads();

    // Step 7: Pack regular bits into packed_out.
    // Per-head stride is padded up to a 4-byte multiple so atomicOr on
    // unsigned-int words stays aligned for every head, regardless of
    // how many regular channels (bit count) are in play. The Go-side
    // allocator uses the same padded value (regularPackedBytes()).
    const int regular_packed_bytes_raw    = (regularCount * bits + 7) / 8;
    const int regular_packed_bytes        = (regular_packed_bytes_raw + 3) & ~3;
    uint8_t *reg_out = packed_out + (cell * numKVHeads + head) * regular_packed_bytes;
    for (int p = threadIdx.x; p < regular_packed_bytes; p += blockDim.x) {
        reg_out[p] = 0;
    }
    __syncthreads();
    {
        uint8_t bitmask = (uint8_t)((1 << bits) - 1);
        for (int r = threadIdx.x; r < regularCount; r += blockDim.x) {
            int bit_offset = r * bits;
            int byte_idx   = bit_offset >> 3;
            int shift      = bit_offset & 7;
            uint8_t v = s_idx[r] & bitmask;
            atomicOr((unsigned int *)(reg_out + (byte_idx & ~3)),
                     (unsigned int)(v << shift) << ((byte_idx & 3) * 8));
            if (shift + bits > 8) {
                int byte_idx2 = byte_idx + 1;
                atomicOr((unsigned int *)(reg_out + (byte_idx2 & ~3)),
                         (unsigned int)(v >> (8 - shift)) << ((byte_idx2 & 3) * 8));
            }
        }
    }
    __syncthreads();

    // Step 8: Quantize outlier channels with their own codebook.
    for (int r = threadIdx.x; r < outlierCount; r += blockDim.x) {
        float v = (outScale > 0.0f) ? (s_outl_val[r] / outScale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numOutlierBoundaries; b++) {
            if (v >= outlier_boundaries[b]) idx++;
        }
        s_outl_idx[r] = (uint8_t)idx;
    }
    __syncthreads();

    // Step 9: Pack outlier bits. Same 4-byte alignment as regular packing.
    const int outlier_packed_bytes_raw = (outlierCount * outlierBits + 7) / 8;
    const int outlier_packed_bytes     = (outlier_packed_bytes_raw + 3) & ~3;
    uint8_t *out_out = outlier_packed + (cell * numKVHeads + head) * outlier_packed_bytes;
    for (int p = threadIdx.x; p < outlier_packed_bytes; p += blockDim.x) {
        out_out[p] = 0;
    }
    __syncthreads();
    {
        uint8_t obmask = (uint8_t)((1 << outlierBits) - 1);
        for (int r = threadIdx.x; r < outlierCount; r += blockDim.x) {
            int bit_offset = r * outlierBits;
            int byte_idx   = bit_offset >> 3;
            int shift      = bit_offset & 7;
            uint8_t v = s_outl_idx[r] & obmask;
            atomicOr((unsigned int *)(out_out + (byte_idx & ~3)),
                     (unsigned int)(v << shift) << ((byte_idx & 3) * 8));
            if (shift + outlierBits > 8) {
                int byte_idx2 = byte_idx + 1;
                atomicOr((unsigned int *)(out_out + (byte_idx2 & ~3)),
                         (unsigned int)(v >> (8 - shift)) << ((byte_idx2 & 3) * 8));
            }
        }
    }

    // Step 10: Write outlier channel indices. uint8_t covers 0..255 safely.
    uint8_t *idx_out = outlier_indices + (cell * numKVHeads + head) * outlierCount;
    for (int r = threadIdx.x; r < outlierCount; r += blockDim.x) {
        idx_out[r] = (uint8_t)s_outl_pos[r];
    }
}

// ── ggml dispatch ─────────────────────────────────────────────────────────────

// Extern declaration for the V encode kernel (defined in tq-encode-v.cu).
extern __global__ void tq_encode_v_kernel(
    const void *v, const float *rotation, uint8_t *packed_out, float *scales_out,
    int firstCell, const float *boundaries,
    int headDim, int numKVHeads, int bits, int numBoundaries, int vIsF32);

void ggml_cuda_tq_encode(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * k          = dst->src[0];
    const struct ggml_tensor * rotation   = dst->src[1];
    // src[2] unused (was cell_idx; now firstCell is in op_params[1])
    const struct ggml_tensor * scales     = dst->src[3];
    const struct ggml_tensor * boundaries = dst->src[4];

    const int headDim    = (int)k->ne[0];
    const int numKVHeads = (int)k->ne[1];
    const int batchSize  = (int)k->ne[2];
    const int bits         = (int)((const int32_t *)dst->op_params)[0];
    const int firstCell    = (int)((const int32_t *)dst->op_params)[1];
    const int outlierBits  = (int)((const int32_t *)dst->op_params)[2];
    const int outlierCount = (int)((const int32_t *)dst->op_params)[3];
    const int numBoundaries = (1 << bits) - 1;
    const int kIsF32     = (k->type == GGML_TYPE_F32) ? 1 : 0;

    dim3 grid(batchSize, numKVHeads);
    int block_size = (headDim < TQ_ENCODE_BLOCK_SIZE) ? headDim : TQ_ENCODE_BLOCK_SIZE;
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;

    cudaStream_t stream = ctx.stream();

    if (outlierCount > 0 && outlierBits > 0 && outlierCount < headDim) {
        const struct ggml_tensor * outlier_packed     = dst->src[5];
        const struct ggml_tensor * outlier_scales     = dst->src[6];
        const struct ggml_tensor * outlier_indices    = dst->src[7];
        const struct ggml_tensor * outlier_boundaries = dst->src[8];
        const int numOutlierBoundaries = (1 << outlierBits) - 1;

        // Shared memory layout for outlier kernel:
        //   s_k[headDim]              f32
        //   s_rot[headDim]             f32
        //   s_reduce[block_size]       f32
        //   s_idx[headDim]             u8
        //   s_is_outlier[headDim]      u8
        //   s_reg_pos[headDim]         i32 (over-sized to headDim; only regularCount used)
        //   s_outl_pos[outlierCount]   i32
        //   s_outl_val[outlierCount]   f32
        //   s_outl_idx[outlierCount]   u8
        size_t smem = (size_t)headDim * 2 * sizeof(float)               // s_k + s_rot
                    + (size_t)block_size * sizeof(float)                 // s_reduce
                    + (size_t)headDim * 2 * sizeof(uint8_t)              // s_idx + s_is_outlier
                    + (size_t)headDim * sizeof(int)                      // s_reg_pos
                    + (size_t)outlierCount * (sizeof(int) + sizeof(float) + sizeof(uint8_t));

        tq_encode_kernel_outlier<<<grid, block_size, smem, stream>>>(
            k->data,
            (const float *)rotation->data,
            (uint8_t     *)dst->data,
            (float       *)scales->data,
            (uint8_t     *)outlier_packed->data,
            (float       *)outlier_scales->data,
            (uint8_t     *)outlier_indices->data,
            firstCell,
            (const float *)boundaries->data,
            (const float *)outlier_boundaries->data,
            headDim, numKVHeads, bits, numBoundaries,
            outlierBits, outlierCount, numOutlierBoundaries, kIsF32
        );
        return;
    }

    size_t smem = (size_t)headDim * 2 * sizeof(float)
                + (size_t)block_size * sizeof(float)
                + (size_t)headDim * sizeof(uint8_t);

    tq_encode_kernel<<<grid, block_size, smem, stream>>>(
        k->data,
        (const float    *)rotation->data,
        (uint8_t        *)dst->data,
        (float          *)scales->data,
        firstCell,
        (const float    *)boundaries->data,
        headDim, numKVHeads, bits, numBoundaries, kIsF32
    );
}

// Combined K+V encode: two back-to-back kernel launches in a single GGML op.
void ggml_cuda_tq_encode_kv(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // src layout: [0]=K, [1]=rotation, [2]=V, [3]=K_scales, [4]=K_bounds,
    //             [5]=V_packed, [6]=V_scales, [7]=V_bounds
    const struct ggml_tensor * k          = dst->src[0];
    const struct ggml_tensor * rotation   = dst->src[1];
    const struct ggml_tensor * v          = dst->src[2];
    const struct ggml_tensor * k_scales   = dst->src[3];
    const struct ggml_tensor * k_bounds   = dst->src[4];
    const struct ggml_tensor * v_packed   = dst->src[5];
    const struct ggml_tensor * v_scales   = dst->src[6];
    const struct ggml_tensor * v_bounds   = dst->src[7];

    int32_t k_bits, v_bits, firstCell;
    memcpy(&k_bits,    (const int32_t *)dst->op_params + 0, sizeof(int32_t));
    memcpy(&v_bits,    (const int32_t *)dst->op_params + 1, sizeof(int32_t));
    memcpy(&firstCell, (const int32_t *)dst->op_params + 2, sizeof(int32_t));

    const int headDim    = (int)k->ne[0];
    const int numKVHeads = (int)k->ne[1];
    const int batchSize  = (int)k->ne[2];
    const int kIsF32     = (k->type == GGML_TYPE_F32) ? 1 : 0;
    const int vIsF32     = (v->type == GGML_TYPE_F32) ? 1 : 0;

    dim3 grid(batchSize, numKVHeads);
    int block_size = (headDim < TQ_ENCODE_BLOCK_SIZE) ? headDim : TQ_ENCODE_BLOCK_SIZE;
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;

    size_t smem = (size_t)headDim * 2 * sizeof(float)
                + (size_t)block_size * sizeof(float)
                + (size_t)headDim * sizeof(uint8_t);

    cudaStream_t stream = ctx.stream();

    // K encode
    tq_encode_kernel<<<grid, block_size, smem, stream>>>(
        k->data,
        (const float *)rotation->data,
        (uint8_t     *)dst->data,
        (float       *)k_scales->data,
        firstCell,
        (const float *)k_bounds->data,
        headDim, numKVHeads, k_bits, (1 << k_bits) - 1, kIsF32
    );

    // V encode
    const float * rotation_ptr = rotation ? (const float *)rotation->data : nullptr;
    tq_encode_v_kernel<<<grid, block_size, smem, stream>>>(
        v->data,
        rotation_ptr,
        (uint8_t     *)v_packed->data,
        (float       *)v_scales->data,
        firstCell,
        (const float *)v_bounds->data,
        headDim, numKVHeads, v_bits, (1 << v_bits) - 1, vIsF32
    );
}
