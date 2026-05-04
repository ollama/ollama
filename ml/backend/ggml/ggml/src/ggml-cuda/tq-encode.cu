#include "tq-encode.cuh"
#include <math.h>

#define TQ_ENCODE_BLOCK_SIZE 128

// ── kernel ──────────────────────────────────────────────────────────────────

// tq_encode_kernel — uniform Lloyd-Max N-bit key encoder.
// Template parameter kAsymmetric: when true, centres each rotated vector by
// its mean before RMS normalization; mean is stored in zeros_out for decode.
// Using a template (static dispatch) avoids an nvcc ICE on compute_70/75/87/90
// that occurs when the same conditional logic is expressed via a runtime pointer.
template<bool kAsymmetric>
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
    int              kIsF32,       // non-zero when k is float32 (vs float16)
    float           *zeros_out,    // [numKVHeads*cells] f32 mean (kAsymmetric only; ignored if !kAsymmetric)
    const float     *k_bias,       // [numKVHeads * headDim] f32, NULL = no bias subtraction
    const float     *codebook      // [1<<bits] f32 Lloyd-Max centroids; NULL = skip EDEN refinement
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
    // If k_bias is provided, subtract it before rotation so the Lloyd-Max
    // codebook sees zero-mean data even when the K projection has a bias term
    // (e.g. Qwen2). The bias term cancels in softmax attention scores so the
    // decoder does not need to add it back.
    int base_k = batch * numKVHeads * headDim + head * headDim;
    int bias_base = head * headDim;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float val;
        if (kIsF32) {
            val = ((const float *)k)[base_k + d];
        } else {
            val = __half2float(__ushort_as_half(((const uint16_t *)k)[base_k + d]));
        }
        if (k_bias != nullptr) {
            val -= k_bias[bias_base + d];
        }
        s_k[d] = val;
    }
    __syncthreads();

    // ── Step 2: Rotation matmul: rotated = R^T @ k ───────────────────────────
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < headDim; j++) {
            sum += rotation[i * headDim + j] * s_k[j];
        }
        s_rot[i] = sum;
    }
    __syncthreads();

    // ── Step 3: Mean (kAsymmetric only) then RMS scale ───────────────────────
    float regMean = 0.0f;
    if (kAsymmetric) {
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
            local_sum += s_rot[i];
        }
        s_reduce[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        // EXPERIMENT: all threads compute regMean directly from the
        // reduction sum to eliminate the write-then-read pattern.
        regMean = s_reduce[0] / (float)headDim;
        if (threadIdx.x == 0) {
            zeros_out[cell * numKVHeads + head] = regMean;
        }
        __syncthreads();
    }

    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float v = s_rot[i] - regMean;  // regMean==0 when !kAsymmetric
        local_sq += v * v;
    }
    s_reduce[threadIdx.x] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        __syncthreads();
    }

    // All threads compute scale from the reduction sum directly. Avoids the
    // write-then-read pattern (s_reduce[0] = scale; sync; scale = s_reduce[0])
    // which can race when the kernel runs the mean reduction immediately
    // before this RMS reduction (asymmetric path), producing non-deterministic
    // garbage on llama3.2:3b. See the mean broadcast above for context.
    float sum_sq = s_reduce[0];
    float scale = (sum_sq > 1e-12f) ? sqrtf(sum_sq / (float)headDim) : 0.0f;
    if (threadIdx.x == 0) {
        scales_out[cell * numKVHeads + head] = scale;
    }
    __syncthreads();

    // ── Step 4: Quantize each element via boundary binary search ─────────────
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float v = (scale > 0.0f) ? ((s_rot[i] - regMean) / scale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numBoundaries; b++) {
            if (v >= boundaries[b]) idx++;
        }
        s_idx[i] = (uint8_t)idx;
    }
    __syncthreads();

    // ── EDEN scale refinement (Option B: two-pass MSE-optimal scale) ────────────
    // Given current assignment {s_idx[i]}, the scale minimising ||v - S·c||²
    // is S* = Σ(v[i]·c[idx[i]]) / Σ(c[idx[i]]²)  (EDEN biased estimator).
    // Two passes: compute S₁ with initial assignment → re-quantize → compute S₂
    // with updated assignment → write S₂ as final scale.
    // Reuses s_k (dead after step 2 rotation) as a second reduction buffer.
    // V and outlier-split paths still use the RMS scale set in step 3.
    if (codebook != nullptr) {
        float *s_reduce2 = s_k;  // headDim ≥ blockDim.x; safe to alias dead buffer

        for (int pass = 0; pass < 2; pass++) {
            float local_num = 0.0f, local_den = 0.0f;
            for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
                float ci = codebook[(int)s_idx[i]];
                float vi = s_rot[i] - regMean;
                local_num += vi * ci;
                local_den += ci * ci;
            }

            s_reduce[threadIdx.x] = local_num;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }

            s_reduce2[threadIdx.x] = local_den;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    s_reduce2[threadIdx.x] += s_reduce2[threadIdx.x + stride];
                __syncthreads();
            }

            float eden_num = s_reduce[0];
            float eden_den = s_reduce2[0];
            float s_eden = (eden_den > 1e-12f && eden_num > 0.0f)
                           ? (eden_num / eden_den) : scale;
            scale = s_eden;

            if (threadIdx.x == 0)
                scales_out[cell * numKVHeads + head] = scale;

            if (pass == 0) {
                // Re-quantize with S₁ to get the assignment used in pass 1 and packing.
                for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
                    float v = (scale > 0.0f) ? ((s_rot[i] - regMean) / scale) : 0.0f;
                    int idx = 0;
                    for (int b = 0; b < numBoundaries; b++) {
                        if (v >= boundaries[b]) idx++;
                    }
                    s_idx[i] = (uint8_t)idx;
                }
                __syncthreads();
            }
        }
    }

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
    int              kIsF32,
    float           *zeros_out,          // regular zeros [numKVHeads*cells] (NULL if symmetric)
    float           *outlier_zeros_out,  // outlier zeros [numKVHeads*cells] (NULL if symmetric)
    uint8_t         *qjl_packed_out,     // QJL sign bits [qjlPackedBytes*numKVHeads*cells] (NULL if no QJL)
    float           *qjl_norm_out,       // QJL residual L2 norm [numKVHeads*cells] (NULL if no QJL)
    const float     *qjl_projection,     // [qjlRows, headDim] f32 projection matrix (NULL if no QJL)
    int              qjlRows,
    const float     *codebook,           // regular codebook [1<<bits] (NULL if no QJL)
    const float     *outlier_codebook,   // outlier codebook [1<<outlierBits] (NULL if no QJL)
    const float     *k_bias              // [numKVHeads * headDim] f32, NULL = no bias subtraction
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

    // Step 1: Load K into s_k as f32, subtracting k_bias if present.
    int base_k = batch * numKVHeads * headDim + head * headDim;
    int bias_base = head * headDim;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float val;
        if (kIsF32) {
            val = ((const float *)k)[base_k + d];
        } else {
            val = __half2float(__ushort_as_half(((const uint16_t *)k)[base_k + d]));
        }
        if (k_bias != nullptr) {
            val -= k_bias[bias_base + d];
        }
        s_k[d] = val;
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

    // Step 5: Per-sub-block stats (mean + RMS scale).
    // For symmetric presets: scale = sqrt(mean(v^2)).
    // For asymmetric presets: mean = avg(v), then scale = sqrt(mean((v-mean)^2)).
    float local_sum_reg = 0.0f;
    float local_sum_out = 0.0f;
    float local_sq_reg = 0.0f;
    float local_sq_out = 0.0f;
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float v = s_rot[i];
        if (s_is_outlier[i]) {
            local_sum_out += v;
            local_sq_out += v * v;
        } else {
            local_sum_reg += v;
            local_sq_reg += v * v;
        }
    }

    float regMean = 0.0f;
    float regScale = 0.0f;
    {
        s_reduce[threadIdx.x] = local_sum_reg;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        // All threads compute regMean directly. The previous write-back-to-
        // shared-then-read pattern raced under NVCC; see the simple kernel
        // for the reproducer. The race appears specifically when a second
        // reduction follows the first (mean → RMS scale), which is the
        // asymmetric path's signature.
        regMean = (regularCount > 0) ? (s_reduce[0] / (float)regularCount) : 0.0f;
        if (threadIdx.x == 0 && zeros_out) {
            zeros_out[cell * numKVHeads + head] = regMean;
        }
        __syncthreads();

        s_reduce[threadIdx.x] = local_sq_reg;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        {
            float sum_sq = s_reduce[0];
            if (zeros_out && regularCount > 0) {
                float centered_sq = sum_sq - regularCount * regMean * regMean;
                if (centered_sq > 1e-12f) regScale = sqrtf(centered_sq / (float)regularCount);
            } else if (sum_sq > 1e-12f && regularCount > 0) {
                regScale = sqrtf(sum_sq / (float)regularCount);
            }
        }
        if (threadIdx.x == 0) {
            scales_out[cell * numKVHeads + head] = regScale;
        }
        __syncthreads();
    }
    __syncthreads();

    float outMean = 0.0f;
    float outScale = 0.0f;
    {
        s_reduce[threadIdx.x] = local_sum_out;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        // All threads compute outMean directly (see regMean above for context).
        outMean = (outlierCount > 0) ? (s_reduce[0] / (float)outlierCount) : 0.0f;
        if (threadIdx.x == 0 && outlier_zeros_out) {
            outlier_zeros_out[cell * numKVHeads + head] = outMean;
        }
        __syncthreads();

        s_reduce[threadIdx.x] = local_sq_out;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        {
            float sum_sq = s_reduce[0];
            if (outlier_zeros_out && outlierCount > 0) {
                float centered_sq = sum_sq - outlierCount * outMean * outMean;
                if (centered_sq > 1e-12f) outScale = sqrtf(centered_sq / (float)outlierCount);
            } else if (sum_sq > 1e-12f && outlierCount > 0) {
                outScale = sqrtf(sum_sq / (float)outlierCount);
            }
        }
        if (threadIdx.x == 0) {
            outlier_scales[cell * numKVHeads + head] = outScale;
        }
        __syncthreads();
    }

    // Step 6: Quantize regular channels. s_idx[r] stores the code at the
    // CONTIGUOUS regular position r, not the original channel index, so the
    // dequant kernel can read them directly into the packed bit stream.
    // For asymmetric presets we centre by regMean before scaling.
    for (int r = threadIdx.x; r < regularCount; r += blockDim.x) {
        int orig = s_reg_pos[r];
        float v = (regScale > 0.0f) ? ((s_rot[orig] - regMean) / regScale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numBoundaries; b++) {
            if (v >= boundaries[b]) idx++;
        }
        s_idx[r] = (uint8_t)idx;
    }
    __syncthreads();

    // EDEN biased scale refinement for regular channels (two-pass).
    // Uses s_k as a second reduction buffer — it is dead after step 2.
    if (codebook != nullptr) {
        float *s_reduce2 = s_k;
        for (int pass = 0; pass < 2; pass++) {
            float local_num = 0.0f, local_den = 0.0f;
            for (int r = threadIdx.x; r < regularCount; r += blockDim.x) {
                int orig = s_reg_pos[r];
                float ci = codebook[(int)s_idx[r]];
                float vi = s_rot[orig] - regMean;
                local_num += vi * ci;
                local_den += ci * ci;
            }
            s_reduce[threadIdx.x] = local_num;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            s_reduce2[threadIdx.x] = local_den;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce2[threadIdx.x] += s_reduce2[threadIdx.x + stride];
                __syncthreads();
            }
            float s_eden = (s_reduce2[0] > 1e-12f && s_reduce[0] > 0.0f) ? (s_reduce[0] / s_reduce2[0]) : regScale;
            regScale = s_eden;
            if (threadIdx.x == 0) scales_out[cell * numKVHeads + head] = regScale;
            if (pass == 0) {
                for (int r = threadIdx.x; r < regularCount; r += blockDim.x) {
                    int orig = s_reg_pos[r];
                    float v = (regScale > 0.0f) ? ((s_rot[orig] - regMean) / regScale) : 0.0f;
                    int idx = 0;
                    for (int b = 0; b < numBoundaries; b++) { if (v >= boundaries[b]) idx++; }
                    s_idx[r] = (uint8_t)idx;
                }
                __syncthreads();
            }
        }
    }

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
    // For asymmetric presets we centre by outMean before scaling.
    for (int r = threadIdx.x; r < outlierCount; r += blockDim.x) {
        float v = (outScale > 0.0f) ? ((s_outl_val[r] - outMean) / outScale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numOutlierBoundaries; b++) {
            if (v >= outlier_boundaries[b]) idx++;
        }
        s_outl_idx[r] = (uint8_t)idx;
    }
    __syncthreads();

    // EDEN biased scale refinement for outlier channels (two-pass).
    if (outlier_codebook != nullptr) {
        float *s_reduce2 = s_k;
        for (int pass = 0; pass < 2; pass++) {
            float local_num = 0.0f, local_den = 0.0f;
            for (int r = threadIdx.x; r < outlierCount; r += blockDim.x) {
                float ci = outlier_codebook[(int)s_outl_idx[r]];
                float vi = s_outl_val[r] - outMean;
                local_num += vi * ci;
                local_den += ci * ci;
            }
            s_reduce[threadIdx.x] = local_num;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            s_reduce2[threadIdx.x] = local_den;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce2[threadIdx.x] += s_reduce2[threadIdx.x + stride];
                __syncthreads();
            }
            float s_eden = (s_reduce2[0] > 1e-12f && s_reduce[0] > 0.0f) ? (s_reduce[0] / s_reduce2[0]) : outScale;
            outScale = s_eden;
            if (threadIdx.x == 0) outlier_scales[cell * numKVHeads + head] = outScale;
            if (pass == 0) {
                for (int r = threadIdx.x; r < outlierCount; r += blockDim.x) {
                    float v = (outScale > 0.0f) ? ((s_outl_val[r] - outMean) / outScale) : 0.0f;
                    int idx = 0;
                    for (int b = 0; b < numOutlierBoundaries; b++) { if (v >= outlier_boundaries[b]) idx++; }
                    s_outl_idx[r] = (uint8_t)idx;
                }
                __syncthreads();
            }
        }
    }

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

    // Step 11: QJL residual sketch (optional).
    // Reconstruct the rotated vector from primary quantization, compute
    // residual = rotated - reconstructed, project onto Gaussian matrix rows,
    // store sign bits and L2 norm.
    if (qjlRows > 0 && qjl_packed_out && qjl_norm_out && qjl_projection) {
        // Reconstruct in shared memory using s_k as scratch (original K no longer needed).
        float *s_recon = s_k;
        for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
            s_recon[i] = 0.0f;
        }
        __syncthreads();

        // Reconstruct regular channels.
        for (int r = threadIdx.x; r < regularCount; r += blockDim.x) {
            int orig = s_reg_pos[r];
            float recon = (regScale > 0.0f) ? (codebook[s_idx[r]] * regScale + regMean) : regMean;
            s_recon[orig] = recon;
        }
        // Reconstruct outlier channels.
        for (int r = threadIdx.x; r < outlierCount; r += blockDim.x) {
            int orig = s_outl_pos[r];
            float recon = (outScale > 0.0f) ? (outlier_codebook[s_outl_idx[r]] * outScale + outMean) : outMean;
            s_recon[orig] = recon;
        }
        __syncthreads();

        // Compute residual = s_rot - s_recon, then L2 norm and projection signs.
        float local_l2 = 0.0f;
        for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
            float diff = s_rot[i] - s_recon[i];
            s_recon[i] = diff;  // reuse s_recon as residual scratch
            local_l2 += diff * diff;
        }
        s_reduce[threadIdx.x] = local_l2;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        float residualNorm = 0.0f;
        if (threadIdx.x == 0) {
            residualNorm = sqrtf(s_reduce[0]);
            qjl_norm_out[cell * numKVHeads + head] = residualNorm;
        }
        __syncthreads();

        // Compute projection dot products and store sign bits.
        // qjl_projection is [qjlRows, headDim] row-major.
        // Each thread handles one or more rows.
        if (residualNorm > 1e-12f) {
            for (int row = threadIdx.x; row < qjlRows; row += blockDim.x) {
                float dot = 0.0f;
                const float *proj_row = qjl_projection + row * headDim;
                for (int col = 0; col < headDim; col++) {
                    dot += s_recon[col] * proj_row[col];
                }
                int bit_idx = row;
                int byte_idx = bit_idx >> 3;
                int bit_in_byte = bit_idx & 7;
                uint8_t mask = (uint8_t)(1 << bit_in_byte);
                if (dot >= 0.0f) {
                    atomicOr((unsigned int *)(qjl_packed_out + (cell * numKVHeads + head) * ((qjlRows + 7) / 8) + (byte_idx & ~3)),
                             (unsigned int)mask << ((byte_idx & 3) * 8));
                }
            }
        } else if (threadIdx.x == 0) {
            // Zero residual: clear QJL packed bytes for this head.
            int qjl_packed_bytes = (qjlRows + 7) / 8;
            uint8_t *qp = qjl_packed_out + (cell * numKVHeads + head) * qjl_packed_bytes;
            for (int p = 0; p < qjl_packed_bytes; p++) qp[p] = 0;
        }
    }
}

// ── ggml dispatch ─────────────────────────────────────────────────────────────

// Extern declaration for the V encode kernel (defined in tq-encode-v.cu).
extern __global__ void tq_encode_v_kernel(
    const void *v, const float *rotation, uint8_t *packed_out, float *scales_out,
    int firstCell, const float *boundaries,
    int headDim, int numKVHeads, int bits, int numBoundaries, int vIsF32,
    const float *codebook);

void ggml_cuda_tq_encode(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * k          = dst->src[0];
    const struct ggml_tensor * rotation   = dst->src[1];
    const struct ggml_tensor * zeros      = dst->src[2];
    const struct ggml_tensor * scales     = dst->src[3];
    const struct ggml_tensor * boundaries = dst->src[4];
    // src[5] = k_bias for simple path (non-outlier); NULL when no bias.

    const int headDim    = (int)k->ne[0];
    const int numKVHeads = (int)k->ne[1];
    const int batchSize  = (int)k->ne[2];
    const int bits         = (int)((const int32_t *)dst->op_params)[0];
    const int firstCell    = (int)((const int32_t *)dst->op_params)[1];
    const int outlierBits  = (int)((const int32_t *)dst->op_params)[2];
    const int outlierCount = (int)((const int32_t *)dst->op_params)[3];
    const int asymmetric   = (int)((const int32_t *)dst->op_params)[4];
    const int qjlRows      = (int)((const int32_t *)dst->op_params)[5];
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
        const struct ggml_tensor * outlier_zeros      = dst->src[9];
        const struct ggml_tensor * qjl_packed         = dst->src[10];
        const struct ggml_tensor * qjl_norm           = dst->src[11];
        const struct ggml_tensor * qjl_projection     = dst->src[12];
        const struct ggml_tensor * codebook           = dst->src[13];
        const struct ggml_tensor * outlier_codebook   = dst->src[14];
        // src[15] = k_bias for outlier path; NULL when no bias.
        const struct ggml_tensor * k_bias_outlier     = dst->src[15];
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
            outlierBits, outlierCount, numOutlierBoundaries, kIsF32,
            asymmetric ? (float *)zeros->data : nullptr,
            asymmetric ? (float *)outlier_zeros->data : nullptr,
            qjlRows > 0 ? (uint8_t *)qjl_packed->data : nullptr,
            qjlRows > 0 ? (float *)qjl_norm->data : nullptr,
            qjlRows > 0 ? (const float *)qjl_projection->data : nullptr,
            qjlRows,
            codebook ? (const float *)codebook->data : nullptr,
            outlier_codebook ? (const float *)outlier_codebook->data : nullptr,
            k_bias_outlier ? (const float *)k_bias_outlier->data : nullptr
        );
        return;
    }

    size_t smem = (size_t)headDim * 2 * sizeof(float)
                + (size_t)block_size * sizeof(float)
                + (size_t)headDim * sizeof(uint8_t);

    const struct ggml_tensor * k_bias = dst->src[5];
    const float * k_bias_ptr = k_bias ? (const float *)k_bias->data : nullptr;
    // src[6] = codebook for EDEN biased scale refinement; NULL disables EDEN.
    const struct ggml_tensor * codebook_t = dst->src[6];
    const float * codebook_ptr = codebook_t ? (const float *)codebook_t->data : nullptr;

    if (asymmetric && zeros) {
        tq_encode_kernel<true><<<grid, block_size, smem, stream>>>(
            k->data,
            (const float    *)rotation->data,
            (uint8_t        *)dst->data,
            (float          *)scales->data,
            firstCell,
            (const float    *)boundaries->data,
            headDim, numKVHeads, bits, numBoundaries, kIsF32,
            (float *)zeros->data,
            k_bias_ptr,
            codebook_ptr
        );
    } else {
        tq_encode_kernel<false><<<grid, block_size, smem, stream>>>(
            k->data,
            (const float    *)rotation->data,
            (uint8_t        *)dst->data,
            (float          *)scales->data,
            firstCell,
            (const float    *)boundaries->data,
            headDim, numKVHeads, bits, numBoundaries, kIsF32,
            nullptr,
            k_bias_ptr,
            codebook_ptr
        );
    }
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

    // src[8] = k_bias, src[9] = k_codebook, src[10] = v_codebook (NULL = RMS only).
    const struct ggml_tensor * k_bias_kv      = dst->src[8];
    const struct ggml_tensor * k_codebook_kv  = dst->src[9];
    const struct ggml_tensor * v_codebook_kv  = dst->src[10];
    const float * k_bias_kv_ptr     = k_bias_kv     ? (const float *)k_bias_kv->data     : nullptr;
    const float * k_codebook_kv_ptr = k_codebook_kv ? (const float *)k_codebook_kv->data : nullptr;
    const float * v_codebook_kv_ptr = v_codebook_kv ? (const float *)v_codebook_kv->data : nullptr;

    // K encode: TQEncodeKV always uses the symmetric kernel (zeros=nullptr).
    // Asymmetric presets route through separate EncodeK+EncodeV (see turboquant_compressed.go).
    tq_encode_kernel<false><<<grid, block_size, smem, stream>>>(
        k->data,
        (const float *)rotation->data,
        (uint8_t     *)dst->data,
        (float       *)k_scales->data,
        firstCell,
        (const float *)k_bounds->data,
        headDim, numKVHeads, k_bits, (1 << k_bits) - 1, kIsF32,
        nullptr,            // zeros (asymmetric not used on combined KV path)
        k_bias_kv_ptr,
        k_codebook_kv_ptr   // EDEN biased scale when codebook provided
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
        headDim, numKVHeads, v_bits, (1 << v_bits) - 1, vIsF32,
        v_codebook_kv_ptr
    );
}
