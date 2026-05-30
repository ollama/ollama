#include "tq-encode.cuh"
#include "tq-wht.cuh"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <vector>

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
    const float     *rotation,     // [headDim] f32 ±1 WHT sign vector
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
    const float     *k_bias,          // [numKVHeads * headDim] f32, NULL = no bias subtraction
    const float     *codebook,        // [1<<bits] f32 Lloyd-Max centroids; NULL = skip EDEN refinement
    const int32_t   *locs,            // [batchSize] i32 physical-cell indices; NULL = contiguous (cell = firstCell + batch)
    int              block_size        // = block_size; passed explicitly to avoid %ntid.x on sm_120
) {
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int cell  = locs ? locs[batch] : (firstCell + batch);

    // Shared memory layout:
    //   s_k[headDim]       – K values as f32; WHT runs in-place, leaving rotated values
    //   s_reduce[BLOCK]    – warp reduction scratch
    //   s_idx[headDim]     – quantized indices (uint8)
    //   s_idx_rms[headDim] – RMS-only quantized indices (for Path B fallback);
    //                       allocated only when codebook != nullptr
    extern __shared__ char s_mem[];
    float   *s_k       = (float *)s_mem;
    float   *s_reduce  = s_k + headDim;
    uint8_t *s_idx     = (uint8_t *)(s_reduce + block_size);
    uint8_t *s_idx_rms = s_idx + headDim;  // valid only when EDEN enabled

    // sm_120: avoid all reads of %ntid.x (blockDim.x) inside the kernel.
    // block_size is passed as a regular int parameter — safe on all architectures.
    const int half_block = block_size >> 1;

    // ── Step 1: Load K[batch, head] into shared memory as f32 ────────────────
    // If k_bias is provided, subtract it before rotation so the Lloyd-Max
    // codebook sees zero-mean data even when the K projection has a bias term
    // (e.g. Qwen2). The bias term cancels in softmax attention scores so the
    // decoder does not need to add it back.
    int base_k = batch * numKVHeads * headDim + head * headDim;
    int bias_base = head * headDim;
    for (int d = threadIdx.x; d < headDim; d += block_size) {
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

    // ── Step 2: WHT rotation F(x) = S·H·S·x/√n (self-inverse) ─────────────
    // kForceFast when block_size==headDim (D<=128): avoids sm_120 Pattern 3 miscompile.
    // D=256 uses block_size=128 < headDim, so kForceFast is unsafe there.
    if (block_size == headDim) {
        apply_shs_wht<false, true>(s_k, rotation, headDim, threadIdx.x, block_size);
    } else {
        apply_shs_wht<false, false>(s_k, rotation, headDim, threadIdx.x, block_size);
    }

    // ── Step 3: Mean (kAsymmetric only) then RMS scale ───────────────────────
    float regMean = 0.0f;
    if (kAsymmetric) {
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < headDim; i += block_size) {
            local_sum += s_k[i];
        }
        s_reduce[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = half_block; stride > 0; stride >>= 1) {
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
    for (int i = threadIdx.x; i < headDim; i += block_size) {
        float v = s_k[i] - regMean;  // regMean==0 when !kAsymmetric
        local_sq += v * v;
    }
    s_reduce[threadIdx.x] = local_sq;
    __syncthreads();

    for (int stride = half_block; stride > 0; stride >>= 1) {
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
    // When Path B is active, scales_out is written at the end of the Path B
    // block with the final (RMS or EDEN) scale. Skip the early write here to
    // avoid a redundant store.
    if (threadIdx.x == 0 && codebook == nullptr) {
        scales_out[cell * numKVHeads + head] = scale;
    }
    __syncthreads();

    // ── Step 4: Quantize each element via boundary binary search ─────────────
    for (int i = threadIdx.x; i < headDim; i += block_size) {
        float v = (scale > 0.0f) ? ((s_k[i] - regMean) / scale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numBoundaries; b++) {
            if (v >= boundaries[b]) idx++;
        }
        s_idx[i] = (uint8_t)idx;
    }
    __syncthreads();

    // ── EDEN scale refinement (Path B: adaptive RMS-vs-EDEN fallback) ───────────
    // Two-pass MSE-optimal scale: S* = Σ(v[i]·c[idx[i]]) / Σ(c[idx[i]]²).
    // EDEN diverges on poorly-matched codebooks (3+ bit on non-Gaussian K) so we
    // compute reconstruction error for both the RMS-only and EDEN-refined
    // (scale, codes) pairs and keep whichever has lower MSE per cell-head.
    // Provably non-worse than RMS-only.
    if (codebook != nullptr) {
        // Save RMS codes + scale before EDEN overwrites them.
        const float scale_rms = scale;
        for (int i = threadIdx.x; i < headDim; i += block_size) {
            s_idx_rms[i] = s_idx[i];
        }
        __syncthreads();

        // Two passes of EDEN refinement, re-quantizing after each pass.
        for (int pass = 0; pass < 2; pass++) {
            float local_num = 0.0f, local_den = 0.0f;
            for (int i = threadIdx.x; i < headDim; i += block_size) {
                float ci = codebook[(int)s_idx[i]];
                float vi = s_k[i] - regMean;
                local_num += vi * ci;
                local_den += ci * ci;
            }

            s_reduce[threadIdx.x] = local_num;
            __syncthreads();
            for (int stride = half_block; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            float eden_num = s_reduce[0];
            __syncthreads();   // ensure all reads of s_reduce[0] complete before reuse

            s_reduce[threadIdx.x] = local_den;
            __syncthreads();
            for (int stride = half_block; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                    s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            float s_eden = (s_reduce[0] > 1e-12f && eden_num > 0.0f)
                           ? (eden_num / s_reduce[0]) : scale;
            scale = s_eden;

            for (int i = threadIdx.x; i < headDim; i += block_size) {
                float v = (scale > 0.0f) ? ((s_k[i] - regMean) / scale) : 0.0f;
                int idx = 0;
                for (int b = 0; b < numBoundaries; b++) {
                    if (v >= boundaries[b]) idx++;
                }
                s_idx[i] = (uint8_t)idx;
            }
            __syncthreads();
        }

        // Compute reconstruction error for EDEN-refined (scale, codes).
        float local_err_eden = 0.0f;
        for (int i = threadIdx.x; i < headDim; i += block_size) {
            float predicted = codebook[(int)s_idx[i]] * scale;
            float actual    = s_k[i] - regMean;
            float diff      = actual - predicted;
            local_err_eden += diff * diff;
        }
        s_reduce[threadIdx.x] = local_err_eden;
        __syncthreads();
        for (int stride = half_block; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        const float err_eden = s_reduce[0];
        __syncthreads();   // ensure all reads of s_reduce[0] complete before reuse

        // Compute reconstruction error for RMS-only (saved scale, codes).
        float local_err_rms = 0.0f;
        for (int i = threadIdx.x; i < headDim; i += block_size) {
            float predicted = codebook[(int)s_idx_rms[i]] * scale_rms;
            float actual    = s_k[i] - regMean;
            float diff      = actual - predicted;
            local_err_rms += diff * diff;
        }
        s_reduce[threadIdx.x] = local_err_rms;
        __syncthreads();
        for (int stride = half_block; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        const float err_rms = s_reduce[0];

        // Pick the lower-error pair. RMS wins → restore saved codes + scale.
        if (err_rms < err_eden) {
            scale = scale_rms;
            for (int i = threadIdx.x; i < headDim; i += block_size) {
                s_idx[i] = s_idx_rms[i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            scales_out[cell * numKVHeads + head] = scale;
        }
        __syncthreads();
    }

    // ── Step 5: Pack bits LSB-first into output (parallel via atomicOr) ─────
    {
        int packed_bytes = (headDim * bits + 7) / 8;
        uint8_t *out = packed_out + (cell * numKVHeads + head) * packed_bytes;
        uint8_t bitmask = (uint8_t)((1 << bits) - 1);

        // Zero output buffer in parallel.
        for (int p = threadIdx.x; p < packed_bytes; p += block_size) {
            out[p] = 0;
        }
        __syncthreads();

        // Each thread packs its own elements using atomicOr on 4-byte aligned words.
        for (int elem = threadIdx.x; elem < headDim; elem += block_size) {
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
    const float     *rotation,           // [headDim] f32 ±1 WHT sign vector
    uint8_t         *packed_out,         // regular packed [regularPackedBytes*numKVHeads*cells]
    float           *scales_out,         // regular scales [numKVHeads*cells]
    uint8_t         *outlier_packed,     // outlier packed [outlierPackedBytes*numKVHeads*cells]
    float           *outlier_scales,     // outlier scales [numKVHeads*cells]
    uint16_t        *outlier_indices,    // outlier channel idx [outlierCount*numKVHeads*cells] (uint16 for D=512 support)
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
    const float     *codebook,           // regular codebook [1<<bits] (NULL if EDEN refinement disabled)
    const float     *outlier_codebook,   // outlier codebook [1<<outlierBits]
    const float     *k_bias,              // [numKVHeads * headDim] f32, NULL = no bias subtraction
    const int32_t   *locs,               // [batchSize] i32 physical-cell indices; NULL = contiguous (cell = firstCell + batch)
    int              block_size           // = blockDim.x; passed explicitly to avoid %ntid.x on sm_120
) {
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int cell  = locs ? locs[batch] : (firstCell + batch);

    // Shared memory layout (laid out contiguously — launch passes total size):
    //   s_k[headDim]              - f32 K values; WHT runs in-place, leaving rotated values
    //   s_reduce[block_size]       - reduction scratch (float)
    //   s_idx[headDim]             - regular quantized indices (uint8)
    //   s_is_outlier[headDim]      - 1 = outlier, 0 = regular (uint8)
    //   s_reg_pos[headDim]         - regular position index map (int, up to headDim entries)
    //   s_outl_pos[outlierCount]   - outlier channel positions (int)
    //   s_outl_val[outlierCount]   - outlier rotated values (float)
    //   s_outl_idx[outlierCount]   - outlier quantized indices (uint8)

    extern __shared__ char s_mem[];
    float   *s_k            = (float *)s_mem;
    float   *s_reduce       = s_k + headDim;
    const int half_block = block_size >> 1;  // block_size is a param, not %ntid.x — safe on sm_120
    uint8_t *s_idx          = (uint8_t *)(s_reduce + block_size);
    uint8_t *s_is_outlier   = s_idx + headDim;
    int     *s_reg_pos      = (int *)(s_is_outlier + headDim);
    int     *s_outl_pos     = s_reg_pos + headDim;
    float   *s_outl_val     = (float *)(s_outl_pos + outlierCount);
    uint8_t *s_outl_idx     = (uint8_t *)(s_outl_val + outlierCount);
    // Path B fallback buffers — only valid when respective codebook != null.
    uint8_t *s_idx_rms      = s_outl_idx + outlierCount;     // [headDim]
    uint8_t *s_outl_idx_rms = s_idx_rms + headDim;           // [outlierCount]

    // Step 1: Load K into s_k as f32, subtracting k_bias if present.
    int base_k = batch * numKVHeads * headDim + head * headDim;
    int bias_base = head * headDim;
    for (int d = threadIdx.x; d < headDim; d += block_size) {
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

    // Step 2: WHT rotation F(x) = S·H·S·x/√n (self-inverse).
    // kForceFast when block_size==headDim (D<=128): avoids sm_120 Pattern 3 miscompile.
    if (block_size == headDim) {
        apply_shs_wht<false, true>(s_k, rotation, headDim, threadIdx.x, block_size);
    } else {
        apply_shs_wht<false, false>(s_k, rotation, headDim, threadIdx.x, block_size);
    }

    // Step 3: Clear outlier mask.
    for (int i = threadIdx.x; i < headDim; i += block_size) {
        s_is_outlier[i] = 0;
    }
    __syncthreads();

    // Step 4: Top-K outlier selection. The former serial-on-thread-0 scan was
    // O(K*headDim) on one thread (~4k dependent shared-mem reads, 127 threads
    // idle): cheap at prefill (many blocks hide the latency) but it dominated
    // single-token decode, where the launch is grid=(1,numKVHeads) → only
    // numKVHeads blocks and the latency is fully exposed (~147µs/layer on P40).
    // See decode-gap profiling 2026-05-28.
    //
    // Both branches below are bit-exact with that serial scan: they select the
    // same outlierCount channels by descending |value| with ties broken to the
    // LOWEST channel index (matching the old `a > best_val` first-max-wins), and
    // emit s_outl_pos[r] in the same descending order, so the packed bytes are
    // identical. s_reduce[block_size] / s_reg_pos[block_size] are borrowed as
    // scratch; s_reg_pos's real regular-position map is built afterwards (below).
    if (block_size == headDim) {
        // Single-pass: bitonic sort of (|value|, index), one thread per element.
        // ~28 compare-exchange steps for headDim=128 vs the argmax fallback's
        // ~32 sequential rounds — far fewer barriers and no per-round rescan.
        // Indices are unique so the key order is a strict total order: fully
        // deterministic, no float arithmetic, sm_120-safe (indexed smem + sync).
        s_reduce[threadIdx.x]  = fabsf(s_k[threadIdx.x]);
        s_reg_pos[threadIdx.x] = threadIdx.x;
        __syncthreads();
        for (int k = 2; k <= headDim; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                int ixj = threadIdx.x ^ j;
                if (ixj > threadIdx.x) {
                    float ma = s_reduce[threadIdx.x],  mb = s_reduce[ixj];
                    int   ia = s_reg_pos[threadIdx.x], ib = s_reg_pos[ixj];
                    // i ranks ahead of ixj iff larger |value|, ties → lower index
                    bool i_first = (ma > mb) || (ma == mb && ia < ib);
                    bool ascending = ((threadIdx.x & k) == 0);  // higher rank → lower pos
                    if (ascending != i_first) {  // swap to satisfy this sub-block's order
                        s_reduce[threadIdx.x]  = mb; s_reduce[ixj]  = ma;
                        s_reg_pos[threadIdx.x] = ib; s_reg_pos[ixj] = ia;
                    }
                }
                __syncthreads();
            }
        }
        // Sorted descending: positions 0..outlierCount-1 are the outliers, in order.
        for (int r = threadIdx.x; r < outlierCount; r += block_size) {
            int oi = s_reg_pos[r];
            s_outl_pos[r]    = oi;
            s_outl_val[r]    = s_k[oi];
            s_is_outlier[oi] = 1;
        }
        __syncthreads();
    } else {
        // Fallback (block_size < headDim, e.g. D=256): per-round parallel argmax.
        // s_reduce holds candidate magnitudes, s_reg_pos the indices; reduction
        // mirrors the Step-5 pattern (half_block, not blockDim.x) — sm_120-safe.
        for (int r = 0; r < outlierCount; r++) {
            float my_best = -1.0f;
            int   my_idx  = headDim;  // sentinel; every real index is < headDim
            for (int i = threadIdx.x; i < headDim; i += block_size) {
                if (s_is_outlier[i]) continue;
                float a = fabsf(s_k[i]);
                if (a > my_best) { my_best = a; my_idx = i; }
            }
            s_reduce[threadIdx.x]  = my_best;
            s_reg_pos[threadIdx.x] = my_idx;
            __syncthreads();
            for (int stride = half_block; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    float ov = s_reduce[threadIdx.x + stride];
                    float cv = s_reduce[threadIdx.x];
                    int   oi = s_reg_pos[threadIdx.x + stride];
                    int   ci = s_reg_pos[threadIdx.x];
                    if (ov > cv || (ov == cv && oi < ci)) {  // max; tie → lower index
                        s_reduce[threadIdx.x]  = ov;
                        s_reg_pos[threadIdx.x] = oi;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                int best_idx = s_reg_pos[0];
                s_is_outlier[best_idx] = 1;
                s_outl_pos[r]          = best_idx;
                s_outl_val[r]          = s_k[best_idx];
            }
            __syncthreads();
        }
    }

    // Build the regular position map (ascending original indices). O(headDim)
    // on thread 0 — 32x lighter than the old selection loop and not the hot
    // path; kept serial for clarity. Overwrites the borrowed index scratch.
    if (threadIdx.x == 0) {
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
    for (int i = threadIdx.x; i < headDim; i += block_size) {
        float v = s_k[i];
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
        for (int stride = half_block; stride > 0; stride >>= 1) {
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
        for (int stride = half_block; stride > 0; stride >>= 1) {
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
        // When Path B is active for the regular sub-block, scales_out is
        // written at the end of that block with the final scale. Skip the
        // early write to avoid a redundant store.
        if (threadIdx.x == 0 && codebook == nullptr) {
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
        for (int stride = half_block; stride > 0; stride >>= 1) {
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
        for (int stride = half_block; stride > 0; stride >>= 1) {
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
        // When Path B is active for the outlier sub-block, outlier_scales is
        // written at the end of that block with the final scale.
        if (threadIdx.x == 0 && outlier_codebook == nullptr) {
            outlier_scales[cell * numKVHeads + head] = outScale;
        }
        __syncthreads();
    }

    // Step 6: Quantize regular channels. s_idx[r] stores the code at the
    // CONTIGUOUS regular position r, not the original channel index, so the
    // dequant kernel can read them directly into the packed bit stream.
    // For asymmetric presets we centre by regMean before scaling.
    for (int r = threadIdx.x; r < regularCount; r += block_size) {
        int orig = s_reg_pos[r];
        float v = (regScale > 0.0f) ? ((s_k[orig] - regMean) / regScale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numBoundaries; b++) {
            if (v >= boundaries[b]) idx++;
        }
        s_idx[r] = (uint8_t)idx;
    }
    __syncthreads();

    // Path B: adaptive RMS-vs-EDEN scale refinement for regular channels.
    // EDEN diverges on poorly-matched codebooks (3+ bit on non-Gaussian K),
    // so we save the RMS-only (scale, codes) pair, run EDEN, then compute
    // reconstruction error for both and keep the lower-MSE pair per cell-head.
    // Provably non-worse than RMS-only.
    if (codebook != nullptr) {
        const float regScale_rms = regScale;
        for (int r = threadIdx.x; r < regularCount; r += block_size) {
            s_idx_rms[r] = s_idx[r];
        }
        __syncthreads();

        for (int pass = 0; pass < 2; pass++) {
            float local_num = 0.0f, local_den = 0.0f;
            for (int r = threadIdx.x; r < regularCount; r += block_size) {
                int orig = s_reg_pos[r];
                float ci = codebook[(int)s_idx[r]];
                float vi = s_k[orig] - regMean;
                local_num += vi * ci;
                local_den += ci * ci;
            }
            s_reduce[threadIdx.x] = local_num;
            __syncthreads();
            for (int stride = half_block; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            float eden_num = s_reduce[0];
            __syncthreads();   // ensure all reads of s_reduce[0] complete before reuse
            s_reduce[threadIdx.x] = local_den;
            __syncthreads();
            for (int stride = half_block; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            float s_eden = (s_reduce[0] > 1e-12f && eden_num > 0.0f) ? (eden_num / s_reduce[0]) : regScale;
            regScale = s_eden;
            // Re-quantize after every pass so stored scale and stored codes
            // always derive from the same iteration.
            for (int r = threadIdx.x; r < regularCount; r += block_size) {
                int orig = s_reg_pos[r];
                float v = (regScale > 0.0f) ? ((s_k[orig] - regMean) / regScale) : 0.0f;
                int idx = 0;
                for (int b = 0; b < numBoundaries; b++) { if (v >= boundaries[b]) idx++; }
                s_idx[r] = (uint8_t)idx;
            }
            __syncthreads();
        }

        // Reconstruction error for EDEN-refined (regScale, s_idx).
        float local_err_eden = 0.0f;
        for (int r = threadIdx.x; r < regularCount; r += block_size) {
            int orig = s_reg_pos[r];
            float predicted = codebook[(int)s_idx[r]] * regScale;
            float actual    = s_k[orig] - regMean;
            float diff      = actual - predicted;
            local_err_eden += diff * diff;
        }
        s_reduce[threadIdx.x] = local_err_eden;
        __syncthreads();
        for (int stride = half_block; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        const float err_eden = s_reduce[0];
        __syncthreads();   // ensure all reads of s_reduce[0] complete before reuse

        // Reconstruction error for RMS-only (regScale_rms, s_idx_rms).
        float local_err_rms = 0.0f;
        for (int r = threadIdx.x; r < regularCount; r += block_size) {
            int orig = s_reg_pos[r];
            float predicted = codebook[(int)s_idx_rms[r]] * regScale_rms;
            float actual    = s_k[orig] - regMean;
            float diff      = actual - predicted;
            local_err_rms += diff * diff;
        }
        s_reduce[threadIdx.x] = local_err_rms;
        __syncthreads();
        for (int stride = half_block; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        const float err_rms = s_reduce[0];

        // RMS wins → restore saved codes + scale.
        if (err_rms < err_eden) {
            regScale = regScale_rms;
            for (int r = threadIdx.x; r < regularCount; r += block_size) {
                s_idx[r] = s_idx_rms[r];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            scales_out[cell * numKVHeads + head] = regScale;
        }
        __syncthreads();
    }

    // Step 7: Pack regular bits into packed_out.
    // Per-head stride is padded up to a 4-byte multiple so atomicOr on
    // unsigned-int words stays aligned for every head, regardless of
    // how many regular channels (bit count) are in play. The Go-side
    // allocator uses the same padded value (regularPackedBytes()).
    const int regular_packed_bytes_raw    = (regularCount * bits + 7) / 8;
    const int regular_packed_bytes        = (regular_packed_bytes_raw + 3) & ~3;
    uint8_t *reg_out = packed_out + (cell * numKVHeads + head) * regular_packed_bytes;
    for (int p = threadIdx.x; p < regular_packed_bytes; p += block_size) {
        reg_out[p] = 0;
    }
    __syncthreads();
    {
        uint8_t bitmask = (uint8_t)((1 << bits) - 1);
        for (int r = threadIdx.x; r < regularCount; r += block_size) {
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
    for (int r = threadIdx.x; r < outlierCount; r += block_size) {
        float v = (outScale > 0.0f) ? ((s_outl_val[r] - outMean) / outScale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numOutlierBoundaries; b++) {
            if (v >= outlier_boundaries[b]) idx++;
        }
        s_outl_idx[r] = (uint8_t)idx;
    }
    __syncthreads();

    // Path B: adaptive RMS-vs-EDEN scale refinement for outlier channels.
    // Same pattern as the regular sub-block above.
    if (outlier_codebook != nullptr) {
        const float outScale_rms = outScale;
        for (int r = threadIdx.x; r < outlierCount; r += block_size) {
            s_outl_idx_rms[r] = s_outl_idx[r];
        }
        __syncthreads();

        for (int pass = 0; pass < 2; pass++) {
            float local_num = 0.0f, local_den = 0.0f;
            for (int r = threadIdx.x; r < outlierCount; r += block_size) {
                float ci = outlier_codebook[(int)s_outl_idx[r]];
                float vi = s_outl_val[r] - outMean;
                local_num += vi * ci;
                local_den += ci * ci;
            }
            s_reduce[threadIdx.x] = local_num;
            __syncthreads();
            for (int stride = half_block; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            float eden_num = s_reduce[0];
            __syncthreads();   // ensure all reads of s_reduce[0] complete before reuse
            s_reduce[threadIdx.x] = local_den;
            __syncthreads();
            for (int stride = half_block; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            float s_eden = (s_reduce[0] > 1e-12f && eden_num > 0.0f) ? (eden_num / s_reduce[0]) : outScale;
            outScale = s_eden;
            for (int r = threadIdx.x; r < outlierCount; r += block_size) {
                float v = (outScale > 0.0f) ? ((s_outl_val[r] - outMean) / outScale) : 0.0f;
                int idx = 0;
                for (int b = 0; b < numOutlierBoundaries; b++) { if (v >= outlier_boundaries[b]) idx++; }
                s_outl_idx[r] = (uint8_t)idx;
            }
            __syncthreads();
        }

        // Reconstruction error for EDEN-refined.
        float local_err_eden = 0.0f;
        for (int r = threadIdx.x; r < outlierCount; r += block_size) {
            float predicted = outlier_codebook[(int)s_outl_idx[r]] * outScale;
            float actual    = s_outl_val[r] - outMean;
            float diff      = actual - predicted;
            local_err_eden += diff * diff;
        }
        s_reduce[threadIdx.x] = local_err_eden;
        __syncthreads();
        for (int stride = half_block; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        const float err_eden = s_reduce[0];
        __syncthreads();   // ensure all reads of s_reduce[0] complete before reuse

        // Reconstruction error for RMS-only.
        float local_err_rms = 0.0f;
        for (int r = threadIdx.x; r < outlierCount; r += block_size) {
            float predicted = outlier_codebook[(int)s_outl_idx_rms[r]] * outScale_rms;
            float actual    = s_outl_val[r] - outMean;
            float diff      = actual - predicted;
            local_err_rms += diff * diff;
        }
        s_reduce[threadIdx.x] = local_err_rms;
        __syncthreads();
        for (int stride = half_block; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        const float err_rms = s_reduce[0];

        if (err_rms < err_eden) {
            outScale = outScale_rms;
            for (int r = threadIdx.x; r < outlierCount; r += block_size) {
                s_outl_idx[r] = s_outl_idx_rms[r];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            outlier_scales[cell * numKVHeads + head] = outScale;
        }
        __syncthreads();
    }

    // Step 9: Pack outlier bits. Same 4-byte alignment as regular packing.
    const int outlier_packed_bytes_raw = (outlierCount * outlierBits + 7) / 8;
    const int outlier_packed_bytes     = (outlier_packed_bytes_raw + 3) & ~3;
    uint8_t *out_out = outlier_packed + (cell * numKVHeads + head) * outlier_packed_bytes;
    for (int p = threadIdx.x; p < outlier_packed_bytes; p += block_size) {
        out_out[p] = 0;
    }
    __syncthreads();
    {
        uint8_t obmask = (uint8_t)((1 << outlierBits) - 1);
        for (int r = threadIdx.x; r < outlierCount; r += block_size) {
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

    // Step 10: Write outlier channel indices. uint16_t covers 0..511 safely.
    uint16_t *idx_out = outlier_indices + (cell * numKVHeads + head) * outlierCount;
    for (int r = threadIdx.x; r < outlierCount; r += block_size) {
        idx_out[r] = (uint16_t)s_outl_pos[r];
    }
}

// ── ggml dispatch ─────────────────────────────────────────────────────────────

// Extern declaration for the V encode kernel (defined in tq-encode-v.cu).
extern __global__ void tq_encode_v_kernel(
    const void *v, const float *rotation, uint8_t *packed_out, float *scales_out,
    int firstCell, const float *boundaries,
    int headDim, int numKVHeads, int bits, int numBoundaries, int vIsF32,
    const float *codebook,
    const int32_t *locs,
    int block_size);

// Path B (adaptive RMS-vs-EDEN scale fallback) is ON by default. The kernel
// computes BOTH the RMS-only and EDEN-refined (scale, codes) pairs, then
// keeps whichever has lower reconstruction error per cell-head. Provably
// non-worse than RMS-only since we take the min of two options. Costs
// ~15-25% extra encoder kernel time and one headDim of threadgroup memory.
//
// OLLAMA_TQ_DISABLE_EDEN=1 forces RMS-only (passes codebook=NULL to the
// kernel), skipping the EDEN comparison entirely. Used as a diagnostic
// escape hatch — Path B is provably correct, so production users should
// not need this.
bool tq_encode_eden_disabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = 0;
        if (const char * env = std::getenv("OLLAMA_TQ_DISABLE_EDEN")) {
            cached = std::atoi(env) != 0 ? 1 : 0;
        }
    }
    return cached != 0;
}

// OLLAMA_TQ_DISABLE_KBIAS=1 forces k_bias_ptr=nullptr, skipping bias subtraction
// in the load step. Diagnostic only — disabling bias subtraction degrades qwen2.5
// quality since K values will be biased (large positive offset in early layers).
static bool tq_encode_kbias_disabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = 0;
        if (const char * env = std::getenv("OLLAMA_TQ_DISABLE_KBIAS")) {
            cached = std::atoi(env) != 0 ? 1 : 0;
        }
    }
    return cached != 0;
}


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
    // op_params[5] is reserved.
    const int numBoundaries = (1 << bits) - 1;
    const int kIsF32     = (k->type == GGML_TYPE_F32) ? 1 : 0;

    dim3 grid(batchSize, numKVHeads);
    int block_size = (headDim < TQ_ENCODE_BLOCK_SIZE) ? headDim : TQ_ENCODE_BLOCK_SIZE;
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;

    cudaStream_t stream = ctx.stream();


    const bool eden_disabled  = tq_encode_eden_disabled();
    const bool kbias_disabled = tq_encode_kbias_disabled();

    const struct ggml_tensor * locs = dst->src[10];
    const int32_t * locs_ptr = locs ? (const int32_t *)locs->data : nullptr;

    if (outlierCount > 0 && outlierBits > 0 && outlierCount < headDim) {
        const struct ggml_tensor * outlier_packed     = dst->src[5];
        const struct ggml_tensor * outlier_scales     = dst->src[6];
        const struct ggml_tensor * outlier_indices    = dst->src[7];
        const struct ggml_tensor * outlier_boundaries = dst->src[8];
        const struct ggml_tensor * outlier_zeros      = dst->src[9];
        // src[11..12] reserved.
        const struct ggml_tensor * codebook           = dst->src[13];
        const struct ggml_tensor * outlier_codebook   = dst->src[14];
        const struct ggml_tensor * k_bias_outlier     = dst->src[15];
        const int numOutlierBoundaries = (1 << outlierBits) - 1;

        // Shared memory layout for outlier kernel:
        //   s_k[headDim]              f32 (rotated in-place)
        //   s_reduce[block_size]       f32
        //   s_idx[headDim]             u8
        //   s_is_outlier[headDim]      u8
        //   s_reg_pos[headDim]         i32 (over-sized to headDim; only regularCount used)
        //   s_outl_pos[outlierCount]   i32
        //   s_outl_val[outlierCount]   f32
        //   s_outl_idx[outlierCount]   u8
        //   s_idx_rms[headDim]         u8  (Path B; allocated only when codebook present)
        //   s_outl_idx_rms[outlierCount] u8 (Path B outlier; allocated only when outlier_codebook present)
        const bool path_b_reg     = (codebook != nullptr && !eden_disabled);
        const bool path_b_outlier = (outlier_codebook != nullptr && !eden_disabled);
        // Kernel computes pointers as s_idx_rms = s_outl_idx + outlierCount and
        // s_outl_idx_rms = s_idx_rms + headDim unconditionally, so when only the
        // outlier path is active we must still reserve the headDim slot or the
        // outlier rms buffer lands past allocated smem. Allocate when EITHER is
        // active. In production both Path B sides travel together so this is
        // also defensive against future preset combinations.
        const bool path_b_any = (path_b_reg || path_b_outlier);
        size_t smem = (size_t)headDim * sizeof(float)                    // s_k
                    + (size_t)block_size * sizeof(float)                 // s_reduce
                    + (size_t)headDim * 2 * sizeof(uint8_t)              // s_idx + s_is_outlier
                    + (size_t)headDim * sizeof(int)                      // s_reg_pos
                    + (size_t)outlierCount * (sizeof(int) + sizeof(float) + sizeof(uint8_t))
                    + (path_b_any     ? (size_t)headDim      * sizeof(uint8_t) : 0)   // s_idx_rms
                    + (path_b_outlier ? (size_t)outlierCount * sizeof(uint8_t) : 0);  // s_outl_idx_rms

        tq_encode_kernel_outlier<<<grid, block_size, smem, stream>>>(
            k->data,
            (const float *)rotation->data,
            (uint8_t     *)dst->data,
            (float       *)scales->data,
            (uint8_t     *)outlier_packed->data,
            (float       *)outlier_scales->data,
            (uint16_t    *)outlier_indices->data,
            firstCell,
            (const float *)boundaries->data,
            (const float *)outlier_boundaries->data,
            headDim, numKVHeads, bits, numBoundaries,
            outlierBits, outlierCount, numOutlierBoundaries, kIsF32,
            asymmetric ? (float *)zeros->data : nullptr,
            asymmetric ? (float *)outlier_zeros->data : nullptr,
            (codebook && !eden_disabled) ? (const float *)codebook->data : nullptr,
            (outlier_codebook && !eden_disabled) ? (const float *)outlier_codebook->data : nullptr,
            (k_bias_outlier && !kbias_disabled) ? (const float *)k_bias_outlier->data : nullptr,
            locs_ptr,
            block_size
        );
        return;
    }

    const struct ggml_tensor * k_bias = dst->src[5];
    const float * k_bias_ptr = (k_bias && !kbias_disabled) ? (const float *)k_bias->data : nullptr;
    // src[6] = codebook for Path B adaptive scale (compare RMS vs EDEN per
    // cell-head, keep lower reconstruction error). NULL forces RMS-only.
    const struct ggml_tensor * codebook_t = dst->src[6];
    const float * codebook_ptr = (codebook_t && !eden_disabled) ? (const float *)codebook_t->data : nullptr;

    // s_idx_rms[headDim] is only touched when codebook != nullptr (Path B
    // adaptive path needs to save RMS codes before EDEN overwrites them).
    size_t smem = (size_t)headDim * sizeof(float)       // s_k
                + (size_t)block_size * sizeof(float)    // s_reduce
                + (size_t)headDim * sizeof(uint8_t)     // s_idx
                + (codebook_ptr ? (size_t)headDim * sizeof(uint8_t) : 0); // s_idx_rms

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
            codebook_ptr,
            locs_ptr,
            block_size
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
            codebook_ptr,
            locs_ptr,
            block_size
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

    cudaStream_t stream = ctx.stream();

    // src[8] = k_bias, src[9] = k_codebook, src[10] = v_codebook (NULL = RMS only).
    // src[11] = locs (NULL = contiguous; [batchSize] i32 = indexed slots).
    const struct ggml_tensor * k_bias_kv      = dst->src[8];
    const struct ggml_tensor * k_codebook_kv  = dst->src[9];
    const struct ggml_tensor * v_codebook_kv  = dst->src[10];
    const struct ggml_tensor * locs_kv             = dst->src[11];
    const bool eden_disabled_kv = tq_encode_eden_disabled();
    const float * k_bias_kv_ptr     = k_bias_kv     ? (const float *)k_bias_kv->data     : nullptr;
    const float * k_codebook_kv_ptr = (k_codebook_kv && !eden_disabled_kv) ? (const float *)k_codebook_kv->data : nullptr;
    const float * v_codebook_kv_ptr = (v_codebook_kv && !eden_disabled_kv) ? (const float *)v_codebook_kv->data : nullptr;
    const int32_t * locs_kv_ptr     = locs_kv ? (const int32_t *)locs_kv->data : nullptr;

    // s_idx_rms only needed when Path B is active for K or V. Same allocation
    // suffices for both since the K and V kernels share the smem layout.
    const bool path_b_kv = (k_codebook_kv_ptr != nullptr) || (v_codebook_kv_ptr != nullptr);
    size_t smem = (size_t)headDim * sizeof(float)                              // s_k / s_v
                + (size_t)block_size * sizeof(float)                           // s_reduce
                + (size_t)headDim * sizeof(uint8_t)                            // s_idx
                + (path_b_kv ? (size_t)headDim * sizeof(uint8_t) : 0);         // s_idx_rms

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
        k_codebook_kv_ptr,  // EDEN biased scale when codebook provided
        locs_kv_ptr,
        block_size
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
        v_codebook_kv_ptr,
        locs_kv_ptr,
        block_size
    );
}
