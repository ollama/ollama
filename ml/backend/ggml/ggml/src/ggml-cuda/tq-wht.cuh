#pragma once

// apply_shs_wht — symmetric randomised Walsh-Hadamard transform F(x)=S·H·S·x/√n.
// F is self-inverse: F(F(x)) = x.  Used for K/V rotation in TurboQuant.
//
// s_x:     [n] f32 in shared memory (already loaded by the caller)
// signs:   [n] f32 ±1, global memory (same vector used for encode and decode)
// n:       headDim, must be a power of 2
// tid:     threadIdx.x
// nthreads: blockDim.x
//
// Template parameter kTrailingSync: when false, the final __syncthreads() is
// omitted.  Safe only when every thread reads only its own position(s) of s_x
// after return (e.g. tq_wht_kernel write-back loop).  Encode callers must
// leave this at the default true so subsequent cross-thread reads are safe.
template<bool kTrailingSync = true>
static __device__ __forceinline__ void apply_shs_wht(
    float * __restrict__ s_x,
    const float * __restrict__ signs,
    int n, int tid, int nthreads)
{
    // S: x[i] *= signs[i]
    for (int i = tid; i < n; i += nthreads) {
        s_x[i] *= signs[i];
    }
    __syncthreads();

    if (nthreads >= n) {
        // Fast path: one thread per element.
        // For strides < warp size (32), threads tid and tid^stride share a warp
        // → exchange via warp shuffle, no shared-memory round-trip needed.
        float val = s_x[tid];

        for (int stride = 1; stride < n && stride < 32; stride <<= 1) {
            float nbr = __shfl_xor_sync(0xFFFFFFFF, val, stride);
            val = (tid & stride) ? (nbr - val) : (val + nbr);
        }

        // For strides >= 32, threads may be in different warps: use shared memory.
        // Two syncs per iteration: (1) ensure writes visible before reads,
        // (2) ensure reads complete before next stride's writes (prevents RAW race
        //     where a faster warp overwrites s_x[tid^stride] before a slower warp
        //     reads it on the previous stride iteration).
        for (int stride = 32; stride < n; stride <<= 1) {
            s_x[tid] = val;
            __syncthreads();
            float other = s_x[tid ^ stride];
            val = (tid & stride) ? (other - val) : (val + other);
            __syncthreads();
        }

        float scale = rsqrtf((float)n);
        s_x[tid] = val * signs[tid] * scale;
    } else {
        // General path: nthreads < n, each thread covers multiple butterfly pairs.
        for (int stride = 1; stride < n; stride <<= 1) {
            for (int i = tid; i < n / 2; i += nthreads) {
                int lo = (i / stride) * (2 * stride) + (i % stride);
                int hi = lo + stride;
                float a = s_x[lo];
                float b = s_x[hi];
                s_x[lo] = a + b;
                s_x[hi] = a - b;
            }
            __syncthreads();
        }

        // S and normalise: x[i] *= signs[i] / sqrt(n)
        float scale = rsqrtf((float)n);
        for (int i = tid; i < n; i += nthreads) {
            s_x[i] *= signs[i] * scale;
        }
    }

    if (kTrailingSync) { __syncthreads(); }
}
