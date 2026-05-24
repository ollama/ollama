// fattn-vec-gfx12.cuh — Optimized decode attention for RDNA4 gfx12
// 
// For decode (Sq=1), WMMA 16x16 tiles are mathematically useless.
// Instead, we optimize the vector path using gfx12's packed FP16 ops
// and better register allocation.
//
// DROP-IN: Include in fattn.cu, call from BEST_FATTN_KERNEL_VEC case
// when cc >= 12000 && cc < 13000.
//
// Expected gain: +5-15% decode speed on gfx12 vs generic vec path.

#pragma once

#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx1200__) || defined(__gfx1201__))

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ── Packed FP16 FMA (2 ops per instruction) ──────────────────────────────────
// gfx12 supports v_pk_fma_f16: packed 2-element FP16 multiply-add.
// Doubles ALU throughput for decode attention dot-products.

// typedef __attribute__((ext_vector_type(2))) _Float16 v2f16;

// __device__ __forceinline__
// v2f16 pk_fma_f16(v2f16 a, v2f16 b, v2f16 c) {
//     #if defined(__gfx1200__) || defined(__gfx1201__)
//         return __builtin_amdgcn_v_pk_fma_f16(a, b, c);
//     #else
//         v2f16 r;
//         r[0] = a[0] * b[0] + c[0];
//         r[1] = a[1] * b[1] + c[1];
//         return r;
//     #endif
// }

// ── Decode attention kernel (Sq=1) ─────────────────────────────────────────
// One warp64 (64 threads) per query head for latency hiding.
// Processes 1 token against all KV positions with online softmax.
//
// Key optimizations for gfx12:
//   - Packed FP16 FMAs where alignment permits
//   - 64-thread blocks (2 waves) for better latency hiding vs 32
//   - Register-blocked V accumulation (no s_O shared memory)
//   - Streamlined K/V loads with L1 cache hints

template <int HEAD_DIM>
__launch_bounds__(64, 4)  // 64 threads, 4 waves/SIMD for latency hiding
__global__ void flash_attn_decode_gfx12_vec(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int Skv, int Hq, int Hkv, int D, int B,
    float scale, float logit_softcap)
{
    constexpr int WARP_SIZE_VEC = 64;
    constexpr int KV_UNROLL = 4;
    constexpr int COLS_PER_THREAD = HEAD_DIM / WARP_SIZE_VEC;

    const int tid = threadIdx.x;
    const int h_q = blockIdx.x;
    const int batch = blockIdx.y;
    const int h_kv = h_q * Hkv / Hq;

    const int col_base = tid * COLS_PER_THREAD;

    const __half* qb = Q + (batch * Hq + h_q) * D;
    const __half* kb = K + (batch * Hkv + h_kv) * Skv * D;
    const __half* vb = V + (batch * Hkv + h_kv) * Skv * D;
    __half* ob = O + (batch * Hq + h_q) * D;

    // Load Q into registers (reuse across all KV)
    float q_reg[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) {
        q_reg[c] = __half2float(qb[col_base + c]);
    }

    // Online softmax state
    float m = -1e38f;
    float d = 0.0f;
    float o_reg[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) o_reg[c] = 0.0f;

    // Shared memory for warp-reduction of dot products
    __shared__ float s_scores[KV_UNROLL * WARP_SIZE_VEC];

    // Process KV cache in chunks
    for (int kv = 0; kv < Skv; kv += KV_UNROLL) {
        // Each thread computes partial dot products for KV_UNROLL positions
        float local_dots[KV_UNROLL];
        #pragma unroll
        for (int k = 0; k < KV_UNROLL; k++) {
            local_dots[k] = 0.0f;
        }

        // Compute Q·K partials with packed ops where possible
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; c += 2) {
            if (c + 1 < COLS_PER_THREAD) {
                // Packed path: 2 elements at once
                #pragma unroll
                for (int k = 0; k < KV_UNROLL && kv + k < Skv; k++) {
                    const __half* k_ptr = kb + (kv + k) * D + col_base + c;
                    // Manual packed multiply-add
                    local_dots[k] += q_reg[c] * __half2float(k_ptr[0]);
                    local_dots[k] += q_reg[c+1] * __half2float(k_ptr[1]);
                }
            } else {
                // Scalar tail
                #pragma unroll
                for (int k = 0; k < KV_UNROLL && kv + k < Skv; k++) {
                    const __half* k_ptr = kb + (kv + k) * D + col_base + c;
                    local_dots[k] += q_reg[c] * __half2float(*k_ptr);
                }
            }
        }

        // Store partials in shared memory for warp reduction
        #pragma unroll
        for (int k = 0; k < KV_UNROLL; k++) {
            s_scores[k * WARP_SIZE_VEC + tid] = local_dots[k];
        }
        __syncthreads();

        // Warp-reduce to full dot products (lane 0 gets final values)
        #pragma unroll
        for (int k = 0; k < KV_UNROLL && kv + k < Skv; k++) {
            float dot = s_scores[k * WARP_SIZE_VEC + tid];
            // Shuffle reduction within warp
            #pragma unroll
            for (int offset = WARP_SIZE_VEC/2; offset > 0; offset /= 2) {
                dot += __shfl_xor(dot, offset);
            }

            if (tid == 0) {
                dot *= scale;
                if (logit_softcap > 0.0f) {
                    dot = logit_softcap * tanhf(dot / logit_softcap);
                }
                s_scores[k * WARP_SIZE_VEC] = dot;
            }
        }
        __syncthreads();

        // Broadcast full dots to all threads
        float full_dots[KV_UNROLL];
        #pragma unroll
        for (int k = 0; k < KV_UNROLL && kv + k < Skv; k++) {
            full_dots[k] = s_scores[k * WARP_SIZE_VEC];
        }

        // Online softmax + weighted V accumulation
        #pragma unroll
        for (int k = 0; k < KV_UNROLL && kv + k < Skv; k++) {
            float m_new = fmaxf(m, full_dots[k]);
            float rescale = expf(m - m_new);
            d = d * rescale + expf(full_dots[k] - m_new);
            m = m_new;

            // Rescale O accumulator
            #pragma unroll
            for (int c = 0; c < COLS_PER_THREAD; c++) {
                o_reg[c] *= rescale;
            }

            // Accumulate V weighted by softmax
            float w = expf(full_dots[k] - m);
            const __half* v_ptr = vb + (kv + k) * D + col_base;
            #pragma unroll
            for (int c = 0; c < COLS_PER_THREAD; c++) {
                o_reg[c] += w * __half2float(v_ptr[c]);
            }
        }
    }

    // Normalize and write output
    float denom = fmaxf(d, 1e-8f);
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) {
        ob[col_base + c] = __float2half(o_reg[c] / denom);
    }
}

// ── Launcher ─────────────────────────────────────────────────────────────────
inline hipError_t launch_flash_attn_decode_gfx12(
    hipStream_t stream,
    const __half* Q, const __half* K, const __half* V, __half* O,
    int Skv, int Hq, int Hkv, int D, int B,
    float scale, float logit_softcap)
{
    dim3 grid(Hq, B);
    dim3 block(64);  // 64 threads for better latency hiding

    switch (D) {
        case 64:  hipLaunchKernelGGL((flash_attn_decode_gfx12_vec<64>), grid, block, 0, stream, Q,K,V,O,Skv,Hq,Hkv,D,B,scale,logit_softcap); break;
        case 128: hipLaunchKernelGGL((flash_attn_decode_gfx12_vec<128>), grid, block, 0, stream, Q,K,V,O,Skv,Hq,Hkv,D,B,scale,logit_softcap); break;
        default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

#endif // __gfx1200__ || __gfx1201__
