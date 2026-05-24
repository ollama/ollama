// fattn-wmma-gfx12-gqa.cuh — GQA-BATCHED WMMA flash attention for RDNA4
// FUTURE VERSION v3.0: processes 2-4 query heads per block
//
// THIS IS A PREVIEW / REFERENCE IMPLEMENTATION.
// Integrate after the v2.0 fixes are proven stable.
//
// GQA batching: For gqa_ratio=4 (4 Q heads per KV head), we process
// Q_TILE x (gqa_batch) query heads per block, loading KV cache once.
// This amortizes KV cache bandwidth by gqa_batch factor.
//
// Expected gain on GQA models (Llama-3, Qwen2.5, Mistral):
//   +20-40% prefill speedup over v2.0 single-head version.

#pragma once

#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx1200__) || defined(__gfx1201__))

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 gfx12_v8f16 __attribute__((ext_vector_type(8)));
typedef float    gfx12_v8f32 __attribute__((ext_vector_type(8)));

__device__ __forceinline__
gfx12_v8f32 gfx12_mma(gfx12_v8f16 a, gfx12_v8f16 b, gfx12_v8f32 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__
gfx12_v8f16 gfx12_load_a_smem(const __half* smem, int stride, int col_off) {
    const int lane = threadIdx.x & 31;
    const int r0   = lane / 2;
    const int r1   = (lane / 2) + 8;
    const int c    = col_off + (lane % 2) * 4;
    gfx12_v8f16 f;
    f[0] = (_Float16)smem[r0 * stride + c + 0];
    f[1] = (_Float16)smem[r0 * stride + c + 1];
    f[2] = (_Float16)smem[r0 * stride + c + 2];
    f[3] = (_Float16)smem[r0 * stride + c + 3];
    f[4] = (_Float16)smem[r1 * stride + c + 0];
    f[5] = (_Float16)smem[r1 * stride + c + 1];
    f[6] = (_Float16)smem[r1 * stride + c + 2];
    f[7] = (_Float16)smem[r1 * stride + c + 3];
    return f;
}

__device__ __forceinline__
gfx12_v8f16 gfx12_load_b_smem(const __half* smem, int stride, int col_off) {
    const int lane = threadIdx.x & 31;
    const int c0   = col_off + lane / 2;
    const int c1   = col_off + (lane / 2) + 8;
    const int r    = (lane % 2) * 4;
    gfx12_v8f16 f;
    f[0] = (_Float16)smem[(r + 0) * stride + c0];
    f[1] = (_Float16)smem[(r + 1) * stride + c0];
    f[2] = (_Float16)smem[(r + 2) * stride + c0];
    f[3] = (_Float16)smem[(r + 3) * stride + c0];
    f[4] = (_Float16)smem[(r + 0) * stride + c1];
    f[5] = (_Float16)smem[(r + 1) * stride + c1];
    f[6] = (_Float16)smem[(r + 2) * stride + c1];
    f[7] = (_Float16)smem[(r + 3) * stride + c1];
    return f;
}

__device__ __forceinline__
void gfx12_store_d_smem(float* smem, int stride, gfx12_v8f32 d) {
    const int lane = threadIdx.x & 31;
    const int r0   = lane / 2;
    const int r1   = (lane / 2) + 8;
    const int c    = (lane % 2) * 4;
    smem[r0 * stride + c + 0] = d[0];
    smem[r0 * stride + c + 1] = d[1];
    smem[r0 * stride + c + 2] = d[2];
    smem[r0 * stride + c + 3] = d[3];
    smem[r1 * stride + c + 0] = d[4];
    smem[r1 * stride + c + 1] = d[5];
    smem[r1 * stride + c + 2] = d[6];
    smem[r1 * stride + c + 3] = d[7];
}

// ── GQA-BATCHED kernel ───────────────────────────────────────────────────────
// Processes GQA_BATCH query heads per block, sharing KV cache loads.
//
// Shared memory scales with GQA_BATCH:
//   s_Q:  GQA_BATCH x 16 x HEAD_DIM x 2 bytes
//   s_K:  16 x (HEAD_DIM+2) x 2 bytes
//   s_V:  16 x (HEAD_DIM+2) x 2 bytes
//   s_sc: GQA_BATCH x 16 x 16 x 4 bytes
//   s_m/s_d/s_rescale: GQA_BATCH x 16 x 4 bytes each
//
// For GQA_BATCH=2, HEAD_DIM=128:
//   s_Q:  8 KB, s_K: 4 KB, s_V: 4 KB, s_sc: 2 KB, misc: ~0.5 KB
//   Total: ~18.5 KB < 32 KB ✓
//
// For GQA_BATCH=4:
//   Total: ~33 KB > 32 KB ✗ (reduce Q_TILE to 8 or use 64KB LDS if available)

template <int HEAD_DIM, int GQA_BATCH>
__launch_bounds__(32, 2)
__global__ void flash_attn_ext_gfx12_prefill_gqa(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int Sq, int Skv, int Hq, int Hkv, int D, int B,
    float scale, float logit_softcap, int causal)
{
    constexpr int Q_TILE  = 16;
    constexpr int KV_TILE = 16;
    constexpr int N_TILES = HEAD_DIM / 16;
    constexpr int K_PAD   = HEAD_DIM + 2;
    constexpr int COLS_PER_THREAD = HEAD_DIM / 32;

    const int lane      = threadIdx.x;
    const int q_start   = blockIdx.x * Q_TILE;
    const int h_q_base  = blockIdx.y * GQA_BATCH;   // first query head this block handles
    const int batch     = blockIdx.z;
    const int h_kv      = h_q_base * Hkv / Hq;       // shared KV head

    if (q_start >= Sq) return;
    const int q_len = min(Q_TILE, Sq - q_start);

    // ── Shared memory: Q and scores are replicated per GQA_BATCH ───────────
    __shared__ __half  s_Q [GQA_BATCH * Q_TILE * HEAD_DIM];
    __shared__ __half  s_K [KV_TILE * K_PAD];
    __shared__ __half  s_V [KV_TILE * K_PAD];
    __shared__ float   s_sc[GQA_BATCH * Q_TILE * KV_TILE];
    __shared__ float   s_m [GQA_BATCH * Q_TILE];
    __shared__ float   s_d [GQA_BATCH * Q_TILE];
    __shared__ float   s_rescale[GQA_BATCH * Q_TILE];

    // ── Register accumulators: one set per GQA_BATCH head ──────────────────
    float o_reg[GQA_BATCH][COLS_PER_THREAD][Q_TILE];
    #pragma unroll
    for (int g = 0; g < GQA_BATCH; g++) {
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
            #pragma unroll
            for (int r = 0; r < Q_TILE; r++) {
                o_reg[g][ci][r] = 0.0f;
            }
        }
    }

    // ── Load all GQA_BATCH Q tiles ─────────────────────────────────────────
    #pragma unroll
    for (int g = 0; g < GQA_BATCH; g++) {
        const int h_q = h_q_base + g;
        if (h_q >= Hq) break;  // handle tail
        const __half* qb = Q + (batch * Hq + h_q) * Sq * D + q_start * D;
        for (int r = 0; r < q_len; r++) {
            for (int c = lane; c < HEAD_DIM; c += 32) {
                s_Q[(g * Q_TILE + r) * HEAD_DIM + c] = qb[r * D + c];
            }
        }
        for (int r = q_len; r < Q_TILE; r++) {
            for (int c = lane; c < HEAD_DIM; c += 32) {
                s_Q[(g * Q_TILE + r) * HEAD_DIM + c] = __float2half(0.f);
            }
        }
        for (int r = lane; r < Q_TILE; r += 32) {
            s_m[g * Q_TILE + r] = -1e38f;
            s_d[g * Q_TILE + r] = 0.0f;
        }
    }
    __syncthreads();

    // ── Base KV pointer (shared across GQA_BATCH) ──────────────────────────
    const __half* kb = K + (batch * Hkv + h_kv) * Skv * D;
    const __half* vb = V + (batch * Hkv + h_kv) * Skv * D;

    // ═══════════════════════════════════════════════════════════════════════
    // MAIN LOOP over KV sequence (shared across GQA_BATCH)
    // ═══════════════════════════════════════════════════════════════════════
    for (int kv_st = 0; kv_st < Skv; kv_st += KV_TILE) {
        const int kv_len = min(KV_TILE, Skv - kv_st);

        // Load K (once, shared)
        for (int r = 0; r < kv_len; r++) {
            for (int c = lane; c < HEAD_DIM; c += 32) {
                s_K[r * K_PAD + c] = kb[(kv_st + r) * D + c];
            }
        }
        for (int r = kv_len; r < KV_TILE; r++) {
            for (int c = lane; c < K_PAD; c += 32) {
                s_K[r * K_PAD + c] = __float2half(0.f);
            }
        }
        __syncthreads();

        // Process each GQA_BATCH head
        #pragma unroll
        for (int g = 0; g < GQA_BATCH; g++) {
            const int h_q = h_q_base + g;
            if (h_q >= Hq) continue;

            __half* ob = O + (batch * Hq + h_q) * Sq * D + q_start * D;

            // QK^T via WMMA
            gfx12_v8f32 acc = {};
            #pragma unroll
            for (int t = 0; t < N_TILES; t++) {
                gfx12_v8f16 a = gfx12_load_a_smem(
                    s_Q + g * Q_TILE * HEAD_DIM, HEAD_DIM, t * 16);
                gfx12_v8f16 b = gfx12_load_b_smem(s_K, K_PAD, t * 16);
                acc = gfx12_mma(a, b, acc);
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                acc[i] *= scale;
                if (logit_softcap > 0.0f) {
                    acc[i] = logit_softcap * tanhf(acc[i] / logit_softcap);
                }
            }
            gfx12_store_d_smem(s_sc + g * Q_TILE * KV_TILE, KV_TILE, acc);
        }
        __syncthreads();

        // Masking (all threads, all GQA_BATCH)
        const int total_sc = GQA_BATCH * Q_TILE * KV_TILE;
        for (int idx = lane; idx < total_sc; idx += 32) {
            const int g = idx / (Q_TILE * KV_TILE);
            const int rem = idx % (Q_TILE * KV_TILE);
            const int r = rem / KV_TILE;
            const int c = rem % KV_TILE;
            bool mask = (c >= kv_len);
            if (causal) {
                mask = mask || (kv_st + c > q_start + r);
            }
            if (mask) {
                s_sc[idx] = -1e38f;
            }
        }
        __syncthreads();

        // Online softmax per GQA_BATCH head
        #pragma unroll
        for (int g = 0; g < GQA_BATCH; g++) {
            if (lane < Q_TILE) {
                const int base = g * Q_TILE + lane;
                const float m_old = s_m[base];
                float row_max = -1e38f;
                #pragma unroll
                for (int c = 0; c < KV_TILE; c++) {
                    row_max = fmaxf(row_max, s_sc[base * KV_TILE + c]);
                }
                const float m_new = fmaxf(m_old, row_max);
                const float rescale = expf(m_old - m_new);
                s_m[base] = m_new;
                s_d[base] *= rescale;
                s_rescale[base] = rescale;
                #pragma unroll
                for (int c = 0; c < KV_TILE; c++) {
                    float e = expf(s_sc[base * KV_TILE + c] - m_new);
                    s_sc[base * KV_TILE + c] = e;
                    s_d[base] += e;
                }
            }
        }
        __syncthreads();

        // Rescale O registers
        #pragma unroll
        for (int g = 0; g < GQA_BATCH; g++) {
            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                #pragma unroll
                for (int r = 0; r < Q_TILE; r++) {
                    o_reg[g][ci][r] *= s_rescale[g * Q_TILE + r];
                }
            }
        }

        // Load V (once, shared)
        for (int r = 0; r < kv_len; r++) {
            for (int c = lane; c < HEAD_DIM; c += 32) {
                s_V[r * K_PAD + c] = vb[(kv_st + r) * D + c];
            }
        }
        for (int r = kv_len; r < KV_TILE; r++) {
            for (int c = lane; c < K_PAD; c += 32) {
                s_V[r * K_PAD + c] = __float2half(0.f);
            }
        }
        __syncthreads();

        // P x V per GQA_BATCH head
        #pragma unroll
        for (int g = 0; g < GQA_BATCH; g++) {
            const int h_q = h_q_base + g;
            if (h_q >= Hq) continue;
            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                const int col = ci * 32 + lane;
                #pragma unroll
                for (int r = 0; r < q_len; r++) {
                    float acc_pv = 0.0f;
                    #pragma unroll
                    for (int kv = 0; kv < KV_TILE; kv++) {
                        acc_pv += s_sc[(g * Q_TILE + r) * KV_TILE + kv]
                                * __half2float(s_V[kv * K_PAD + col]);
                    }
                    o_reg[g][ci][r] += acc_pv;
                }
            }
        }
    }

    // Write output for all GQA_BATCH heads
    #pragma unroll
    for (int g = 0; g < GQA_BATCH; g++) {
        const int h_q = h_q_base + g;
        if (h_q >= Hq) break;
        __half* ob = O + (batch * Hq + h_q) * Sq * D + q_start * D;
        for (int r = 0; r < q_len; r++) {
            const float denom = fmaxf(s_d[g * Q_TILE + r], 1e-8f);
            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                const int col = ci * 32 + lane;
                ob[r * D + col] = __float2half(o_reg[g][ci][r] / denom);
            }
        }
    }
}

// ── GQA launcher ─────────────────────────────────────────────────────────────
// Automatically selects GQA_BATCH based on gqa_ratio and shared memory.
inline hipError_t launch_flash_attn_ext_gfx12_gqa(
    hipStream_t stream,
    const __half* Q, const __half* K, const __half* V, __half* O,
    int Sq, int Skv, int Hq, int Hkv, int D, int B,
    float scale, float logit_softcap, bool causal)
{
    constexpr int Q_TILE = 16;
    const int gqa_ratio = Hq / Hkv;

    dim3 grid((Sq + Q_TILE - 1) / Q_TILE, (Hq + 1) / 2, B);
    dim3 block(32);

    auto smem_2 = [&](int hd) -> size_t {
        const int k_pad = hd + 2;
        return 2 * sizeof(__half) * Q_TILE * hd
             + sizeof(__half) * KV_TILE * k_pad * 2
             + sizeof(float) * (2 * Q_TILE * KV_TILE + 3 * 2 * Q_TILE);
    };

    // For now, only GQA_BATCH=2 is supported (fits in 32KB for D<=128)
    // GQA_BATCH=4 requires Q_TILE=8 or 64KB LDS.
    switch (D) {
        case 64:  hipLaunchKernelGGL((flash_attn_ext_gfx12_prefill_gqa< 64,2>), grid, block, smem_2( 64), stream, Q,K,V,O,Sq,Skv,Hq,Hkv,D,B,scale,logit_softcap,(int)causal); break;
        case 128: hipLaunchKernelGGL((flash_attn_ext_gfx12_prefill_gqa<128,2>), grid, block, smem_2(128), stream, Q,K,V,O,Sq,Skv,Hq,Hkv,D,B,scale,logit_softcap,(int)causal); break;
        default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

#endif // __gfx1200__ || __gfx1201__
