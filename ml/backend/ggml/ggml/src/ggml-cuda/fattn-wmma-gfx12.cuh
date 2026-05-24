// fattn-wmma-gfx12.cuh — WMMA flash attention for RDNA4 (gfx1200/gfx1201)
// FIXED VERSION v2.0: bank-conflict-free, all-threads utilized, register-accumulated O
//
// REPLACES: -DGGML_HIP_ROCWMMA_FATTN=ON which is broken on gfx12
// See: llama.cpp issues #13110, #19269
//
// GROUND-TRUTH SPEC (verified from LLVM clang builtins PR #175039)
// ---------------------------------------------------------------
// gfx12 uses 128b WMMA mode — DIFFERENT from gfx11 (RDNA3) 256b mode.
//
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
//   v8f16 a, v8f16 b, v8f32 c) -> v8f32
//
// Wave32: 32 lanes x 8 f16 = 256 f16 = 16x16 matrix tile.
// Do NOT use the gfx11 builtin on gfx12.
//
// BUILD
// -----
// cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_GFX12_WMMA=ON
// No -DGGML_HIP_ROCWMMA_FATTN needed.
//
// REQUIRES: ROCm 7.0+ (gfx12 WMMA builtins landed in ROCm 7.x clang)

#pragma once

#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx1200__) || defined(__gfx1201__))

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ── Vector types for gfx12 WMMA ──────────────────────────────────────────────
typedef _Float16 gfx12_v8f16 __attribute__((ext_vector_type(8)));
typedef float    gfx12_v8f32 __attribute__((ext_vector_type(8)));

// ── Core MMA call ─────────────────────────────────────────────────────────────
__device__ __forceinline__
gfx12_v8f32 gfx12_mma(gfx12_v8f16 a, gfx12_v8f16 b, gfx12_v8f32 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// ── Load A fragment from shared memory (row-major Q tile) ────────────────────
// gfx12 fragment layout (wave32, 128b mode):
//   Rows 0-7:  lanes 0-15 (each lane owns row lane/2 and lane/2+8)
//   Rows 8-15: same lanes
//   Each lane holds 4 elements from row r0, 4 from row r1
//   Columns: (lane%2)*4 .. (lane%2)*4+3
__device__ __forceinline__
gfx12_v8f16 gfx12_load_a_smem(const __half* smem, int stride, int col_off) {
    const int lane = threadIdx.x & 31;
    const int r0   = lane / 2;          // row 0..7
    const int r1   = (lane / 2) + 8;    // row 8..15
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

// ── Load B fragment (K transposed / V transposed layout) ──────────────────────
// Reads column-major-like tiles from row-major storage for GEMM B operand.
__device__ __forceinline__
gfx12_v8f16 gfx12_load_b_smem(const __half* smem, int stride, int col_off) {
    const int lane = threadIdx.x & 31;
    const int c0   = col_off + lane / 2;        // col 0..7
    const int c1   = col_off + (lane / 2) + 8;  // col 8..15
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

// ── Store accumulator to shared memory ───────────────────────────────────────
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

// ── Flash attention prefill kernel for gfx12 ─────────────────────────────────
// One wave32 (32 threads) per (Q_TILE=16) x (1 query head).
// Computes full flash attention for 16 Q tokens against all KV tokens.
//
// SHARED MEMORY (HEAD_DIM=128, padded K/V):
//   s_Q:     16 x 128 x 2         = 4 KB
//   s_K:     16 x 130 x 2 (pad)   = 4.06 KB   // +2 avoids bank conflicts
//   s_V:     16 x 130 x 2 (pad)   = 4.06 KB
//   s_sc:    16 x 16 x 4          = 1 KB
//   s_m:     16 x 4               = 64 B
//   s_d:     16 x 4               = 64 B
//   s_rescale: 16 x 4             = 64 B
// Total shared: ~13.3 KB  (no s_O — accumulated in registers!)
//
// REGISTER USAGE per thread (D=128):
//   o_reg[4][16] = 64 floats = 256 bytes  (4 columns x 16 rows)
//   + temporaries ~64 bytes
//   Total ~320 bytes / thread — well under RDNA4 VGPR budget.

template <int HEAD_DIM>
__launch_bounds__(32, 2)
__global__ void flash_attn_ext_gfx12_prefill(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int Sq, int Skv, int Hq, int Hkv, int D,
    float scale, float logit_softcap, int causal)
{
    constexpr int Q_TILE  = 16;
    constexpr int KV_TILE = 16;
    constexpr int N_TILES = HEAD_DIM / 16;
    constexpr int K_PAD   = HEAD_DIM + 2;      // padded stride for K/V
    constexpr int COLS_PER_THREAD = HEAD_DIM / 32;  // columns each thread owns

    const int lane    = threadIdx.x;
    const int q_start = blockIdx.x * Q_TILE;
    const int h_q     = blockIdx.y;
    const int batch   = blockIdx.z;
    const int h_kv    = h_q * Hkv / Hq;

    if (q_start >= Sq) return;
    const int q_len = min(Q_TILE, Sq - q_start);

    // ── Shared memory ──────────────────────────────────────────────────────
    __shared__ __half  s_Q [Q_TILE * HEAD_DIM];
    __shared__ __half  s_K [KV_TILE * K_PAD];
    __shared__ __half  s_V [KV_TILE * K_PAD];
    __shared__ float   s_sc[Q_TILE * KV_TILE];
    __shared__ float   s_m [Q_TILE];
    __shared__ float   s_d [Q_TILE];
    __shared__ float   s_rescale[Q_TILE];

    // ── Register accumulators for O (bank-conflict-free) ───────────────────
    // Each thread accumulates COLS_PER_THREAD columns across all Q rows.
    float o_reg[COLS_PER_THREAD][Q_TILE];
    #pragma unroll
    for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
        #pragma unroll
        for (int r = 0; r < Q_TILE; r++) {
            o_reg[ci][r] = 0.0f;
        }
    }

    // ── Base pointers ─────────────────────────────────────────────────────
    const __half* qb = Q + (batch * Hq + h_q) * Sq * D + q_start * D;
    const __half* kb = K + (batch * Hkv + h_kv) * Skv * D;
    const __half* vb = V + (batch * Hkv + h_kv) * Skv * D;
    __half* ob = O + (batch * Hq + h_q) * Sq * D + q_start * D;

    // ── Load Q tile ────────────────────────────────────────────────────────
    for (int r = 0; r < q_len; r++) {
        for (int c = lane; c < HEAD_DIM; c += 32) {
            s_Q[r * HEAD_DIM + c] = qb[r * D + c];
        }
    }
    // Zero-pad Q rows beyond q_len (needed for clean WMMA)
    for (int r = q_len; r < Q_TILE; r++) {
        for (int c = lane; c < HEAD_DIM; c += 32) {
            s_Q[r * HEAD_DIM + c] = __float2half(0.f);
        }
    }

    // Init softmax state
    for (int r = lane; r < Q_TILE; r += 32) {
        s_m[r] = -1e38f;
        s_d[r] = 0.0f;
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════════════
    // MAIN LOOP over KV sequence
    // ═══════════════════════════════════════════════════════════════════════
    for (int kv_st = 0; kv_st < Skv; kv_st += KV_TILE) {
        const int kv_len = min(KV_TILE, Skv - kv_st);

        // ── Load K tile (padded, bank-conflict-free) ─────────────────────────
        for (int r = 0; r < kv_len; r++) {
            for (int c = lane; c < HEAD_DIM; c += 32) {
                s_K[r * K_PAD + c] = kb[(kv_st + r) * D + c];
            }
        }
        // Zero-pad K rows beyond kv_len
        for (int r = kv_len; r < KV_TILE; r++) {
            for (int c = lane; c < K_PAD; c += 32) {
                s_K[r * K_PAD + c] = __float2half(0.f);
            }
        }
        __syncthreads();

        // ── QK^T via WMMA ────────────────────────────────────────────────────
        gfx12_v8f32 acc = {};
        #pragma unroll
        for (int t = 0; t < N_TILES; t++) {
            gfx12_v8f16 a = gfx12_load_a_smem(s_Q, HEAD_DIM, t * 16);
            gfx12_v8f16 b = gfx12_load_b_smem(s_K, K_PAD, t * 16);
            acc = gfx12_mma(a, b, acc);
        }
        // Scale + softcap
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[i] *= scale;
            if (logit_softcap > 0.0f) {
                acc[i] = logit_softcap * tanhf(acc[i] / logit_softcap);
            }
        }
        gfx12_store_d_smem(s_sc, KV_TILE, acc);
        __syncthreads();

        // ── Causal + padding mask (ALL 32 threads) ───────────────────────────
        for (int idx = lane; idx < Q_TILE * KV_TILE; idx += 32) {
            const int r = idx / KV_TILE;
            const int c = idx % KV_TILE;
            bool mask = (c >= kv_len);
            if (causal) {
                mask = mask || (kv_st + c > q_start + r);
            }
            if (mask) {
                s_sc[idx] = -1e38f;
            }
        }
        __syncthreads();

        // ── Online softmax (lanes 0-15 handle one row each) ────────────────
        if (lane < Q_TILE) {
            const float m_old = s_m[lane];
            float row_max = -1e38f;
            #pragma unroll
            for (int c = 0; c < KV_TILE; c++) {
                row_max = fmaxf(row_max, s_sc[lane * KV_TILE + c]);
            }
            const float m_new = fmaxf(m_old, row_max);
            const float rescale = expf(m_old - m_new);
            s_m[lane] = m_new;
            s_d[lane] *= rescale;
            s_rescale[lane] = rescale;
            #pragma unroll
            for (int c = 0; c < KV_TILE; c++) {
                float e = expf(s_sc[lane * KV_TILE + c] - m_new);
                s_sc[lane * KV_TILE + c] = e;
                s_d[lane] += e;
            }
        }
        __syncthreads();

        // ── FIXED: Rescale O registers directly (no shared memory RMW) ───────
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
            #pragma unroll
            for (int r = 0; r < Q_TILE; r++) {
                o_reg[ci][r] *= s_rescale[r];
            }
        }

        // ── Load V tile (padded) ───────────────────────────────────────────
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

        // ── FIXED: P x V into registers (bank-conflict-free) ─────────────────
        // Each thread handles COLS_PER_THREAD columns.
        // s_V is padded so consecutive KV rows hit different banks.
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
            const int col = ci * 32 + lane;   // global column index this thread owns
            #pragma unroll
            for (int r = 0; r < q_len; r++) {
                float acc_pv = 0.0f;
                #pragma unroll
                for (int kv = 0; kv < KV_TILE; kv++) {
                    acc_pv += s_sc[r * KV_TILE + kv] * __half2float(s_V[kv * K_PAD + col]);
                }
                o_reg[ci][r] += acc_pv;
            }
        }
        // Note: we don't need __syncthreads() here because o_reg is private.
        // Next iteration's s_Q/s_K/s_V loads don't overlap with our register use.
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Write output: normalize and store
    // ═══════════════════════════════════════════════════════════════════════
    for (int r = 0; r < q_len; r++) {
        const float denom = fmaxf(s_d[r], 1e-8f);
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
            const int col = ci * 32 + lane;
            ob[r * D + col] = __float2half(o_reg[ci][r] / denom);
        }
    }
}

// ── Launcher ──────────────────────────────────────────────────────────────────
// Use for prefill (Sq >= 16). For decode (Sq=1), use existing vec FA.
inline hipError_t launch_flash_attn_ext_gfx12(
    hipStream_t stream,
    const __half* Q, const __half* K, const __half* V, __half* O,
    int Sq, int Skv, int Hq, int Hkv, int D, int B,
    float scale, float logit_softcap, bool causal)
{
    constexpr int Q_TILE  = 16;
    constexpr int KV_TILE = 16;

    dim3 grid((Sq + Q_TILE - 1) / Q_TILE, Hq, B);
    dim3 block(32);

    auto smem = [&](int hd) -> size_t {
        const int k_pad = hd + 2;
        return sizeof(__half) * Q_TILE * hd          // s_Q
             + sizeof(__half) * KV_TILE * k_pad * 2  // s_K + s_V (padded)
             + sizeof(float) * (Q_TILE * KV_TILE     // s_sc
                                 + 3 * Q_TILE);       // s_m + s_d + s_rescale
    };

    switch (D) {
        case 64:  hipLaunchKernelGGL((flash_attn_ext_gfx12_prefill< 64>), grid, block, smem( 64), stream, Q,K,V,O,Sq,Skv,Hq,Hkv,D,scale,logit_softcap,(int)causal); break;
        case 128: hipLaunchKernelGGL((flash_attn_ext_gfx12_prefill<128>), grid, block, smem(128), stream, Q,K,V,O,Sq,Skv,Hq,Hkv,D,scale,logit_softcap,(int)causal); break;
        // D=256: Q_TILE must drop to 8 to fit shared memory. Not implemented here.
        default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

#endif // __gfx1200__ || __gfx1201__
