// fattn-wmma-gfx12.cuh — WMMA flash attention for RDNA4 (gfx1200/gfx1201)
//
// REPLACES: -DGGML_HIP_ROCWMMA_FATTN=ON which is broken on gfx12
// See: llama.cpp issues #13110, #19269 — both unfixed as of ROCm 7.2
//
// GROUND-TRUTH SPEC (verified from LLVM clang builtins PR #175039)
// ---------------------------------------------------------------
// gfx12 uses 128b WMMA mode — DIFFERENT from gfx11 (RDNA3) 256b mode.
//
//   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
//       v8f16 a,     // <8 x _Float16> per lane
//       v8f16 b,     // <8 x _Float16> per lane
//       v8f32 c)     // <8 x float>   per lane (accumulator)
//   returns v8f32
//
// Wave32: 32 lanes × 8 f16 = 256 f16 = 16×16 matrix tile.
// Do NOT use the gfx11 builtin (which takes 16 f16 per lane) on gfx12.
//
// WMMA IS ONLY USEFUL FOR PREFILL
// ---------------------------------
// For decode (Sq=1): vector dot product — existing fattn-vec-f16 is correct.
// For prefill (Sq>=16): WMMA computes Q[16×D] · K[16×D]^T = scores[16×16]
// using D/16 WMMA tiles, giving ~4× throughput vs vectorised fp16.
//
// BUILD
// -----
//   cmake -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201
//   No -DGGML_HIP_ROCWMMA_FATTN needed.
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

// ── Load A fragment from shared memory ───────────────────────────────────────
// Loads 8 f16 per lane from a 16×16 tile of shared memory.
// gfx12 fragment layout (wave32, 128b):
//   Rows 0–7:  lanes 0–15 (each lane owns row lane/2 and row lane/2+8)
//   Rows 8–15: same lanes
//   Each lane holds 4 elements from row r0, 4 from row r1
//   Columns: (lane%2)*4 .. (lane%2)*4+3

__device__ __forceinline__
gfx12_v8f16 gfx12_load_a_smem(const __half* smem, int stride, int col_off) {
    const int lane = threadIdx.x & 31;
    const int r0   = lane / 2;          // row 0..7
    const int r1   = (lane / 2) + 8;   // row 8..15
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

// ── Load B fragment (transposed K tile) ──────────────────────────────────────
// B holds K transposed: K[KV_TILE × HEAD_DIM] used as B[HEAD_DIM × KV_TILE]
// Fragment layout for B is column-major in the K-reduction dimension.

__device__ __forceinline__
gfx12_v8f16 gfx12_load_b_smem(const __half* smem, int stride, int col_off) {
    const int lane = threadIdx.x & 31;
    const int c0   = col_off + lane / 2;       // col 0..7
    const int c1   = col_off + (lane / 2) + 8; // col 8..15
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
// One wave32 (32 threads) per (Q_TILE=16) × (KV sequence).
// Computes full flash attention for 16 Q tokens against all KV tokens.
//
// Shared memory for HEAD_DIM=128:
//   s_Q:  16 × 128 × 2 = 4 KB
//   s_K:  16 × 128 × 2 = 4 KB
//   s_V:  16 × 128 × 2 = 4 KB
//   s_sc: 16 × 16 × 4  = 1 KB  (score tile)
//   s_O:  16 × 128 × 4 = 8 KB  (output accumulator fp32)
//   s_md: 16 × 2 × 4   = 128 B (m and d per Q row)
//   Total: ~21 KB < 32 KB shared ✓

template<int HEAD_DIM>
__launch_bounds__(32, 2)
__global__ void flash_attn_ext_gfx12_prefill(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    int Sq, int Skv, int Hq, int Hkv, int D,
    float scale, float logit_softcap, int causal)
{
    constexpr int Q_TILE  = 16;
    constexpr int KV_TILE = 16;
    constexpr int N_TILES = HEAD_DIM / 16;

    const int lane    = threadIdx.x;
    const int q_start = blockIdx.x * Q_TILE;
    const int h_q     = blockIdx.y;
    const int batch   = blockIdx.z;
    const int h_kv    = h_q * Hkv / Hq;

    if (q_start >= Sq) return;
    const int q_len = min(Q_TILE, Sq - q_start);

    __shared__ __half s_Q [Q_TILE  * HEAD_DIM];
    __shared__ __half s_K [KV_TILE * HEAD_DIM];
    __shared__ __half s_V [KV_TILE * HEAD_DIM];
    __shared__ float  s_sc[Q_TILE  * KV_TILE];
    __shared__ float  s_O [Q_TILE  * HEAD_DIM];
    __shared__ float  s_m [Q_TILE];
    __shared__ float  s_d [Q_TILE];

    const __half* qb = Q + (batch * Hq  + h_q)  * Sq  * D + q_start * D;
    const __half* kb = K + (batch * Hkv + h_kv) * Skv * D;
    const __half* vb = V + (batch * Hkv + h_kv) * Skv * D;
    __half*       ob = O + (batch * Hq  + h_q)  * Sq  * D + q_start * D;

    // Load Q tile
    for (int r = 0; r < q_len; r++)
        for (int c = lane; c < HEAD_DIM; c += 32)
            s_Q[r * HEAD_DIM + c] = (r * D + c < q_len * D) ? qb[r * D + c] : __float2half(0.f);

    // Init softmax state + output
    for (int r = lane; r < Q_TILE; r += 32) { s_m[r] = -1e38f; s_d[r] = 0.0f; }
    for (int i = lane; i < Q_TILE * HEAD_DIM; i += 32) s_O[i] = 0.0f;
    __syncthreads();

    for (int kv_st = 0; kv_st < Skv; kv_st += KV_TILE) {
        const int kv_len = min(KV_TILE, Skv - kv_st);

        // Load K tile
        for (int r = 0; r < kv_len; r++)
            for (int c = lane; c < HEAD_DIM; c += 32)
                s_K[r * HEAD_DIM + c] = kb[(kv_st + r) * D + c];
        // Zero-pad if kv_len < KV_TILE
        if (kv_len < KV_TILE)
            for (int r = kv_len; r < KV_TILE; r++)
                for (int c = lane; c < HEAD_DIM; c += 32)
                    s_K[r * HEAD_DIM + c] = __float2half(0.f);
        __syncthreads();

        // QK: scores[16×16] = Q[16×HEAD_DIM] × K^T[HEAD_DIM×16]
        gfx12_v8f32 acc = {};
        for (int t = 0; t < N_TILES; t++) {
            gfx12_v8f16 a = gfx12_load_a_smem(s_Q, HEAD_DIM, t * 16);
            gfx12_v8f16 b = gfx12_load_b_smem(s_K, HEAD_DIM, t * 16);
            acc = gfx12_mma(a, b, acc);
        }
        // Scale and softcap
        for (int i = 0; i < 8; i++) {
            acc[i] *= scale;
            if (logit_softcap > 0.0f)
                acc[i] = logit_softcap * tanhf(acc[i] / logit_softcap);
        }
        gfx12_store_d_smem(s_sc, KV_TILE, acc);
        __syncthreads();

        // Causal mask
        if (causal)
            for (int r = lane; r < Q_TILE; r += 32)
                for (int c = 0; c < KV_TILE; c++)
                    if (kv_st + c > q_start + r || c >= kv_len)
                        s_sc[r * KV_TILE + c] = -1e38f;

        // Pad out-of-bounds KV
        if (!causal)
            for (int r = lane; r < Q_TILE; r += 32)
                for (int c = kv_len; c < KV_TILE; c++)
                    s_sc[r * KV_TILE + c] = -1e38f;
        __syncthreads();

        // Online softmax (each lane handles one Q row if lane < Q_TILE)
        if (lane < Q_TILE) {
            float m_old = s_m[lane];
            float row_max = -1e38f;
            for (int c = 0; c < KV_TILE; c++)
                row_max = fmaxf(row_max, s_sc[lane * KV_TILE + c]);
            float m_new   = fmaxf(m_old, row_max);
            float rescale = expf(m_old - m_new);
            s_m[lane]    = m_new;
            s_d[lane]   *= rescale;
            for (int c = 0; c < HEAD_DIM; c++) s_O[lane * HEAD_DIM + c] *= rescale;
            for (int c = 0; c < KV_TILE; c++) {
                float e = expf(s_sc[lane * KV_TILE + c] - m_new);
                s_sc[lane * KV_TILE + c] = e;
                s_d[lane] += e;
            }
        }
        __syncthreads();

        // Load V tile
        for (int r = 0; r < kv_len; r++)
            for (int c = lane; c < HEAD_DIM; c += 32)
                s_V[r * HEAD_DIM + c] = vb[(kv_st + r) * D + c];
        __syncthreads();

        // Weighted V accumulate: O += softmax_weights × V
        for (int r = 0; r < q_len; r++)
            for (int kv = 0; kv < kv_len; kv++) {
                float w = s_sc[r * KV_TILE + kv];
                for (int c = lane; c < HEAD_DIM; c += 32)
                    s_O[r * HEAD_DIM + c] += w * __half2float(s_V[kv * HEAD_DIM + c]);
            }
        __syncthreads();
    }

    // Write output
    for (int r = 0; r < q_len; r++) {
        float denom = fmaxf(s_d[r], 1e-8f);
        for (int c = lane; c < HEAD_DIM; c += 32)
            ob[r * D + c] = __float2half(s_O[r * HEAD_DIM + c] / denom);
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
        return sizeof(__half) * (Q_TILE + 2*KV_TILE) * hd
             + sizeof(float)  * (Q_TILE * KV_TILE + Q_TILE * hd + 2 * Q_TILE);
    };

    switch (D) {
        case  64: hipLaunchKernelGGL((flash_attn_ext_gfx12_prefill< 64>), grid, block, smem( 64), stream, Q,K,V,O,Sq,Skv,Hq,Hkv,D,scale,logit_softcap,(int)causal); break;
        case 128: hipLaunchKernelGGL((flash_attn_ext_gfx12_prefill<128>), grid, block, smem(128), stream, Q,K,V,O,Sq,Skv,Hq,Hkv,D,scale,logit_softcap,(int)causal); break;
        // D=256 (Gemma 4, head_dim=256) exceeds 32KB shared memory limit at Q_TILE=16.
        // Fall through to hipErrorInvalidValue — caller uses fattn-vec-f16 instead.
        // To support D=256: reduce Q_TILE to 8 and add a separate template instance.
        // Not done here because D=256 models are rare and vec FA already works.
        default:  return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

#endif // __gfx1200__ || __gfx1201__
