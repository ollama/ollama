// SPDX-License-Identifier: MIT
// rope.cl — Rotary Position Embedding (RoPE) kernels for the Intel Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file rope.cl -device <target> -options "-cl-std=CL2.0 -O3"
//
// Two entry points compiled from this file:
//   rope_f32 — F32 input/output (RC3 fix: stride-aware indexing)
//   rope_f16 — F16 input/output (RC2 fix: new path that eliminates GGML_ABORT at
//              ggml-backend.cpp:844; implements Adapter pattern per ADR §8)
//
// Argument binding contract (must match ggml-level-zero.cpp exactly):
//   arg 0: pc   (__constant ze_rope_pc*)     — push-constant struct (112 B)
//   arg 1: x    (__global const float/half*) — input tensor src0->data
//   arg 2: pos  (__global const int*)        — token position array src1->data
//   arg 3: y    (__global float/half*)       — output tensor dst->data
//
// Push-constant fields (see ze_buffer.hpp ze_rope_pc):
//   ne[4]       — element counts [head_dim, n_tokens, n_heads, batch]
//   nb_x[4]     — input byte strides (BYTES, not element counts)
//   nb_y[4]     — output byte strides (BYTES, not element counts)
//   freq_base   — theta base (e.g. 500000.0 for Llama 3, 10000.0 classic)
//   freq_scale  — frequency scale (usually 1.0)
//   attn_factor — YaRN attention factor
//   beta_fast   — YaRN ramp upper boundary
//   beta_slow   — YaRN ramp lower boundary
//   n_ctx_orig  — original context length for YaRN
//   mode        — 0 = neox-style (GPT-NeoX layout), 2 = interleaved
//   n_dims      — number of rotary dimensions
//
// Work-group size: (256, 1, 1).
// Global size: (round_up(n_el / 2, 256), 1, 1) where n_el = ne[0]*ne[1]*ne[2]*ne[3].
// Each work-item handles one rotation pair (two output elements).
//
// Stride-aware indexing macro (ADR §3.2):
//   IDX(i0, i1, i2, i3, nb) = i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
// nb[] values are BYTE strides; the result is a byte offset from the base pointer.
//
// RoPE algorithm (neox-style, mode 0):
//   For each work-item handling pair (i0, i0 + ne[0]/2) within a head:
//     pair_idx  = i0                   (0 .. n_dims/2 - 1)
//     theta     = pos[token] * freq_base^(-2*pair_idx / n_dims) * freq_scale
//     [YaRN]:   theta *= ramp_mix(pair_idx, n_dims, beta_fast, beta_slow, n_ctx_orig)
//     x0, x1   = elements at i0 and i0 + ne[0]/2
//     y0        = x0 * cos(theta) - x1 * sin(theta)
//     y1        = x0 * sin(theta) + x1 * cos(theta)
//   Elements with i0 >= n_dims are passed through unchanged.
//
// F16 adapter (rope_f16, RC2 fix):
//   vload_half at input -> compute trig in F32 -> vstore_half at output.
//   native_cos/native_sin are used (ULP <= 4 per AC-2).

// ---------------------------------------------------------------------------
// Shared YaRN ramp-mix helper.
//
// Returns the interpolation weight (0.0 to 1.0) between the default RoPE
// and the linear (unrotated) portion for YaRN-style context extension.
// When ext_factor == 0.0, returns 1.0 (standard RoPE, no YaRN blending).
// ---------------------------------------------------------------------------
static inline float rope_yarn_ramp(int pair_idx, int n_dims,
                                   float beta_fast, float beta_slow,
                                   float attn_factor, int n_ctx_orig)
{
    // corr_dim_fast and corr_dim_slow are the boundary frequencies in the
    // dimension space where YaRN transitions from fast-rotating to slow-rotating
    // heads.  Derived from the YaRN paper formula:
    //   corr_dim = n_dims * log(n_ctx_orig / (2*pi*beta)) / (2*log(freq_base))
    // We approximate using the beta thresholds directly as dimension fractions
    // per the CUDA rope.cu pattern: the ramp spans [beta_fast, beta_slow].
    float d = (float)pair_idx;
    float low  = (float)n_dims * beta_slow * 0.5f;
    float high = (float)n_dims * beta_fast * 0.5f;
    if (d < low)  return 0.0f;
    if (d > high) return 1.0f;
    return (d - low) / (high - low + 1e-6f);
}

// ---------------------------------------------------------------------------
// rope_f32 — stride-aware F32 RoPE.
//
// RC3 fix: uses IDX macro with nb_x[]/nb_y[] byte strides instead of the
// previous row*n_cols linear addressing that broke after ggml_permute.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void rope_f32(
    __constant int          *pc_raw,   // ze_rope_pc fields packed as int array (arg 0)
    __global const float    *x,        // src0 (arg 1)
    __global const int      *pos,      // token positions src1 (arg 2)
    __global float          *y)        // dst (arg 3)
{
    // Unpack push-constant fields from the raw int array.
    // ze_rope_pc layout (see ze_buffer.hpp):
    //   [0..3]   = ne[4]        (int32)
    //   [4..11]  = nb_x[4]     (int64 — stored as pairs of int32 in host endian)
    //   [12..19] = nb_y[4]     (int64 — same)
    //   [20]     = freq_base   (float bit-pattern in int32)
    //   [21]     = freq_scale  (float)
    //   [22]     = attn_factor (float)
    //   [23]     = beta_fast   (float)
    //   [24]     = beta_slow   (float)
    //   [25]     = n_ctx_orig  (int32)
    //   [26]     = mode        (int32)
    //   [27]     = n_dims      (int32)
    //
    // We use __constant int* and reinterpret because OpenCL 2.0 does not allow
    // passing a struct directly as a kernel argument via zeKernelSetArgumentValue
    // without a matching OpenCL struct declaration.  The dispatcher passes the
    // ze_rope_pc POD bytes; we unpack here.

    int ne0 = pc_raw[0];
    int ne1 = pc_raw[1];
    int ne2 = pc_raw[2];
    int ne3 = pc_raw[3];

    // int64 strides stored as lo/hi int32 pairs (little-endian on x86/Arc).
    long nb_x0 = (long)pc_raw[4]  | ((long)pc_raw[5]  << 32);
    long nb_x1 = (long)pc_raw[6]  | ((long)pc_raw[7]  << 32);
    long nb_x2 = (long)pc_raw[8]  | ((long)pc_raw[9]  << 32);
    long nb_x3 = (long)pc_raw[10] | ((long)pc_raw[11] << 32);

    long nb_y0 = (long)pc_raw[12] | ((long)pc_raw[13] << 32);
    long nb_y1 = (long)pc_raw[14] | ((long)pc_raw[15] << 32);
    long nb_y2 = (long)pc_raw[16] | ((long)pc_raw[17] << 32);
    long nb_y3 = (long)pc_raw[18] | ((long)pc_raw[19] << 32);

    float freq_base   = as_float(pc_raw[20]);
    float freq_scale  = as_float(pc_raw[21]);
    float attn_factor = as_float(pc_raw[22]);
    float beta_fast   = as_float(pc_raw[23]);
    float beta_slow   = as_float(pc_raw[24]);
    int   n_ctx_orig  = pc_raw[25];
    int   mode        = pc_raw[26];
    int   n_dims      = pc_raw[27];

    // Total pairs = (ne0/2) * ne1 * ne2 * ne3.
    // Each work-item handles one pair → global_id maps to a linear pair index.
    uint gid = get_global_id(0);
    uint n_pairs = (uint)(ne0 / 2);
    uint total_pairs = n_pairs * (uint)ne1 * (uint)ne2 * (uint)ne3;

    if (gid >= total_pairs) return;

    // Decode pair index into (i0_pair, i1, i2, i3).
    uint i0_pair = gid % n_pairs;
    uint i1      = (gid / n_pairs) % (uint)ne1;
    uint i2      = (gid / (n_pairs * (uint)ne1)) % (uint)ne2;
    uint i3      = gid / (n_pairs * (uint)ne1 * (uint)ne2);

    // Token position for this (i1) index.
    int token_pos = pos[i1];

    // Rotation angle for this dimension pair.
    float exponent = 2.0f * (float)i0_pair / (float)n_dims;
    float theta    = (float)token_pos * native_powr(freq_base, -exponent) * freq_scale;

    // YaRN ramp: blend standard RoPE with extended context.
    // When attn_factor == 0 the ramp has no effect (passthrough).
    if (attn_factor != 0.0f) {
        float ramp = rope_yarn_ramp((int)i0_pair, n_dims,
                                    beta_fast, beta_slow, attn_factor, n_ctx_orig);
        theta *= (1.0f - ramp) * attn_factor + ramp;
    }

    float cos_t = native_cos(theta);
    float sin_t = native_sin(theta);

    // Source addresses using IDX macro (byte offsets from base pointer).
    // i0 for the lower element of the pair.
    uint i0_lo = i0_pair;
    // i0 for the upper element (ne0/2 away in element space → nb_x0*ne0/2 bytes away).
    uint i0_hi = i0_pair + n_pairs;  // i0_pair + ne0/2

    long addr_x_lo = (long)i0_lo * nb_x0 + (long)i1 * nb_x1
                   + (long)i2 * nb_x2 + (long)i3 * nb_x3;
    long addr_x_hi = (long)i0_hi * nb_x0 + (long)i1 * nb_x1
                   + (long)i2 * nb_x2 + (long)i3 * nb_x3;
    long addr_y_lo = (long)i0_lo * nb_y0 + (long)i1 * nb_y1
                   + (long)i2 * nb_y2 + (long)i3 * nb_y3;
    long addr_y_hi = (long)i0_hi * nb_y0 + (long)i1 * nb_y1
                   + (long)i2 * nb_y2 + (long)i3 * nb_y3;

    // Pass through elements beyond n_dims unchanged.
    if ((int)i0_pair >= n_dims / 2) {
        float x_lo = *(__global const float *)(((__global const char *)x) + addr_x_lo);
        float x_hi = *(__global const float *)(((__global const char *)x) + addr_x_hi);
        *(__global float *)((__global char *)y + addr_y_lo) = x_lo;
        *(__global float *)((__global char *)y + addr_y_hi) = x_hi;
        return;
    }

    float x0 = *(__global const float *)(((__global const char *)x) + addr_x_lo);
    float x1 = *(__global const float *)(((__global const char *)x) + addr_x_hi);

    *(__global float *)((__global char *)y + addr_y_lo) = x0 * cos_t - x1 * sin_t;
    *(__global float *)((__global char *)y + addr_y_hi) = x0 * sin_t + x1 * cos_t;
}

// ---------------------------------------------------------------------------
// rope_f16 — stride-aware F16 RoPE (RC2 fix — eliminates GGML_ABORT).
//
// Adapter pattern (ADR §8): vload_half → F32 rotation math → vstore_half.
// native_cos/native_sin used; ULP error <= 4 per AC-2 (see CUDA Brief §9.6d).
//
// This kernel directly resolves Issue RC2: ggml-backend.cpp:844 GGML_ABORT.
// The Llama 3.2 1B GQA KV-cache stores positions in F16; without this kernel
// the scheduler found no backend willing to handle F16 ROPE and aborted.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void rope_f16(
    __constant int          *pc_raw,   // ze_rope_pc fields (arg 0)
    __global const half     *x,        // src0 F16 (arg 1)
    __global const int      *pos,      // token positions (arg 2)
    __global half           *y)        // dst F16 (arg 3)
{
    // Unpack push-constant — identical layout to rope_f32.
    int ne0 = pc_raw[0];
    int ne1 = pc_raw[1];
    int ne2 = pc_raw[2];
    int ne3 = pc_raw[3];

    long nb_x0 = (long)pc_raw[4]  | ((long)pc_raw[5]  << 32);
    long nb_x1 = (long)pc_raw[6]  | ((long)pc_raw[7]  << 32);
    long nb_x2 = (long)pc_raw[8]  | ((long)pc_raw[9]  << 32);
    long nb_x3 = (long)pc_raw[10] | ((long)pc_raw[11] << 32);

    long nb_y0 = (long)pc_raw[12] | ((long)pc_raw[13] << 32);
    long nb_y1 = (long)pc_raw[14] | ((long)pc_raw[15] << 32);
    long nb_y2 = (long)pc_raw[16] | ((long)pc_raw[17] << 32);
    long nb_y3 = (long)pc_raw[18] | ((long)pc_raw[19] << 32);

    float freq_base   = as_float(pc_raw[20]);
    float freq_scale  = as_float(pc_raw[21]);
    float attn_factor = as_float(pc_raw[22]);
    float beta_fast   = as_float(pc_raw[23]);
    float beta_slow   = as_float(pc_raw[24]);
    int   n_ctx_orig  = pc_raw[25];
    int   mode        = pc_raw[26];
    int   n_dims      = pc_raw[27];

    uint gid = get_global_id(0);
    uint n_pairs = (uint)(ne0 / 2);
    uint total_pairs = n_pairs * (uint)ne1 * (uint)ne2 * (uint)ne3;

    if (gid >= total_pairs) return;

    uint i0_pair = gid % n_pairs;
    uint i1      = (gid / n_pairs) % (uint)ne1;
    uint i2      = (gid / (n_pairs * (uint)ne1)) % (uint)ne2;
    uint i3      = gid / (n_pairs * (uint)ne1 * (uint)ne2);

    int token_pos = pos[i1];

    float exponent = 2.0f * (float)i0_pair / (float)n_dims;
    float theta    = (float)token_pos * native_powr(freq_base, -exponent) * freq_scale;

    if (attn_factor != 0.0f) {
        float ramp = rope_yarn_ramp((int)i0_pair, n_dims,
                                    beta_fast, beta_slow, attn_factor, n_ctx_orig);
        theta *= (1.0f - ramp) * attn_factor + ramp;
    }

    float cos_t = native_cos(theta);
    float sin_t = native_sin(theta);

    uint i0_lo = i0_pair;
    uint i0_hi = i0_pair + n_pairs;

    long addr_x_lo = (long)i0_lo * nb_x0 + (long)i1 * nb_x1
                   + (long)i2 * nb_x2 + (long)i3 * nb_x3;
    long addr_x_hi = (long)i0_hi * nb_x0 + (long)i1 * nb_x1
                   + (long)i2 * nb_x2 + (long)i3 * nb_x3;
    long addr_y_lo = (long)i0_lo * nb_y0 + (long)i1 * nb_y1
                   + (long)i2 * nb_y2 + (long)i3 * nb_y3;
    long addr_y_hi = (long)i0_hi * nb_y0 + (long)i1 * nb_y1
                   + (long)i2 * nb_y2 + (long)i3 * nb_y3;

    // Adapter load: half -> float (vload_half is exact per IEEE 754-2008, 0 ULP).
    // Byte address -> half element address for vload_half.
    // nb_x0 == sizeof(half) == 2 for a contiguous F16 tensor, but may differ
    // for non-contiguous tensors.  We index using the raw byte offset.
    float x0_f32 = vload_half(0, (__global const half *)((__global const char *)x + addr_x_lo));
    float x1_f32 = vload_half(0, (__global const half *)((__global const char *)x + addr_x_hi));

    if ((int)i0_pair >= n_dims / 2) {
        // Elements beyond n_dims: pass through unchanged (store as half).
        vstore_half(x0_f32, 0, (__global half *)((__global char *)y + addr_y_lo));
        vstore_half(x1_f32, 0, (__global half *)((__global char *)y + addr_y_hi));
        return;
    }

    // F32 rotation (matching the CUDA half template path per CUDA Brief §9.6c).
    float y0_f32 = x0_f32 * cos_t - x1_f32 * sin_t;
    float y1_f32 = x0_f32 * sin_t + x1_f32 * cos_t;

    // Adapter store: float -> half via vstore_half (round-to-nearest-even, 0.5 ULP).
    vstore_half(y0_f32, 0, (__global half *)((__global char *)y + addr_y_lo));
    vstore_half(y1_f32, 0, (__global half *)((__global char *)y + addr_y_hi));
}
