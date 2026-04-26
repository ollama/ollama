// SPDX-License-Identifier: MIT
// rms_norm.cl — RMS Normalisation kernels for the Intel Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file rms_norm.cl -device <target> -options "-cl-std=CL2.0 -O3"
//
// Two entry points compiled from this file:
//   rms_norm_f32 — F32 input/output with stride-aware row addressing (RC3 fix)
//   rms_norm_f16 — F16 input/output with F32 accumulator (numerical stability)
//
// CRITICAL — Bug #10 invariant (commit 32f6fac9, ADR Section 10.1):
//   Neither kernel has a weight argument.  The learnable scale (gamma) is applied
//   by a downstream GGML_OP_MUL node in the compute graph.  The argument list
//   MUST remain 3 arguments: (pc_raw, x, y).
//   Verification: grep -n "weight" rms_norm.cl MUST return zero matches.
//
// Argument binding contract (must match ggml-level-zero.cpp exactly):
//   arg 0: pc_raw (__constant int*)        — ze_rms_norm_pc as raw int array (88 B)
//   arg 1: x      (__global const T*)      — input tensor src0->data
//   arg 2: y      (__global T*)            — output tensor dst->data
//
// Push-constant fields (see ze_buffer.hpp ze_rms_norm_pc):
//   ne[4]   — element counts [ne0=n_cols, ne1, ne2, ne3]
//   nb_x[4] — input byte strides (BYTES)
//   nb_y[4] — output byte strides (BYTES)
//   eps     — stability epsilon (e.g. 1e-5)
//   _pad    — 4-byte alignment pad (ignored)
//
// Work-group size: (256, 1, 1).
// Global size:     (256, ne[1]*ne[2]*ne[3], 1) — one work-group per row.
// The Y dimension of the global size directly encodes the flattened row index
// across all batch dimensions (ne1, ne2, ne3).
//
// Stride-aware row addressing (RC3 fix):
//   Row (i1, i2, i3) starts at byte offset:
//     base_x = i3 * nb_x[3] + i2 * nb_x[2] + i1 * nb_x[1]
//   Element i0 within the row:
//     byte offset = base_x + i0 * nb_x[0]
//
// Algorithm per row:
//   Phase 1: each thread accumulates partial sum(x[i]^2) via F32 accumulator.
//   Phase 2: tree-reduce across 256 threads using local memory.
//   Phase 3: compute inv_rms = rsqrt(mean_sq + eps), normalise all elements.
//
// F16 variant (rms_norm_f16):
//   Loads via vload_half → accumulate in F32 (prevents overflow for ncols >= 2048
//   per CUDA Brief §9.8d) → stores via vstore_half.

#define WG_SIZE 256

// ---------------------------------------------------------------------------
// rms_norm_f32 — stride-aware F32 RMS normalisation.
//
// Bug #10 signature guard: exactly 3 arguments.  No weight parameter.
// RC3 fix: row addressing uses nb_x[] byte strides instead of row*n_cols.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void rms_norm_f32(
    __constant int          *pc_raw,   // ze_rms_norm_pc (arg 0)
    __global const float    *x,        // input  src0 (arg 1)
    __global       float    *y)        // output dst  (arg 2)
{
    __local float scratch[WG_SIZE];

    // Unpack ze_rms_norm_pc from raw int array.
    // Layout: ne[4](0..3), nb_x[4](4..11 as int64 pairs), nb_y[4](12..19), eps(20), _pad(21).
    int ne0 = pc_raw[0];  // n_cols
    // ne1..ne3 decoded from the grid Y-dimension below; ne0 is sufficient for the inner loop.

    long nb_x0 = (long)pc_raw[4]  | ((long)pc_raw[5]  << 32);
    long nb_x1 = (long)pc_raw[6]  | ((long)pc_raw[7]  << 32);
    long nb_x2 = (long)pc_raw[8]  | ((long)pc_raw[9]  << 32);
    long nb_x3 = (long)pc_raw[10] | ((long)pc_raw[11] << 32);

    long nb_y0 = (long)pc_raw[12] | ((long)pc_raw[13] << 32);
    long nb_y1 = (long)pc_raw[14] | ((long)pc_raw[15] << 32);
    long nb_y2 = (long)pc_raw[16] | ((long)pc_raw[17] << 32);
    long nb_y3 = (long)pc_raw[18] | ((long)pc_raw[19] << 32);

    float eps = as_float(pc_raw[20]);

    // ne1, ne2, ne3 for batch index decode.
    int ne1 = pc_raw[1];
    int ne2 = pc_raw[2];
    // int ne3 = pc_raw[3];  // not needed explicitly; total rows = get_global_size(1)

    uint lid     = get_local_id(0);
    uint row_idx = get_group_id(1);  // flattened row index across (i1, i2, i3)

    // Decode row index into (i1, i2, i3).
    uint i1 = row_idx % (uint)ne1;
    uint i2 = (row_idx / (uint)ne1) % (uint)ne2;
    uint i3 = row_idx / ((uint)ne1 * (uint)ne2);

    // Byte offset to the start of this row in src and dst.
    long base_x = (long)i3 * nb_x3 + (long)i2 * nb_x2 + (long)i1 * nb_x1;
    long base_y = (long)i3 * nb_y3 + (long)i2 * nb_y2 + (long)i1 * nb_y1;

    // Phase 1: partial sum of squares.
    float partial = 0.0f;
    for (int i = (int)lid; i < ne0; i += WG_SIZE) {
        long byte_off = base_x + (long)i * nb_x0;
        float v = *(__global const float *)((__global const char *)x + byte_off);
        partial += v * v;
    }
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: binary tree reduction.
    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean_sq = scratch[0] / (float)ne0;
    float inv_rms = native_rsqrt(mean_sq + eps);

    // Phase 3: normalise.  No weight multiply per Bug #10 invariant.
    for (int i = (int)lid; i < ne0; i += WG_SIZE) {
        long byte_off_x = base_x + (long)i * nb_x0;
        long byte_off_y = base_y + (long)i * nb_y0;
        float v = *(__global const float *)((__global const char *)x + byte_off_x);
        *(__global float *)((__global char *)y + byte_off_y) = v * inv_rms;
    }
}

// ---------------------------------------------------------------------------
// rms_norm_f16 — stride-aware F16 RMS normalisation with F32 accumulator.
//
// Bug #10 signature guard: exactly 3 arguments.  No weight parameter.
// F32 accumulator mandatory (per CUDA Brief §9.8d): F16^2 would overflow for
// row widths >= 2048 elements (common in large Llama attention heads).
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void rms_norm_f16(
    __constant int          *pc_raw,   // ze_rms_norm_pc (arg 0)
    __global const half     *x,        // input  src0 F16 (arg 1)
    __global       half     *y)        // output dst  F16 (arg 2)
{
    __local float scratch[WG_SIZE];

    int ne0 = pc_raw[0];

    long nb_x0 = (long)pc_raw[4]  | ((long)pc_raw[5]  << 32);
    long nb_x1 = (long)pc_raw[6]  | ((long)pc_raw[7]  << 32);
    long nb_x2 = (long)pc_raw[8]  | ((long)pc_raw[9]  << 32);
    long nb_x3 = (long)pc_raw[10] | ((long)pc_raw[11] << 32);

    long nb_y0 = (long)pc_raw[12] | ((long)pc_raw[13] << 32);
    long nb_y1 = (long)pc_raw[14] | ((long)pc_raw[15] << 32);
    long nb_y2 = (long)pc_raw[16] | ((long)pc_raw[17] << 32);
    long nb_y3 = (long)pc_raw[18] | ((long)pc_raw[19] << 32);

    float eps = as_float(pc_raw[20]);

    int ne1 = pc_raw[1];
    int ne2 = pc_raw[2];

    uint lid     = get_local_id(0);
    uint row_idx = get_group_id(1);

    uint i1 = row_idx % (uint)ne1;
    uint i2 = (row_idx / (uint)ne1) % (uint)ne2;
    uint i3 = row_idx / ((uint)ne1 * (uint)ne2);

    long base_x = (long)i3 * nb_x3 + (long)i2 * nb_x2 + (long)i1 * nb_x1;
    long base_y = (long)i3 * nb_y3 + (long)i2 * nb_y2 + (long)i1 * nb_y1;

    // Phase 1: F16 load -> F32 accumulate (prevents overflow at ncols >= 2048).
    float partial = 0.0f;
    for (int i = (int)lid; i < ne0; i += WG_SIZE) {
        long byte_off = base_x + (long)i * nb_x0;
        float v = vload_half(0, (__global const half *)((__global const char *)x + byte_off));
        partial += v * v;
    }
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: binary tree reduction (same as F32 variant).
    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean_sq = scratch[0] / (float)ne0;
    float inv_rms = native_rsqrt(mean_sq + eps);

    // Phase 3: normalise, load as F16, store as F16.  No weight per Bug #10.
    for (int i = (int)lid; i < ne0; i += WG_SIZE) {
        long byte_off_x = base_x + (long)i * nb_x0;
        long byte_off_y = base_y + (long)i * nb_y0;
        float v = vload_half(0, (__global const half *)((__global const char *)x + byte_off_x));
        vstore_half(v * inv_rms, 0, (__global half *)((__global char *)y + byte_off_y));
    }
}
