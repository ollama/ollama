// SPDX-License-Identifier: MIT
// softmax.cl — Softmax kernel for the Intel Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file softmax.cl -device <target> -options "-cl-std=CL2.0 -O3"
//
// One entry point compiled from this file:
//   softmax_f32 — stride-aware F32 softmax with optional mask and ALiBi (RC3 fix)
//
// Previous entry points removed in this rewrite:
//   softmax_causal_f32 — subsumed by softmax_f32 with has_mask + mask=-INF encoding
//
// Argument binding contract (must match ggml-level-zero.cpp exactly):
//   arg 0: pc_raw (__constant int*)       — ze_softmax_pc as raw int array (128 B)
//   arg 1: x      (__global const float*) — input tensor src0->data
//   arg 2: mask   (__global const float*) — mask tensor src1->data (may be null buffer)
//   arg 3: y      (__global float*)       — output tensor dst->data
//
// Push-constant fields (see ze_buffer.hpp ze_softmax_pc):
//   ne[4]      — element counts [ne0=n_cols, ne1, ne2, ne3]
//   nb_x[4]    — input byte strides (BYTES)
//   nb_y[4]    — output byte strides (BYTES)
//   nb_mask[4] — mask byte strides (BYTES; zeroed when has_mask=0)
//   scale      — pre-scale applied to each input element (typically 1/sqrt(d_k))
//   max_bias   — ALiBi max bias (0.0 when has_alibi=0)
//   has_mask   — 1 if mask buffer is valid, 0 otherwise
//   has_alibi  — 1 if ALiBi slope is applied, 0 otherwise
//
// Work-group size: (256, 1, 1).
// Global size:     (256, ne[1]*ne[2]*ne[3], 1) — one work-group per row.
// The grid Y dimension encodes the flattened row index (i1, i2, i3).
//
// Stride-aware indexing (ADR §3.2):
//   Row base (byte offset from data pointer):
//     base_x    = i3*nb_x[3] + i2*nb_x[2] + i1*nb_x[1]
//     base_mask = i3*nb_mask[3] + i2*nb_mask[2] + (i1 % mask_rows)*nb_mask[1]
//   Element i0: byte_off = base + i0*nb_X[0]
//
// Algorithm (two-pass numerically-stable softmax per CUDA Brief §9.9c/e):
//   Pass 1: find row maximum with scale + optional mask + optional ALiBi applied.
//           barrier(CLK_LOCAL_MEM_FENCE) after reduction (correctness requirement).
//   Pass 2: compute exp(xi - max), accumulate sum, write exp values.
//           barrier between exp-sum and normalise.
//   Pass 3: divide each element by the sum to produce normalised probabilities.
//
// ALiBi slope formula (CUDA softmax.cu §9.9d):
//   n_head_log2 = largest power of 2 <= n_heads (= ne[2])
//   For head index h = i2:
//     if h < n_head_log2: slope = exp2f(-max_bias * (h+1))            [m0 path]
//     else:               slope = exp2f(-max_bias * 0.5 * (2*(h-n_head_log2)+1)) [m1 path]
//   xi += slope * col_index_within_row

#define WG_SIZE 256

// ---------------------------------------------------------------------------
// softmax_f32 — stride-aware F32 softmax with optional mask and ALiBi.
//
// RC3 fix: all row accesses use nb_x[], nb_mask[] byte strides rather than
// the previous row*n_cols linear addressing that broke on non-contiguous tensors.
// Correctness requirement: barrier(CLK_LOCAL_MEM_FENCE) between the max-reduction
// pass and the exp-sum pass (per CUDA Brief §9.9e and ADR §3.2 work-split).
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void softmax_f32(
    __constant int          *pc_raw,   // ze_softmax_pc (arg 0)
    __global const float    *x,        // input  src0 (arg 1)
    __global const float    *mask,     // mask   src1 (arg 2, may be null/unused)
    __global       float    *y)        // output dst  (arg 3)
{
    __local float scratch[WG_SIZE];

    // Unpack ze_softmax_pc from the raw int array.
    // Layout offsets (int32 elements):
    //   [0..3]   = ne[4]        (int32 x4)
    //   [4..11]  = nb_x[4]     (int64 x4 = 8 int32 values)
    //   [12..19] = nb_y[4]     (int64 x4)
    //   [20..27] = nb_mask[4]  (int64 x4)
    //   [28]     = scale        (float as int32 bits)
    //   [29]     = max_bias     (float)
    //   [30]     = has_mask     (int32)
    //   [31]     = has_alibi    (int32)

    int ne0 = pc_raw[0];
    int ne1 = pc_raw[1];
    int ne2 = pc_raw[2];
    // int ne3 = pc_raw[3];  // decoded from grid

    long nb_x0 = (long)pc_raw[4]  | ((long)pc_raw[5]  << 32);
    long nb_x1 = (long)pc_raw[6]  | ((long)pc_raw[7]  << 32);
    long nb_x2 = (long)pc_raw[8]  | ((long)pc_raw[9]  << 32);
    long nb_x3 = (long)pc_raw[10] | ((long)pc_raw[11] << 32);

    long nb_y0 = (long)pc_raw[12] | ((long)pc_raw[13] << 32);
    long nb_y1 = (long)pc_raw[14] | ((long)pc_raw[15] << 32);
    long nb_y2 = (long)pc_raw[16] | ((long)pc_raw[17] << 32);
    long nb_y3 = (long)pc_raw[18] | ((long)pc_raw[19] << 32);

    long nb_mask0 = (long)pc_raw[20] | ((long)pc_raw[21] << 32);
    long nb_mask1 = (long)pc_raw[22] | ((long)pc_raw[23] << 32);
    long nb_mask2 = (long)pc_raw[24] | ((long)pc_raw[25] << 32);
    long nb_mask3 = (long)pc_raw[26] | ((long)pc_raw[27] << 32);

    float scale    = as_float(pc_raw[28]);
    float max_bias = as_float(pc_raw[29]);
    int   has_mask  = pc_raw[30];
    int   has_alibi = pc_raw[31];

    uint lid     = get_local_id(0);
    uint row_idx = get_group_id(1);

    // Decode (i1, i2, i3) from flattened row index.
    uint i1 = row_idx % (uint)ne1;
    uint i2 = (row_idx / (uint)ne1) % (uint)ne2;
    uint i3 = row_idx / ((uint)ne1 * (uint)ne2);

    // Byte base for this row.
    long base_x = (long)i3 * nb_x3 + (long)i2 * nb_x2 + (long)i1 * nb_x1;
    long base_y = (long)i3 * nb_y3 + (long)i2 * nb_y2 + (long)i1 * nb_y1;

    // Mask is broadcast across the batch dimensions (ne2, ne3) — use i1 modulo
    // the number of mask rows (ne_mask[1]).  The mask byte strides in nb_mask[]
    // are set to zero for dimensions the mask broadcasts across (Null Object pattern).
    // For the common causal mask case: nb_mask[2]=0, nb_mask[3]=0, nb_mask[1] is set.
    long base_mask = 0;
    if (has_mask) {
        base_mask = (long)i3 * nb_mask3 + (long)i2 * nb_mask2 + (long)i1 * nb_mask1;
    }

    // ALiBi slope for this head (i2).
    float alibi_slope = 0.0f;
    if (has_alibi) {
        int n_heads = ne2;
        // n_head_log2 = largest power of 2 <= n_heads.
        int n_head_log2 = 1;
        while (n_head_log2 * 2 <= n_heads) n_head_log2 *= 2;

        int h = (int)i2;
        if (h < n_head_log2) {
            alibi_slope = exp2(-max_bias * (float)(h + 1));
        } else {
            alibi_slope = exp2(-max_bias * 0.5f * (float)(2 * (h - n_head_log2) + 1));
        }
    }

    // -------------------------------------------------------------------------
    // Pass 1: find row maximum.
    // Input xi = x[i] * scale + optional_mask[i] + optional_alibi_slope * i.
    // -------------------------------------------------------------------------
    float lmax = -INFINITY;
    for (int i = (int)lid; i < ne0; i += WG_SIZE) {
        long byte_off = base_x + (long)i * nb_x0;
        float xi = *(__global const float *)((__global const char *)x + byte_off) * scale;

        if (has_alibi) {
            xi += alibi_slope * (float)i;
        }
        if (has_mask) {
            long mask_off = base_mask + (long)i * nb_mask0;
            xi += *(__global const float *)((__global const char *)mask + mask_off);
        }
        lmax = fmax(lmax, xi);
    }
    scratch[lid] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree-reduce to find the true row maximum across all 256 threads.
    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = scratch[0];

    // -------------------------------------------------------------------------
    // Pass 2: compute exp(xi - max) and accumulate partial sums.
    // Write intermediate exp values into y[] for pass 3.
    // The barrier between this pass and pass 3 is the correctness requirement
    // (per CUDA Brief §9.9e and ADR work-split §3.2).
    // -------------------------------------------------------------------------
    float lsum = 0.0f;
    for (int i = (int)lid; i < ne0; i += WG_SIZE) {
        long byte_off_x = base_x + (long)i * nb_x0;
        long byte_off_y = base_y + (long)i * nb_y0;

        float xi = *(__global const float *)((__global const char *)x + byte_off_x) * scale;

        if (has_alibi) {
            xi += alibi_slope * (float)i;
        }
        if (has_mask) {
            long mask_off = base_mask + (long)i * nb_mask0;
            xi += *(__global const float *)((__global const char *)mask + mask_off);
        }

        float ev = native_exp(xi - row_max);
        *(__global float *)((__global char *)y + byte_off_y) = ev;
        lsum += ev;
    }
    scratch[lid] = lsum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree-reduce to find the total sum.
    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_sum = (scratch[0] > 0.0f) ? (1.0f / scratch[0]) : 0.0f;

    // -------------------------------------------------------------------------
    // Pass 3: normalise by dividing each exp value by the total sum.
    // -------------------------------------------------------------------------
    for (int i = (int)lid; i < ne0; i += WG_SIZE) {
        long byte_off_y = base_y + (long)i * nb_y0;
        float ev = *(__global float *)((__global char *)y + byte_off_y);
        *(__global float *)((__global char *)y + byte_off_y) = ev * inv_sum;
    }
}
