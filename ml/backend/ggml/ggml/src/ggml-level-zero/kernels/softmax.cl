// SPDX-License-Identifier: MIT
// softmax.cl — Softmax and causal-softmax kernels for the Intel Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file softmax.cl -device <target> -options "-cl-std=CL2.0 -O3"
//
// Two entry points are compiled from this file:
//   softmax_f32        — standard softmax (no causal mask)
//   softmax_causal_f32 — causal softmax (masks positions > cur_pos to -INFINITY)
//
// Argument binding contract (must match ggml-level-zero.cpp exactly):
//   arg 0: x      (__global const float*) — input row tensor  src0->data
//   arg 1: y      (__global float*)       — output row tensor  node->data
//   arg 2: n_cols (uint)                  — row width == src0->ne[0]
//   arg 3: scale  (float)                 — pre-scale applied to input (typically 1.0f)
//   arg 4: cur_pos (uint)  [causal only]  — current token position == op_params[1]
//
// Work-group size: (256, 1, 1).  One work-group per row.
// groupCountX = n_rows.  Each work-group processes its row cooperatively.
//
// Algorithm: numerically stable three-phase softmax:
//   Phase 1: reduce to find the row maximum (for exp stability).
//   Phase 2: compute exp(x*scale - max) and reduce to find the sum.
//   Phase 3: normalise by dividing each element by the sum.

#define WG_SIZE 256

// ---------------------------------------------------------------------------
// Standard (full-context) softmax
//
// Applies scale to every input element before computing max and exp.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void softmax_f32(
    __global const float *x,
    __global       float *y,
    uint   n_cols,
    float  scale)
{
    __local float scratch[WG_SIZE];

    uint row = get_group_id(0);
    uint lid = get_local_id(0);

    __global const float *x_row = x + row * n_cols;
    __global       float *y_row = y + row * n_cols;

    // Phase 1: find row maximum for numerical stability.
    float lmax = -INFINITY;
    for (uint i = lid; i < n_cols; i += WG_SIZE) {
        float v = x_row[i] * scale;
        if (v > lmax) lmax = v;
    }
    scratch[lid] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            if (scratch[lid + s] > scratch[lid]) scratch[lid] = scratch[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = scratch[0];

    // Phase 2: compute exp(x*scale - max) and accumulate partial sums.
    float lsum = 0.0f;
    for (uint i = lid; i < n_cols; i += WG_SIZE) {
        float v  = native_exp(x_row[i] * scale - row_max);
        y_row[i] = v;
        lsum    += v;
    }
    scratch[lid] = lsum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_sum = 1.0f / scratch[0];

    // Phase 3: normalise.
    for (uint i = lid; i < n_cols; i += WG_SIZE) {
        y_row[i] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Causal (autoregressive) softmax
//
// Positions j > cur_pos are treated as -INFINITY before computing max, so
// their exp value becomes 0 and they contribute nothing to the sum.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void softmax_causal_f32(
    __global const float *x,
    __global       float *y,
    uint   n_cols,
    float  scale,
    uint   cur_pos)
{
    __local float scratch[WG_SIZE];

    uint row = get_group_id(0);
    uint lid = get_local_id(0);

    __global const float *x_row = x + row * n_cols;
    __global       float *y_row = y + row * n_cols;

    // Phase 1: find max over unmasked positions (j <= cur_pos).
    float lmax = -INFINITY;
    for (uint i = lid; i < n_cols; i += WG_SIZE) {
        float v = (i <= cur_pos) ? (x_row[i] * scale) : -INFINITY;
        if (v > lmax) lmax = v;
    }
    scratch[lid] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            if (scratch[lid + s] > scratch[lid]) scratch[lid] = scratch[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = scratch[0];

    // Phase 2: exp for unmasked positions, zero for masked positions.
    float lsum = 0.0f;
    for (uint i = lid; i < n_cols; i += WG_SIZE) {
        float v;
        if (i <= cur_pos) {
            v = native_exp(x_row[i] * scale - row_max);
        } else {
            v = 0.0f;
        }
        y_row[i] = v;
        lsum    += v;
    }
    scratch[lid] = lsum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_sum = (scratch[0] > 0.0f) ? (1.0f / scratch[0]) : 0.0f;

    // Phase 3: normalise.
    for (uint i = lid; i < n_cols; i += WG_SIZE) {
        y_row[i] *= inv_sum;
    }
}
