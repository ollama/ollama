// SPDX-License-Identifier: MIT
// rms_norm.cl — RMS Normalisation kernel for the Intel Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file rms_norm.cl -device <target> -options "-cl-std=CL2.0 -O3"
//
// One entry point is compiled from this file:
//   rms_norm — applies RMS normalisation (no weight; weight applied by downstream mul)
//
// Argument binding contract (must match ggml-level-zero.cpp exactly):
//   arg 0: x      (__global const float*) — input tensor     src0->data
//   arg 1: y      (__global float*)       — output tensor    node->data
//   arg 2: n_cols (uint)                  — row width == src0->ne[0]
//   arg 3: eps    (float)                 — stability epsilon == *(float*)&op_params[0]
//
// Work-group size: (256, 1, 1).  One work-group per input row (token).
// groupCountX = n_rows.
//
// Formula applied per row:
//   sum_sq  = sum(x[i]^2)  for i in [0, n_cols)
//   inv_rms = rsqrt(sum_sq / n_cols + eps)
//   y[i]    = x[i] * inv_rms
//
// GGML_OP_RMS_NORM has signature (src[0]=x, op_params[0]=eps).
// There is no weight operand; the learnable scale (gamma) is a separate
// GGML_OP_MUL node downstream in the compute graph.

#define WG_SIZE 256

// ---------------------------------------------------------------------------
// RMS normalisation without affine weight rescaling.
//
// Uses a local-memory parallel reduction to compute the mean of squared
// inputs across the row.  The result is broadcast to all threads so they can
// normalise their assigned columns independently.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void rms_norm(
    __global const float *x,
    __global       float *y,
    uint  n_cols,
    float eps)
{
    __local float scratch[WG_SIZE];

    uint row = get_group_id(0);
    uint lid = get_local_id(0);

    __global const float *x_row = x + row * n_cols;
    __global       float *y_row = y + row * n_cols;

    // Phase 1: each thread accumulates partial sum of squares over its stripe.
    float partial = 0.0f;
    for (uint i = lid; i < n_cols; i += WG_SIZE) {
        float v  = x_row[i];
        partial += v * v;
    }
    scratch[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: binary tree reduction to produce the full sum of squares.
    for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean_sq = scratch[0] / (float)n_cols;
    float inv_rms = native_rsqrt(mean_sq + eps);

    // Phase 3: normalise each element.
    for (uint i = lid; i < n_cols; i += WG_SIZE) {
        y_row[i] = x_row[i] * inv_rms;
    }
}
