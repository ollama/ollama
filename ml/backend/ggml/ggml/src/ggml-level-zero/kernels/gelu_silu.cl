// SPDX-License-Identifier: MIT
// gelu_silu.cl — Element-wise activation, arithmetic, and copy kernels for the
//                Intel Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file gelu_silu.cl -device <target> -options "-cl-std=CL2.0 -O3"
//
// Five entry points are compiled from this file (gelu_silu module):
//   gelu_f32  — Gaussian Error Linear Unit via erf formula
//   silu_f32  — Sigmoid Linear Unit (x * sigmoid(x))
//   add_f32   — element-wise addition
//   mul_f32   — element-wise multiplication
//   copy_f32  — element-wise copy (contiguous tensor copy)
//
// Argument binding contract for gelu_f32, silu_f32, copy_f32
// (must match ggml-level-zero.cpp exactly):
//   arg 0: src  (__global const float*) — input  src0->data
//   arg 1: dst  (__global float*)       — output node->data
//   arg 2: n_el (uint)                  — total element count == ggml_nelements(node)
//
// Argument binding contract for add_f32, mul_f32:
//   arg 0: a    (__global const float*) — first input  src0->data
//   arg 1: b    (__global const float*) — second input src1->data
//   arg 2: c    (__global float*)       — output       node->data
//   arg 3: n_el (uint)                  — total element count == ggml_nelements(node)
//
// Work-group size: (256, 1, 1) for all five kernels.
// groupCountX = ceil(n_el / 256).

// 1/sqrt(2) constant used by the GELU erf formula.
#define M_SQRT1_2_F  0.70710678118654752440f

// ---------------------------------------------------------------------------
// GELU activation using the exact erf-based formula.
//
// Formula: y = x * 0.5 * (1 + erf(x * M_SQRT1_2))
// This is the standard (non-approximate) GELU as used by BERT / GPT-2 in
// GGML when the unary op selects GGML_UNARY_OP_GELU.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void gelu_f32(
    __global const float *src,
    __global       float *dst,
    uint n_el)
{
    uint i = get_global_id(0);
    if (i >= n_el) return;
    float v = src[i];
    dst[i]  = v * 0.5f * (1.0f + erf(v * M_SQRT1_2_F));
}

// ---------------------------------------------------------------------------
// SiLU activation: y = x * sigmoid(x) = x / (1 + exp(-x)).
//
// Used for GGML_UNARY_OP_SILU.  native_exp is appropriate here because the
// small approximation error in native_exp is well below the numerical noise
// of fp32 activations in inference.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void silu_f32(
    __global const float *src,
    __global       float *dst,
    uint n_el)
{
    uint i = get_global_id(0);
    if (i >= n_el) return;
    float v = src[i];
    dst[i]  = v / (1.0f + native_exp(-v));
}

// ---------------------------------------------------------------------------
// Element-wise F32 addition: c[i] = a[i] + b[i].
//
// Used for GGML_OP_ADD.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void add_f32(
    __global const float *a,
    __global const float *b,
    __global       float *c,
    uint n_el)
{
    uint i = get_global_id(0);
    if (i >= n_el) return;
    c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Element-wise F32 multiplication: c[i] = a[i] * b[i].
//
// Used for GGML_OP_MUL.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mul_f32(
    __global const float *a,
    __global const float *b,
    __global       float *c,
    uint n_el)
{
    uint i = get_global_id(0);
    if (i >= n_el) return;
    c[i] = a[i] * b[i];
}

// ---------------------------------------------------------------------------
// Contiguous F32 tensor copy: dst[i] = src[i].
//
// Used for GGML_OP_CONT (make-contiguous).
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void copy_f32(
    __global const float *src,
    __global       float *dst,
    uint n_el)
{
    uint i = get_global_id(0);
    if (i >= n_el) return;
    dst[i] = src[i];
}
