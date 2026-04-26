// SPDX-License-Identifier: MIT
// gelu_silu.cl — Element-wise activation, arithmetic, and copy kernels for the
//                Intel Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file gelu_silu.cl -device <target> -options "-cl-std=CL2.0 -O3"
//
// Eight entry points compiled from this file:
//   gelu_f32  — Gaussian Error Linear Unit via erf formula
//   silu_f32  — Sigmoid Linear Unit (x * sigmoid(x))
//   copy_f32  — element-wise contiguous copy
//   add_f32   — stride-aware element-wise F32 add with broadcast (RC3 fix)
//   add_f16   — stride-aware element-wise F16 add with broadcast (RC3 fix)
//   mul_f32   — stride-aware element-wise F32 multiply with broadcast (RC3 fix)
//   mul_f16   — RESERVED (stub only; see ADR-L0-002)
//
// Note on add/mul kernel placement (phase-c-worksplit.md §3.2):
//   The work-split explicitly authorises fpga-engineer to add add_f32, add_f16,
//   mul_f32 to this file OR to create a new binop.cl if embedded-squad-lead
//   approves.  These kernels are semantically unrelated to gelu/silu activations
//   but are placed here to avoid creating a new file without formal authorisation.
//   A future refactor may extract them to binop.cl under ADR-L0-002.
//
// Argument binding for gelu_f32, silu_f32, copy_f32 (unchanged):
//   arg 0: src  (__global const float*) — input  src0->data
//   arg 1: dst  (__global float*)       — output node->data
//   arg 2: n_el (uint)                  — total element count
//
// Argument binding for add_f32, add_f16, mul_f32 (new stride-aware):
//   arg 0: pc_raw (__constant int*)     — ze_binop_pc as raw int array (144 B)
//   arg 1: a    (__global const T*)     — src0->data
//   arg 2: b    (__global const T*)     — src1->data
//   arg 3: d    (__global T*)           — dst->data
//
// Push-constant fields for binop kernels (ze_binop_pc, see ze_buffer.hpp):
//   ne_a[4]  — src0 element counts
//   ne_b[4]  — src1 element counts
//   ne_d[4]  — dst  element counts (== ne_a for add/mul)
//   nb_a[4]  — src0 byte strides
//   nb_b[4]  — src1 byte strides (0 = broadcast on that dimension)
//   nb_d[4]  — dst  byte strides
//
// Broadcast convention (Null Object pattern, ADR §8):
//   nb_b[k] = 0 means src1 does not advance on dimension k — natural broadcast.
//   The IDX computation `i_k * 0 = 0` keeps the address at the row/slice base.
//   No explicit branch is required in the kernel body.
//
// Work-group size: (256, 1, 1) for all kernels.
// Global size for binop: (round_up(ne_d_total, 256), 1, 1).

// 1/sqrt(2) constant used by the GELU erf formula.
#define M_SQRT1_2_F  0.70710678118654752440f

// ---------------------------------------------------------------------------
// GELU activation using the exact erf-based formula.
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
// Contiguous F32 tensor copy.
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

// =============================================================================
// Stride-aware binop kernels — add_f32, add_f16, mul_f32, mul_f16 (RESERVED).
//
// These replace the old contiguous-only add_f32 / mul_f32 that lived here.
// They implement the RC3 fix (stride-aware indexing) and broadcast support
// (Null Object pattern via zero byte strides on broadcast dimensions).
// =============================================================================

// ---------------------------------------------------------------------------
// Internal helper: decode a flat linear element index into 4D coordinates and
// compute the byte address from the push-constant fields.
//
// idx        — flat element index in [0, ne_d_total)
// ne_d[4]    — destination dimension sizes
// nb_X[0..3] — byte strides for tensor X
//
// Returns the byte offset from the tensor base pointer.
// ---------------------------------------------------------------------------
static inline long binop_byte_addr(uint idx,
                                   const int ne_d0, const int ne_d1,
                                   const int ne_d2,
                                   const long nb0, const long nb1,
                                   const long nb2, const long nb3)
{
    int i0 = (int)(idx % (uint)ne_d0);
    int i1 = (int)((idx / (uint)ne_d0) % (uint)ne_d1);
    int i2 = (int)((idx / ((uint)ne_d0 * (uint)ne_d1)) % (uint)ne_d2);
    int i3 = (int)(idx / ((uint)ne_d0 * (uint)ne_d1 * (uint)ne_d2));
    return (long)i0 * nb0 + (long)i1 * nb1 + (long)i2 * nb2 + (long)i3 * nb3;
}

// ---------------------------------------------------------------------------
// add_f32 — stride-aware F32 element-wise add with broadcast.
//
// RC3 fix: replaces the old contiguous-only `c[i] = a[i] + b[i]` with full
// IDX(i0,i1,i2,i3,nb) byte-stride addressing.
// Broadcast: nb_b[k]=0 causes src1 address to stay at base for dimension k.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void add_f32(
    __constant int          *pc_raw,   // ze_binop_pc (arg 0)
    __global const float    *a,        // src0 (arg 1)
    __global const float    *b,        // src1 (arg 2)
    __global       float    *d)        // dst  (arg 3)
{
    // Unpack ze_binop_pc.
    // Layout: ne_a[4](0..3), ne_b[4](4..7), ne_d[4](8..11),
    //         nb_a[4](12..19), nb_b[4](20..27), nb_d[4](28..35) — all int32 indices.
    int ne_d0 = pc_raw[8];
    int ne_d1 = pc_raw[9];
    int ne_d2 = pc_raw[10];
    int ne_d3 = pc_raw[11];

    long nb_a0 = (long)pc_raw[12] | ((long)pc_raw[13] << 32);
    long nb_a1 = (long)pc_raw[14] | ((long)pc_raw[15] << 32);
    long nb_a2 = (long)pc_raw[16] | ((long)pc_raw[17] << 32);
    long nb_a3 = (long)pc_raw[18] | ((long)pc_raw[19] << 32);

    long nb_b0 = (long)pc_raw[20] | ((long)pc_raw[21] << 32);
    long nb_b1 = (long)pc_raw[22] | ((long)pc_raw[23] << 32);
    long nb_b2 = (long)pc_raw[24] | ((long)pc_raw[25] << 32);
    long nb_b3 = (long)pc_raw[26] | ((long)pc_raw[27] << 32);

    long nb_d0 = (long)pc_raw[28] | ((long)pc_raw[29] << 32);
    long nb_d1 = (long)pc_raw[30] | ((long)pc_raw[31] << 32);
    long nb_d2 = (long)pc_raw[32] | ((long)pc_raw[33] << 32);
    long nb_d3 = (long)pc_raw[34] | ((long)pc_raw[35] << 32);

    uint idx = get_global_id(0);
    uint ne_total = (uint)ne_d0 * (uint)ne_d1 * (uint)ne_d2 * (uint)ne_d3;
    if (idx >= ne_total) return;

    long addr_a = binop_byte_addr(idx, ne_d0, ne_d1, ne_d2, nb_a0, nb_a1, nb_a2, nb_a3);
    long addr_b = binop_byte_addr(idx, ne_d0, ne_d1, ne_d2, nb_b0, nb_b1, nb_b2, nb_b3);
    long addr_d = binop_byte_addr(idx, ne_d0, ne_d1, ne_d2, nb_d0, nb_d1, nb_d2, nb_d3);

    float va = *(__global const float *)((__global const char *)a + addr_a);
    float vb = *(__global const float *)((__global const char *)b + addr_b);
    *(__global float *)((__global char *)d + addr_d) = va + vb;
}

// ---------------------------------------------------------------------------
// add_f16 — stride-aware F16 element-wise add with broadcast.
//
// F16 load -> F32 intermediate sum -> F16 store (round-to-nearest-even via
// vstore_half).  Push-constant layout identical to add_f32.
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void add_f16(
    __constant int          *pc_raw,   // ze_binop_pc (arg 0)
    __global const half     *a,        // src0 F16 (arg 1)
    __global const half     *b,        // src1 F16 (arg 2)
    __global       half     *d)        // dst  F16 (arg 3)
{
    int ne_d0 = pc_raw[8];
    int ne_d1 = pc_raw[9];
    int ne_d2 = pc_raw[10];
    int ne_d3 = pc_raw[11];

    long nb_a0 = (long)pc_raw[12] | ((long)pc_raw[13] << 32);
    long nb_a1 = (long)pc_raw[14] | ((long)pc_raw[15] << 32);
    long nb_a2 = (long)pc_raw[16] | ((long)pc_raw[17] << 32);
    long nb_a3 = (long)pc_raw[18] | ((long)pc_raw[19] << 32);

    long nb_b0 = (long)pc_raw[20] | ((long)pc_raw[21] << 32);
    long nb_b1 = (long)pc_raw[22] | ((long)pc_raw[23] << 32);
    long nb_b2 = (long)pc_raw[24] | ((long)pc_raw[25] << 32);
    long nb_b3 = (long)pc_raw[26] | ((long)pc_raw[27] << 32);

    long nb_d0 = (long)pc_raw[28] | ((long)pc_raw[29] << 32);
    long nb_d1 = (long)pc_raw[30] | ((long)pc_raw[31] << 32);
    long nb_d2 = (long)pc_raw[32] | ((long)pc_raw[33] << 32);
    long nb_d3 = (long)pc_raw[34] | ((long)pc_raw[35] << 32);

    uint idx = get_global_id(0);
    uint ne_total = (uint)ne_d0 * (uint)ne_d1 * (uint)ne_d2 * (uint)ne_d3;
    if (idx >= ne_total) return;

    long addr_a = binop_byte_addr(idx, ne_d0, ne_d1, ne_d2, nb_a0, nb_a1, nb_a2, nb_a3);
    long addr_b = binop_byte_addr(idx, ne_d0, ne_d1, ne_d2, nb_b0, nb_b1, nb_b2, nb_b3);
    long addr_d = binop_byte_addr(idx, ne_d0, ne_d1, ne_d2, nb_d0, nb_d1, nb_d2, nb_d3);

    float va = vload_half(0, (__global const half *)((__global const char *)a + addr_a));
    float vb = vload_half(0, (__global const half *)((__global const char *)b + addr_b));
    vstore_half(va + vb, 0, (__global half *)((__global char *)d + addr_d));
}

// ---------------------------------------------------------------------------
// mul_f32 — stride-aware F32 element-wise multiply with broadcast.
//
// Identical address logic to add_f32; only the binary operation differs.
// mul_f16 is NOT implemented in v1 (see stub below and ADR §9.13).
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mul_f32(
    __constant int          *pc_raw,   // ze_binop_pc (arg 0)
    __global const float    *a,        // src0 (arg 1)
    __global const float    *b,        // src1 (arg 2)
    __global       float    *d)        // dst  (arg 3)
{
    int ne_d0 = pc_raw[8];
    int ne_d1 = pc_raw[9];
    int ne_d2 = pc_raw[10];
    int ne_d3 = pc_raw[11];

    long nb_a0 = (long)pc_raw[12] | ((long)pc_raw[13] << 32);
    long nb_a1 = (long)pc_raw[14] | ((long)pc_raw[15] << 32);
    long nb_a2 = (long)pc_raw[16] | ((long)pc_raw[17] << 32);
    long nb_a3 = (long)pc_raw[18] | ((long)pc_raw[19] << 32);

    long nb_b0 = (long)pc_raw[20] | ((long)pc_raw[21] << 32);
    long nb_b1 = (long)pc_raw[22] | ((long)pc_raw[23] << 32);
    long nb_b2 = (long)pc_raw[24] | ((long)pc_raw[25] << 32);
    long nb_b3 = (long)pc_raw[26] | ((long)pc_raw[27] << 32);

    long nb_d0 = (long)pc_raw[28] | ((long)pc_raw[29] << 32);
    long nb_d1 = (long)pc_raw[30] | ((long)pc_raw[31] << 32);
    long nb_d2 = (long)pc_raw[32] | ((long)pc_raw[33] << 32);
    long nb_d3 = (long)pc_raw[34] | ((long)pc_raw[35] << 32);

    uint idx = get_global_id(0);
    uint ne_total = (uint)ne_d0 * (uint)ne_d1 * (uint)ne_d2 * (uint)ne_d3;
    if (idx >= ne_total) return;

    long addr_a = binop_byte_addr(idx, ne_d0, ne_d1, ne_d2, nb_a0, nb_a1, nb_a2, nb_a3);
    long addr_b = binop_byte_addr(idx, ne_d0, ne_d1, ne_d2, nb_b0, nb_b1, nb_b2, nb_b3);
    long addr_d = binop_byte_addr(idx, ne_d0, ne_d1, ne_d2, nb_d0, nb_d1, nb_d2, nb_d3);

    float va = *(__global const float *)((__global const char *)a + addr_a);
    float vb = *(__global const float *)((__global const char *)b + addr_b);
    *(__global float *)((__global char *)d + addr_d) = va * vb;
}

// ---------------------------------------------------------------------------
// mul_f16 — RESERVED — ADR-L0-002
//
// This kernel slot is reserved for future implementation.  The dispatcher
// (ggml-level-zero.cpp, Group C1) returns false from supports_op() for
// GGML_OP_MUL with F16×F16 operands, routing those operations to the CPU
// backend in v1.  Do NOT add a kernel body here until ADR-L0-002 is approved.
// ---------------------------------------------------------------------------
/* RESERVED — ADR-L0-002 */
// __kernel void mul_f16(...) { /* body intentionally omitted in v1 */ }
