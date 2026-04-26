// SPDX-License-Identifier: MIT
// kv_cache.cl — KV-cache stride-aware copy and SET_ROWS kernels for the Intel
//               Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file kv_cache.cl -spv_only -output_no_suffix -output kv_cache
//
// Entry points:
//   kv_cache_write  — stride-aware F32->F32 copy into the cache
//   kv_cache_read   — stride-aware F32->F32 copy from the cache
//   set_rows_f32    — SET_ROWS scatter-write: F32 src -> F32 dst at I32 row indices
//   set_rows_f16    — SET_ROWS scatter-write: F32 src -> F16 dst at I32 row indices
//
// Work-item mapping: one work-item per element.
// Boundary guard: if (i >= ne_total) return.
//
// set_rows argument contract:
//   arg 0: src0        (__global const float*)  — source rows (F32)
//   arg 1: src1        (__global const int*)    — row index table (I32)
//   arg 2: dst         (__global float/half*)   — destination cache buffer
//   arg 3: ne00        (int) — elements per row in src0
//   arg 4: ne01        (int) — number of rows in src0 per batch
//   arg 5: ne02        (int) — batch size (dim 2)
//   arg 6: ne03        (int) — outer batch (dim 3)
//   arg 7: s01         (int) — src0 row stride in elements
//   arg 8: s02         (int) — src0 batch stride in elements
//   arg 9: s03         (int) — src0 outer-batch stride in elements
//   arg 10: s10        (int) — src1 row stride in indices
//   arg 11: s11        (int) — src1 batch stride in indices
//   arg 12: s12        (int) — src1 outer-batch stride in indices
//   arg 13: s1         (int) — dst row stride in elements (nb1/sizeof(element))
//   arg 14: s2         (int) — dst batch stride in elements
//   arg 15: s3         (int) — dst outer-batch stride in elements

// ---------------------------------------------------------------------------
// Stride-aware copy from source into the KV cache destination.
//
// Copies n_elements float values using independent strides on source and
// destination.  A stride value of 1 gives contiguous access; larger values
// allow scatter/gather patterns for non-contiguous cache layouts.
// ---------------------------------------------------------------------------
__kernel
void kv_cache_write(
    __global const float *src,
    __global       float *dst,
    uint n_elements,
    uint src_stride,
    uint dst_stride)
{
    uint i = get_global_id(0);
    if (i >= n_elements) return;
    dst[i * dst_stride] = src[i * src_stride];
}

// ---------------------------------------------------------------------------
// Stride-aware copy from the KV cache source into a destination buffer.
//
// Mirror of kv_cache_write with roles reversed.  The same stride contract
// applies: element i of the output is read from src at index i * src_stride.
// ---------------------------------------------------------------------------
__kernel
void kv_cache_read(
    __global const float *src,
    __global       float *dst,
    uint n_elements,
    uint src_stride,
    uint dst_stride)
{
    uint i = get_global_id(0);
    if (i >= n_elements) return;
    dst[i * dst_stride] = src[i * src_stride];
}

// ---------------------------------------------------------------------------
// SET_ROWS F32->F32: scatter-write src0 rows into dst at row indices from src1.
//
// Mirrors ggml_compute_forward_set_rows_f32 (ggml-cpu/ops.cpp).
// Global work size: ne00 * ne01 * ne02 * ne03.
//
// Stride convention (element strides, not byte strides):
//   s01 = src0->nb[1]/sizeof(float)  — src0 row stride
//   s02, s03 — src0 batch/outer strides
//   s10 = src0->nb[0]/sizeof(idx)    — src1 element stride (1 for contiguous)
//   s11 = src0->nb[1]/sizeof(idx)    — src1 row-1 stride
//   s12 = src0->nb[2]/sizeof(idx)    — src1 row-2 stride
//   ne11, ne12                        — src1 dimensions for modulo broadcast
//   s1, s2, s3 — dst element strides
// ---------------------------------------------------------------------------
__kernel
void set_rows_f32(
    __global const float *src0,
    __global const int   *src1,
    __global       float *dst,
    int ne00, int ne01, int ne02, int ne03,
    int ne11, int ne12,
    int s01, int s02, int s03,
    int s10, int s11, int s12,
    int s1,  int s2,  int s3)
{
    int i = get_global_id(0);
    int ne_total = ne00 * ne01 * ne02 * ne03;
    if (i >= ne_total) return;

    int i00  = i % ne00;
    int tmp  = i / ne00;
    int i01  = tmp % ne01;
    tmp      = tmp / ne01;
    int i02  = tmp % ne02;
    int i03  = tmp / ne02;

    int i10 = i01;
    int i11 = (ne11 > 0) ? (i02 % ne11) : 0;
    int i12 = (ne12 > 0) ? (i03 % ne12) : 0;

    int dst_row  = src1[i10 * s10 + i11 * s11 + i12 * s12];
    float val    = src0[i01 * s01 + i02 * s02 + i03 * s03 + i00];
    dst[dst_row * s1 + i02 * s2 + i03 * s3 + i00] = val;
}

// ---------------------------------------------------------------------------
// SET_ROWS F32->F16: scatter-write src0 rows (F32) into dst (F16) at row
// indices from src1.  Required for flash-attention KV cache (F16 storage).
// ---------------------------------------------------------------------------
__kernel
void set_rows_f16(
    __global const float  *src0,
    __global const int    *src1,
    __global       ushort *dst,
    int ne00, int ne01, int ne02, int ne03,
    int ne11, int ne12,
    int s01, int s02, int s03,
    int s10, int s11, int s12,
    int s1,  int s2,  int s3)
{
    int i = get_global_id(0);
    int ne_total = ne00 * ne01 * ne02 * ne03;
    if (i >= ne_total) return;

    int i00  = i % ne00;
    int tmp  = i / ne00;
    int i01  = tmp % ne01;
    tmp      = tmp / ne01;
    int i02  = tmp % ne02;
    int i03  = tmp / ne02;

    int i10 = i01;
    int i11 = (ne11 > 0) ? (i02 % ne11) : 0;
    int i12 = (ne12 > 0) ? (i03 % ne12) : 0;

    int dst_row = src1[i10 * s10 + i11 * s11 + i12 * s12];
    float val   = src0[i01 * s01 + i02 * s02 + i03 * s03 + i00];

    int dst_idx = dst_row * s1 + i02 * s2 + i03 * s3 + i00;
    vstore_half(val, dst_idx, (__global half *)dst);
}
