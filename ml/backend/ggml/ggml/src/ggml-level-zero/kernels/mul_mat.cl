// SPDX-License-Identifier: MIT
// mul_mat.cl — Stride-aware, 3D-batched matrix-multiplication kernels.
//
// Compiled AOT to SPIR-V by ocloc.
//
// Four entry points (API Contract ADR-L0-001 §4.2, catalog entries 1–4):
//   mul_mat_f32  — F32 x F32  → F32   (entry #1)
//   mul_mat_f16  — F16 x F32  → F32   (entry #2)
//   mul_mat_q8_0 — Q8_0 x F32 → F32   (entry #3)
//   mul_mat_q4_0 — Q4_0 x F32 → F32   (entry #4)
//
// Argument contract (ADR §3.5, §9.1–9.4):
//   arg 0: mul_mat_pc push-constant struct (by value, 160 bytes)
//   arg 1: A    — weight matrix (type varies per kernel)
//   arg 2: B    — __global const float* activations [K x N] GGML layout
//   arg 3: D    — __global float* output             [M x N] GGML layout
//
// Indexing macro (ADR §3.2 — BINDING, no kernel may deviate):
//   IDX(i0,i1,i2,i3,nb) = i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
//   nb[] values are in BYTES; result is a BYTE offset from the base pointer.
//
// Batch loop convention (ADR §5):
//   global Z encodes (i2, i3) pair: i2 = gid_z % ne_d[2], i3 = gid_z / ne_d[2]
//   Broadcast: broadcast_aK flag → i2_a or i3_a is pinned to 0 for all Z values.
//
// Quantised block layouts (matching ggml-common.h exactly):
//
//   block_q8_0 (Q8_0_BLOCK_BYTES = 34 bytes):
//     bytes 0-1  : ggml_half (float16) scale  d
//     bytes 2-33 : int8[32]            quants qs  (32 elements)
//
//   block_q4_0 (Q4_0_BLOCK_BYTES = 18 bytes):
//     bytes 0-1  : ggml_half (float16) scale  d
//     bytes 2-17 : uint8[16] packed nibbles   qs  (2 nibbles per byte, 32 elems)
//
// Scale dequantisation uses a manual IEEE-754 uint16→float32 conversion to
// avoid any dependency on the cl_khr_fp16 extension for pointer loads.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE  16
#define QK4_0 32
#define QK8_0 32

// Byte sizes of the GGML quantised blocks (must match ggml-common.h).
#define Q4_0_BLOCK_BYTES 18u   // 2-byte half scale + 16 nibble bytes
#define Q8_0_BLOCK_BYTES 34u   // 2-byte half scale + 32 int8  bytes

// ---------------------------------------------------------------------------
// Stride-aware element address macro (ADR §3.2).
// nb[] are byte strides; returns byte offset from the tensor base pointer.
// Usage: *(T*)((char*)base_ptr + IDX4(i0,i1,i2,i3,nb))
// ---------------------------------------------------------------------------
#define IDX4(i0,i1,i2,i3,nb) \
    ((long)(i0)*(long)(nb)[0] + (long)(i1)*(long)(nb)[1] + \
     (long)(i2)*(long)(nb)[2] + (long)(i3)*(long)(nb)[3])

// ---------------------------------------------------------------------------
// Manual IEEE-754 F16→F32 bit-conversion.
// Reads a uint16 value and converts to float without using the half data type
// for pointer loads.  Works correctly for all normal/zero values.
// Subnormals and infinities are mapped to 0.0f (safe for weight scales).
// ---------------------------------------------------------------------------
#define F16_BITS_TO_F32(h16) ({ \
    uint _h = (uint)(h16);           \
    uint _e = (_h >> 10u) & 0x1Fu;  \
    as_float(_e == 0u ? 0u :         \
             ((_h & 0x8000u) << 16u) | \
             ((_e + 112u) << 23u)    | \
             ((_h & 0x03FFu) << 13u)); \
})

// ---------------------------------------------------------------------------
// Push-constant struct family 1 — mul_mat_pc (ADR §3.4, 160 bytes).
// Bound at arg index 0 for all four mul_mat variants.
// ---------------------------------------------------------------------------
typedef struct {
    int  ne_a[4];          // src0 element counts (A, weight)
    int  ne_b[4];          // src1 element counts (B, activation)
    int  ne_d[4];          // dst  element counts (D, output)
    long nb_a[4];          // src0 byte strides
    long nb_b[4];          // src1 byte strides
    long nb_d[4];          // dst  byte strides
    int  broadcast_a2;     // 1 if ne_a[2]==1 && ne_b[2]>1 (A broadcast on dim 2)
    int  broadcast_a3;     // 1 if ne_a[3]==1 && ne_b[3]>1 (A broadcast on dim 3)
    int  broadcast_b2;     // 1 if ne_b[2]==1 && ne_a[2]>1 (B broadcast on dim 2)
    int  broadcast_b3;     // 1 if ne_b[3]==1 && ne_a[3]>1 (B broadcast on dim 3)
} mul_mat_pc;
// Static size assertion: sizeof(mul_mat_pc) == 160 bytes.
// 3*4*sizeof(int) = 48 B, 3*4*sizeof(long) = 96 B, 4*sizeof(int) = 16 B → total 160 B.

// ---------------------------------------------------------------------------
// F32 x F32 → F32  (entry #1, ADR §9.1)
//
// D[i0,i1,i2,i3] = sum_k A[k,i0,i2_a,i3_a] * B[k,i1,i2_b,i3_b]
//   i0 = col (output feature / M dimension)
//   i1 = row (batch / N dimension)
//   i2,i3 decoded from global Z; broadcast flags select i2_a/i3_a from pc
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void mul_mat_f32(
    mul_mat_pc                   pc,
    __global const float        *A,
    __global const float        *B,
    __global       float        *D)
{
    int col   = get_global_id(0);   // i0: output column (M dimension)
    int row   = get_global_id(1);   // i1: output row    (N dimension)
    int gid_z = (int)get_group_id(2);

    if (col >= pc.ne_d[0] || row >= pc.ne_d[1]) return;

    // Decode (i2, i3) from flattened Z (ADR §5.3).
    int i2 = gid_z % pc.ne_d[2];
    int i3 = gid_z / pc.ne_d[2];

    // Apply broadcast flags (ADR §5.2).
    int i2_a = pc.broadcast_a2 ? 0 : i2;
    int i3_a = pc.broadcast_a3 ? 0 : i3;
    int i2_b = pc.broadcast_b2 ? 0 : i2;
    int i3_b = pc.broadcast_b3 ? 0 : i3;

    // Base pointers for this (i2,i3) batch slice.
    // IDX4 returns byte offset; cast to char* then index.
    int K = pc.ne_a[0];

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // A element (k, col, i2_a, i3_a) — nb_a[0] = sizeof(float) = 4
        long addr_a = IDX4(k, col, i2_a, i3_a, pc.nb_a);
        float a_val = *(const __global float *)((const __global char *)A + addr_a);

        // B element (k, row, i2_b, i3_b) — nb_b[0] = sizeof(float) = 4
        long addr_b = IDX4(k, row, i2_b, i3_b, pc.nb_b);
        float b_val = *(const __global float *)((const __global char *)B + addr_b);

        acc += a_val * b_val;
    }

    // Output D element (col, row, i2, i3)
    long addr_d = IDX4(col, row, i2, i3, pc.nb_d);
    *((__global float *)(((__global char *)D) + addr_d)) = acc;
}

// ---------------------------------------------------------------------------
// F16 x F32 → F32  (entry #2, ADR §9.2, CUDA §9.2)
//
// A loaded via vload_half (F16→F32 at load); accumulator stays F32.
// Output is F32 (dst is always F32 for mul_mat variants, ADR §4.2).
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void mul_mat_f16(
    mul_mat_pc                   pc,
    __global const half         *A,
    __global const float        *B,
    __global       float        *D)
{
    int col   = get_global_id(0);
    int row   = get_global_id(1);
    int gid_z = (int)get_group_id(2);

    if (col >= pc.ne_d[0] || row >= pc.ne_d[1]) return;

    int i2 = gid_z % pc.ne_d[2];
    int i3 = gid_z / pc.ne_d[2];

    int i2_a = pc.broadcast_a2 ? 0 : i2;
    int i3_a = pc.broadcast_a3 ? 0 : i3;
    int i2_b = pc.broadcast_b2 ? 0 : i2;
    int i3_b = pc.broadcast_b3 ? 0 : i3;

    int K = pc.ne_a[0];

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // F16 load via vload_half at byte-addressed offset.
        // nb_a[0] = sizeof(half) = 2 for F16 tensors.
        long addr_a = IDX4(k, col, i2_a, i3_a, pc.nb_a);
        // vload_half takes an element index and a half* base; divide byte offset by 2.
        float a_val = vload_half((size_t)(addr_a / 2L),
                                 (const __global half *)((const __global char *)A));

        long addr_b = IDX4(k, row, i2_b, i3_b, pc.nb_b);
        float b_val = *(const __global float *)((const __global char *)B + addr_b);

        acc += a_val * b_val;
    }

    long addr_d = IDX4(col, row, i2, i3, pc.nb_d);
    *((__global float *)(((__global char *)D) + addr_d)) = acc;
}

// ---------------------------------------------------------------------------
// Q8_0 x F32 → F32  (entry #3, ADR §9.3, CUDA §9.3)
//
// Block layout (Q8_0_BLOCK_BYTES = 34 bytes):
//   bytes 0-1  : float16 scale d (uint16, converted via F16_BITS_TO_F32)
//   bytes 2-33 : int8[32] quants qs
//
// Dequant: w_i = (float)(qs[i]) * d
// Block index: kbx = k >> 5  (k / QK8_0, using bit-shift per ADR §7 DSA)
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void mul_mat_q8_0(
    mul_mat_pc                   pc,
    __global const uchar        *A,
    __global const float        *B,
    __global       float        *D)
{
    int col   = get_global_id(0);   // M dimension (weight row index = output column)
    int row   = get_global_id(1);   // N dimension (activation row / batch element)
    int gid_z = (int)get_group_id(2);

    if (col >= pc.ne_d[0] || row >= pc.ne_d[1]) return;

    int i2 = gid_z % pc.ne_d[2];
    int i3 = gid_z / pc.ne_d[2];

    int i2_a = pc.broadcast_a2 ? 0 : i2;
    int i3_a = pc.broadcast_a3 ? 0 : i3;
    int i2_b = pc.broadcast_b2 ? 0 : i2;
    int i3_b = pc.broadcast_b3 ? 0 : i3;

    int K  = pc.ne_a[0];           // inner dimension (must be multiple of QK8_0)
    int nb = K / QK8_0;            // number of Q8_0 blocks per row

    // Base byte offset for this weight row (col) in the quantised A tensor.
    // nb_a[1] is the byte stride between consecutive quantised rows.
    // nb_a[0] = Q8_0_BLOCK_BYTES describes block stride (ADR §9.3 note).
    long a_row_base = (long)col * (long)pc.nb_a[1]
                    + (long)i2_a * (long)pc.nb_a[2]
                    + (long)i3_a * (long)pc.nb_a[3];

    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        // Block byte offset within row b (ADR §7 DSA: k >> 5 for block index).
        long block_base = a_row_base + (long)b * Q8_0_BLOCK_BYTES;

        // Scale: float16 at bytes 0-1 of block (manual F16→F32 conversion).
        uint h16 = (uint)(*(__global const ushort *)(A + block_base));
        float d  = F16_BITS_TO_F32(h16);

        // Inner loop over QK8_0 = 32 elements within the block.
        for (int i = 0; i < QK8_0; ++i) {
            char  q = (char)(A[block_base + 2u + i]);
            float w = (float)q * d;

            // B element: (b * QK8_0 + i, row, i2_b, i3_b)
            int   k_elem  = b * QK8_0 + i;
            long  addr_b  = IDX4(k_elem, row, i2_b, i3_b, pc.nb_b);
            float b_val   = *(const __global float *)((const __global char *)B + addr_b);

            acc += w * b_val;
        }
    }

    // Write output (stride-aware).
    long addr_d = IDX4(col, row, i2, i3, pc.nb_d);
    *((__global float *)(((__global char *)D) + addr_d)) = acc;
}

// ---------------------------------------------------------------------------
// Q4_0 x F32 → F32  (entry #4, ADR §9.4, CUDA §9.4)
//
// Block layout (Q4_0_BLOCK_BYTES = 18 bytes):
//   bytes 0-1  : float16 scale d (uint16, converted via F16_BITS_TO_F32)
//   bytes 2-17 : uint8[16] packed nibbles (2 elements per byte, 32 total)
//
// Dequant element 2i  : ((packed & 0x0F) - 8) * d   low nibble
// Dequant element 2i+1: ((packed >> 4)   - 8) * d   high nibble
// Block index: kbx = k >> 5  (ADR §7 DSA: bit-shift for O(1) block indexing)
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void mul_mat_q4_0(
    mul_mat_pc                   pc,
    __global const uchar        *A,
    __global const float        *B,
    __global       float        *D)
{
    int col   = get_global_id(0);
    int row   = get_global_id(1);
    int gid_z = (int)get_group_id(2);

    if (col >= pc.ne_d[0] || row >= pc.ne_d[1]) return;

    int i2 = gid_z % pc.ne_d[2];
    int i3 = gid_z / pc.ne_d[2];

    int i2_a = pc.broadcast_a2 ? 0 : i2;
    int i3_a = pc.broadcast_a3 ? 0 : i3;
    int i2_b = pc.broadcast_b2 ? 0 : i2;
    int i3_b = pc.broadcast_b3 ? 0 : i3;

    int K  = pc.ne_a[0];
    int nb = K / QK4_0;

    // Base byte offset for this weight row in the quantised A tensor.
    long a_row_base = (long)col * (long)pc.nb_a[1]
                    + (long)i2_a * (long)pc.nb_a[2]
                    + (long)i3_a * (long)pc.nb_a[3];

    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        long block_base = a_row_base + (long)b * Q4_0_BLOCK_BYTES;

        // Scale: float16 at bytes 0-1.
        uint h16 = (uint)(*(__global const ushort *)(A + block_base));
        float d  = F16_BITS_TO_F32(h16);

        // Nibble loop: 16 bytes → 32 elements.
        for (int i = 0; i < (QK4_0 / 2); ++i) {
            uchar packed = A[block_base + 2u + i];
            float w0 = (float)((int)(packed & 0x0Fu) - 8) * d;
            float w1 = (float)((int)((packed >> 4u) & 0x0Fu) - 8) * d;

            int  k0 = b * QK4_0 + i * 2;
            int  k1 = k0 + 1;

            long addr_b0 = IDX4(k0, row, i2_b, i3_b, pc.nb_b);
            long addr_b1 = IDX4(k1, row, i2_b, i3_b, pc.nb_b);
            float act0 = *(const __global float *)((const __global char *)B + addr_b0);
            float act1 = *(const __global float *)((const __global char *)B + addr_b1);

            acc += w0 * act0 + w1 * act1;
        }
    }

    long addr_d = IDX4(col, row, i2, i3, pc.nb_d);
    *((__global float *)(((__global char *)D) + addr_d)) = acc;
}
