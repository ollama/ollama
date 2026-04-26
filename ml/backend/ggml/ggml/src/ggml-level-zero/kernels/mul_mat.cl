// SPDX-License-Identifier: MIT
// mul_mat.cl — Matrix-multiplication kernels for the Intel Level Zero / OpenCL backend.
//
// Compiled AOT to SPIR-V by ocloc.
//
// Four entry points:
//   mul_mat_f32  — F32 x F32  → F32
//   mul_mat_f16  — F16 x F32  → F32
//   mul_mat_q8_0 — Q8_0 x F32 → F32
//   mul_mat_q4_0 — Q4_0 x F32 → F32
//
// Argument contract:
//   arg 0: A        — weight matrix (type varies)
//   arg 1: B        — __global const float* activations  [N x K] GGML layout
//   arg 2: C        — __global float* output             [N x M] GGML layout
//   arg 3: M (uint) — src0->ne[1]  (weight rows / output features)
//   arg 4: N (uint) — src1->ne[1]  (batch size / sequence length)
//   arg 5: K (uint) — src0->ne[0]  (inner dimension / input features)
//
// GGML memory layout — ne[0] is the FASTEST (innermost, contiguous) dimension:
//
//   A (src0, weights):  element (k, m) at A[m * K + k]          ne[0]=K, ne[1]=M
//   B (src1, acts):     element (k, n) at B[n * K + k]          ne[0]=K, ne[1]=N
//   C (dst,  output):   element (m, n) at C[n * M + m]          ne[0]=M, ne[1]=N
//
//   Computation: C[m, n] = sum_k A[m*K+k] * B[n*K+k]
//     col = get_global_id(0) → m index (0..M-1)
//     row = get_global_id(1) → n index (0..N-1)
//
// Quantised block layouts (matching ggml-common.h exactly):
//
//   block_q4_0 (18 bytes):
//     bytes 0-1  : ggml_half (float16) scale  d   ← uint16, NOT float32
//     bytes 2-17 : uint8[16] packed nibbles   qs  (2 nibbles per byte, 32 elems)
//
//   block_q8_0 (34 bytes):
//     bytes 0-1  : ggml_half (float16) scale  d   ← uint16, NOT float32
//     bytes 2-33 : int8[32]            quants qs  (32 elements)
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
// Manual IEEE-754 F16→F32 bit-conversion.
// Reads 2 bytes from A[base+0..1] as uint16, converts to float without
// using the half data type.  Works correctly for all normal/zero values.
// Subnormals and infinities are mapped to 0.0f (safe for weight scales).
// ---------------------------------------------------------------------------
#define F16_TO_F32(A, base) ({ \
    uint _h16  = (uint)(*(__global const ushort *)((A) + (base))); \
    uint _e    = (_h16 >> 10u) & 0x1Fu;                           \
    uint _bits = (_e == 0u) ? 0u :                                 \
                 ((_h16 & 0x8000u) << 16u) |                       \
                 ((_e + 112u) << 23u) |                            \
                 ((_h16 & 0x03FFu) << 13u);                        \
    as_float(_bits); \
})

// ---------------------------------------------------------------------------
// F32 x F32 → F32
// C[n*M+m] = sum_k A[m*K+k] * B[n*K+k]
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void mul_mat_f32(
    __global const float *A,
    __global const float *B,
    __global       float *C,
    uint M, uint N, uint K)
{
    uint col = get_global_id(0);   // m
    uint row = get_global_id(1);   // n
    if (col >= M || row >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k)
        acc += A[col * K + k] * B[row * K + k];
    C[row * M + col] = acc;
}

// ---------------------------------------------------------------------------
// F16 x F32 → F32
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void mul_mat_f16(
    __global const half  *A,
    __global const float *B,
    __global       float *C,
    uint M, uint N, uint K)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    if (col >= M || row >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k)
        acc += vload_half(col * K + k, A) * B[row * K + k];
    C[row * M + col] = acc;
}

// ---------------------------------------------------------------------------
// Q8_0 x F32 → F32
//
// Block layout (Q8_0_BLOCK_BYTES = 34 bytes):
//   bytes 0-1  : float16 scale d (as uint16, converted via F16_TO_F32)
//   bytes 2-33 : int8[32] quants qs
//
// Dequant: w_i = (float)(qs[i]) * d
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void mul_mat_q8_0(
    __global const uchar *A,
    __global const float *B,
    __global       float *C,
    uint M, uint N, uint K)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    if (col >= M || row >= N) return;

    float acc = 0.0f;
    uint  nb  = K / QK8_0;

    for (uint b = 0; b < nb; ++b) {
        uint  base = (col * nb + b) * Q8_0_BLOCK_BYTES;
        // Scale: float16 at bytes 0-1, converted without half pointer.
        uint  h16  = (uint)(*(__global const ushort *)(A + base));
        uint  e    = (h16 >> 10u) & 0x1Fu;
        float d    = as_float(e == 0u ? 0u :
                        ((h16 & 0x8000u) << 16u) | ((e + 112u) << 23u) | ((h16 & 0x03FFu) << 13u));

        for (uint i = 0; i < QK8_0; ++i) {
            char  q   = (char)(A[base + 2u + i]);
            float w   = (float)q * d;
            float act = B[row * K + b * QK8_0 + i];
            acc += w * act;
        }
    }
    C[row * M + col] = acc;
}

// ---------------------------------------------------------------------------
// Q4_0 x F32 → F32
//
// Block layout (Q4_0_BLOCK_BYTES = 18 bytes):
//   bytes 0-1  : float16 scale d (as uint16, converted via inline F16→F32)
//   bytes 2-17 : uint8[16] packed nibbles (2 elements per byte, 32 total)
//
// Dequant element 2i  : ((packed & 0x0F) - 8) * d   low nibble
// Dequant element 2i+1: ((packed >> 4)   - 8) * d   high nibble
// ---------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void mul_mat_q4_0(
    __global const uchar *A,
    __global const float *B,
    __global       float *C,
    uint M, uint N, uint K)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    if (col >= M || row >= N) return;

    float acc = 0.0f;
    uint  nb  = K / QK4_0;

    for (uint b = 0; b < nb; ++b) {
        uint  base = (col * nb + b) * Q4_0_BLOCK_BYTES;
        // Scale: float16 at bytes 0-1, manual conversion.
        uint  h16  = (uint)(*(__global const ushort *)(A + base));
        uint  e    = (h16 >> 10u) & 0x1Fu;
        float d    = as_float(e == 0u ? 0u :
                        ((h16 & 0x8000u) << 16u) | ((e + 112u) << 23u) | ((h16 & 0x03FFu) << 13u));

        for (uint i = 0; i < (QK4_0 / 2u); ++i) {
            // Packed nibbles at bytes 2..17.
            uchar packed = A[base + 2u + i];
            float w0 = (float)((int)(packed & 0x0Fu) - 8) * d;
            float w1 = (float)((int)((packed >> 4u) & 0x0Fu) - 8) * d;
            float act0 = B[row * K + b * QK4_0 + i * 2u];
            float act1 = B[row * K + b * QK4_0 + i * 2u + 1u];
            acc += w0 * act0 + w1 * act1;
        }
    }
    C[row * M + col] = acc;
}
