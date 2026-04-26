// SPDX-License-Identifier: MIT
// rope.cl — Rotary Position Embedding (RoPE) kernel for the Intel Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file rope.cl -device <target> -options "-cl-std=CL2.0 -O3"
//
// One entry point is compiled from this file:
//   rope — applies rotary embeddings in-place or to a separate output buffer
//
// Argument binding contract (must match ggml-level-zero.cpp exactly):
//   arg 0: x          (__global float*)       — source tensor src0->data
//   arg 1: y          (__global float*)       — destination tensor node->data
//   arg 2: pos        (__global const int*)   — token position array src1->data
//   arg 3: n_heads    (uint)                  — number of attention heads == src0->ne[2]
//   arg 4: n_dims     (uint)                  — rotary embedding dimension == op_params[0]
//   arg 5: theta_base (float)                 — frequency base == *(float*)&op_params[4]
//   arg 6: freq_scale (float)                 — frequency scale == *(float*)&op_params[5]
//
// Work-group size: (256, 1, 1).
// groupCountX = ceil(n_el / 256) where n_el = ggml_nelements(node).
// Each work-item processes one element index from the flattened output tensor.
//
// RoPE formula per element pair (i0, i0 + n_dims/2) within each head:
//   dim_pair  = i0 mod (n_dims/2)
//   freq      = freq_scale / pow(theta_base, 2*dim_pair / n_dims)
//   angle     = pos[token] * freq
//   y[i0]             = x[i0] * cos(angle) - x[i0 + n_dims/2] * sin(angle)
//   y[i0 + n_dims/2]  = x[i0] * sin(angle) + x[i0 + n_dims/2] * cos(angle)
//
// The kernel is launched over ALL elements (n_el), but only processes
// elements whose intra-head offset falls in the first half of n_dims.
// Elements in the upper half (offset >= n_dims/2) are written by the
// corresponding lower-half work-item and are skipped to avoid double-write.

// ---------------------------------------------------------------------------
// Rotary position embedding applied element-pair wise.
//
// The tensor layout is [n_tokens x n_heads x n_dims] or equivalent.
// The global element index is decoded into (token, head, dim_offset) to
// address the rotation angle correctly.
// ---------------------------------------------------------------------------
__kernel
void rope(
    __global       float       *x,
    __global       float       *y,
    __global const int         *pos,
    uint  n_heads,
    uint  n_dims,
    float theta_base,
    float freq_scale)
{
    uint gid = get_global_id(0);

    uint half_dims = n_dims / 2u;
    uint head_size = n_dims;

    // Compute the position of this work-item in the (token, head, dim) layout.
    // Total elements per (token, head) slice = n_dims.
    uint head_elem  = gid % head_size;      // element index within a single head
    uint head_idx   = (gid / head_size) % n_heads;
    uint token_idx  = gid / (head_size * n_heads);

    // Only the first half of each head's dimensions generates output pairs.
    // The second half is written by the corresponding first-half work-item.
    if (head_elem >= half_dims) return;

    // Compute rotation frequency for this dimension pair.
    float dim_pair  = (float)head_elem;
    float exponent  = 2.0f * dim_pair / (float)n_dims;
    float freq      = freq_scale / native_powr(theta_base, exponent);
    float angle     = (float)pos[token_idx] * freq;

    float cos_a = native_cos(angle);
    float sin_a = native_sin(angle);

    // Source indices for the element pair.
    uint base = (token_idx * n_heads + head_idx) * n_dims;
    uint i0   = base + head_elem;
    uint i1   = base + head_elem + half_dims;

    float x0 = x[i0];
    float x1 = x[i1];

    y[i0] = x0 * cos_a - x1 * sin_a;
    y[i1] = x0 * sin_a + x1 * cos_a;
}
