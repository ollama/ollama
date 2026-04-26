// SPDX-License-Identifier: MIT
// attention.cl — Tiled scaled dot-product attention kernel stub for the
//                Intel Level Zero backend.
//
// Compiled AOT to SPIR-V by ocloc:
//   ocloc compile -file attention.cl -device <target> -options "-cl-std=CL2.0 -O3"
//
// One entry point is compiled from this file:
//   attention_tiled — scaled dot-product attention with causal mask and online softmax
//
// STATUS: Stub — SPIR-V must compile cleanly but this kernel is NOT currently
// dispatched by ggml-level-zero.cpp.  The dispatch switch has no case for
// GGML_OP_FLASH_ATTN_EXT that references this kernel.  The entry point is
// registered in the module cache at startup so the SPIR-V blob is verified,
// but no zeCommandListAppendLaunchKernel call targets "attention_tiled".
//
// Argument binding contract (matches dispatch-contract specification):
//   arg 0: Q       (__global const float*) — query matrix
//   arg 1: K       (__global const float*) — key matrix
//   arg 2: V       (__global const float*) — value matrix
//   arg 3: out     (__global float*)       — output matrix
//   arg 4: seq_len (uint)                  — sequence length (number of queries)
//   arg 5: head_dim (uint)                 — per-head embedding dimension
//   arg 6: scale   (float)                 — QK dot-product scale = 1/sqrt(head_dim)
//
// Work-item mapping: one work-item per query position (q_idx = get_global_id(0)).
// Boundary guard: if (q_idx >= seq_len) return.
//
// Fixed-size local arrays use preprocessor constants to satisfy the OpenCL C
// 1.2/2.0 requirement that __local array sizes are compile-time constants.

#define TILE     16
#define HEAD_DIM 64

// ---------------------------------------------------------------------------
// Tiled scaled dot-product attention with online (Flash-Attention style) softmax.
//
// Each work-item owns one query row (q_idx).  It streams over all key/value
// positions, maintains running max and sum for numerical stability, and
// accumulates the weighted value sum directly — no intermediate attention
// matrix is materialised.
//
// Causal masking: key positions kv_idx > q_idx contribute zero probability.
// ---------------------------------------------------------------------------
__kernel
void attention_tiled(
    __global const float *Q,
    __global const float *K,
    __global const float *V,
    __global       float *out,
    uint   seq_len,
    uint   head_dim,
    float  scale)
{
    uint q_idx = get_global_id(0);
    if (q_idx >= seq_len) return;

    // Running accumulators for online softmax (Flash-Attention Algorithm 1).
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Output accumulator vector.  Fixed size HEAD_DIM must cover the actual
    // head_dim; caller must ensure head_dim <= HEAD_DIM.
    float acc[HEAD_DIM];
    for (uint d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;

    uint q_base = q_idx * head_dim;

    for (uint kv_idx = 0; kv_idx <= q_idx; ++kv_idx) {
        uint kv_base = kv_idx * head_dim;

        // Compute dot product Q[q_idx] . K[kv_idx] scaled.
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += Q[q_base + d] * K[kv_base + d];
        }
        float score = dot * scale;

        // Online softmax update.
        float new_max = (score > row_max) ? score : row_max;
        float exp_old = native_exp(row_max - new_max);
        float exp_new = native_exp(score - new_max);

        float new_sum = row_sum * exp_old + exp_new;

        // Rescale accumulated output and add new value contribution.
        float rescale = (new_sum > 0.0f) ? (row_sum * exp_old / new_sum) : 0.0f;
        float weight  = (new_sum > 0.0f) ? (exp_new / new_sum) : 0.0f;

        for (uint d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * rescale + V[kv_base + d] * weight;
        }

        row_max = new_max;
        row_sum = new_sum;
    }

    // Write result.
    uint out_base = q_idx * head_dim;
    for (uint d = 0; d < head_dim; ++d) {
        out[out_base + d] = acc[d];
    }
}
