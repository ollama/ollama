#pragma once

// Public C ABI launchers for the V-only TQ fused flash-attention kernel.
// One entry point per head dimension (D ∈ {64, 128, 256, 512}). Callable from
// cgo via Go bindings for kernel-level unit testing and for the ggml
// op dispatch path once wired.
//
// All launchers share the same parameter shape. logit_softcap == 0.0f
// means "no softcap" (selects USE_LOGIT_SOFTCAP=false template);
// non-zero (e.g. 50.0f for Gemma 2/3) applies logit_softcap * tanh(sum).
//
// Limitations (shared across all D):
//   * No outliers / asymmetric primary on V
//   * Contiguous cell range (no indexed locs)
//   * No ALiBi / softmax sink
//
// Shapes (all dims in elements):
//   Q      : [D, ncols, nHeadsQ, nSeq] f32, row-major (D contiguous)
//   K      : [D, nCells, nKVHeads, nSeq] f16
//   V_pack : [packedBytes * nKVHeads, capacity] i8, cell-major
//            packedBytes = round_up((D * v_bits + 7) / 8, 4)
//   v_scl  : [nKVHeads, capacity] f32
//   v_cb   : [1<<v_bits] f32
//   mask   : [nCells, ncols] f16 or NULL
//   dst    : [D, ncols, nHeadsQ, nSeq] f32 (written row-major same as Q)
//
// Output is in the WHT-rotated V coordinate system; caller must apply
// WHTUndo before returning the tensor to the stock FA output path.

#include "common.cuh"

extern "C" {

void ggml_cuda_tq_fattn_vec_vonly_d64(
    cudaStream_t stream,
    const float * Q,             int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    const __half * K,            int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    const uint8_t * V_packed,    int32_t v_packedBytes,
    const __half * mask,         int32_t mask_ne0, int32_t mask_nb1,
    float * dst,
    float * partial_buf,
    const float * v_scales,
    const float * v_codebook,
    const int32_t * locs,
    int v_bits,
    float scale,
    float logit_softcap,
    int firstCell,
    int nCells,
    int nSplits,
    int nHeadsQ,
    int nKVHeads,
    int nSeq,
    int ncols);

void ggml_cuda_tq_fattn_vec_vonly_d128(
    cudaStream_t stream,
    const float * Q,             int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    const __half * K,            int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    const uint8_t * V_packed,    int32_t v_packedBytes,
    const __half * mask,         int32_t mask_ne0, int32_t mask_nb1,
    float * dst,
    float * partial_buf,
    const float * v_scales,
    const float * v_codebook,
    const int32_t * locs,
    int v_bits,
    float scale,
    float logit_softcap,
    int firstCell,
    int nCells,
    int nSplits,
    int nHeadsQ,
    int nKVHeads,
    int nSeq,
    int ncols);

void ggml_cuda_tq_fattn_vec_vonly_d256(
    cudaStream_t stream,
    const float * Q,             int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    const __half * K,            int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    const uint8_t * V_packed,    int32_t v_packedBytes,
    const __half * mask,         int32_t mask_ne0, int32_t mask_nb1,
    float * dst,
    float * partial_buf,
    const float * v_scales,
    const float * v_codebook,
    const int32_t * locs,
    int v_bits,
    float scale,
    float logit_softcap,
    int firstCell,
    int nCells,
    int nSplits,
    int nHeadsQ,
    int nKVHeads,
    int nSeq,
    int ncols);

void ggml_cuda_tq_fattn_vec_vonly_d512(
    cudaStream_t stream,
    const float * Q,             int32_t nb_q01, int32_t nb_q02, int64_t nb_q03,
    const __half * K,            int32_t nb_k11, int32_t nb_k12, int64_t nb_k13,
    const uint8_t * V_packed,    int32_t v_packedBytes,
    const __half * mask,         int32_t mask_ne0, int32_t mask_nb1,
    float * dst,
    float * partial_buf,
    const float * v_scales,
    const float * v_codebook,
    const int32_t * locs,
    int v_bits,
    float scale,
    float logit_softcap,
    int firstCell,
    int nCells,
    int nSplits,
    int nHeadsQ,
    int nKVHeads,
    int nSeq,
    int ncols);

}  // extern "C"
