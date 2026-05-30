package ggml

// #include "ggml/include/ggml.h"
import "C"

import ml "github.com/ollama/ollama/ml"

// tqTensor wraps a packed-K buffer with the metadata needed for the fused TQ
// flash attention kernel.  The Tensor field holds the encode result (a view of
// the persistent packed-K buffer).
//
// When vPacked is non-nil, the K+V fused kernel is used: V is decoded inline
// from vPacked, bypassing the separate TQ_DEQUANT op for V.
type tqTensor struct {
	*Tensor           // packed K view ([packedBytes*nKVHeads, capacity] i8; encode result)
	scales    *Tensor // K scales [nKVHeads, capacity] f32
	codebook  *Tensor // K codebook [1<<bits] f32
	bits      int
	headDim   int
	nKVHeads  int
	nCells    int
	firstCell int
	// V packed fields (nil = V is f16 from inner cache; non-nil = K+V fused)
	vPacked   *Tensor // packed V view [v_packedBytes*nKVHeads, capacity] i8
	vScales   *Tensor // V scales [nKVHeads, capacity] f32
	vCodebook *Tensor // V codebook [1<<vBits] f32
	vBits     int
	// Asymmetric primary quantization fields
	asymmetric bool
	zeros      *Tensor // K zeros [nKVHeads, capacity] f32 (nil if symmetric)
	// WHT rotation sign vector [headDim] f32 ±1 (nil if no rotation).
	// The symmetric WHT F(x)=S·H·S·x/√n is self-inverse; same vector rotates Q
	// and undoes V rotation on the attention output.
	signs        *Tensor
	vIsWHT       bool    // V is WHT-encoded; K-only fused path must undo WHT on attnOut
	// locs is the [nCells]i32 physical-slot index tensor used by indexed
	// (non-contiguous) caches. nil = contiguous fast path (cell = firstCell+c).
	locs *Tensor
	// Outlier-split fields (populated when outlierCount > 0).
	// The fused kernel decodes the dual-stream (regular + outlier) packed K
	// inline to eliminate the separate DequantK materialisation on *qa/*q presets.
	outlierPacked      *Tensor // [outlierPackedBytes*nKVHeads, capacity] i8 (nil if no outliers)
	outlierScales      *Tensor // [nKVHeads, capacity] f32
	outlierIndices     *Tensor // [outlierCount*nKVHeads, capacity] i8, head-dim positions
	outlierZeros       *Tensor // [nKVHeads, capacity] f32 (nil if !asymmetric)
	outlierCodebook    *Tensor // [1<<outlierBits] f32 (nil if no outliers)
	outlierBits        int
	outlierCount       int
	outlierPackedBytes int
}

// Permute propagates the tqTensor wrapper through the key permutation that
// ScaledDotProductAttention applies before the flash-attention dispatch.
// The packed-K layout is custom (not standard ggml strides), so we preserve
// the wrapper metadata and let the CUDA kernel ignore the permuted strides.
func (t *tqTensor) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	return &tqTensor{
		Tensor:             t.Tensor.Permute(ctx, shape...).(*Tensor),
		scales:             t.scales,
		codebook:           t.codebook,
		bits:               t.bits,
		headDim:            t.headDim,
		nKVHeads:           t.nKVHeads,
		nCells:             t.nCells,
		firstCell:          t.firstCell,
		vPacked:            t.vPacked,
		vScales:            t.vScales,
		vCodebook:          t.vCodebook,
		vBits:              t.vBits,
		asymmetric:         t.asymmetric,
		zeros:              t.zeros,
		signs:              t.signs,
		vIsWHT:             t.vIsWHT,
		locs:               t.locs,
		outlierPacked:      t.outlierPacked,
		outlierScales:      t.outlierScales,
		outlierIndices:     t.outlierIndices,
		outlierZeros:       t.outlierZeros,
		outlierCodebook:    t.outlierCodebook,
		outlierBits:        t.outlierBits,
		outlierCount:       t.outlierCount,
		outlierPackedBytes: t.outlierPackedBytes,
	}
}

// tqVTensor wraps a packed-V buffer with the metadata needed for the V-only
// fused TQ flash attention kernel (raw f16 K + packed V decoded inline).
// Implements ml.Tensor via the embedded packed-V tensor.
type tqVTensor struct {
	*Tensor           // packed V view ([v_packedBytes*nKVHeads, capacity] i8)
	vScales   *Tensor // V scales [nKVHeads, capacity] f32
	vCodebook *Tensor // V codebook [1<<vBits] f32
	vBits     int
	headDim   int
	nKVHeads  int
	nCells    int
	firstCell int
	// signs is the WHT sign vector [headDim] f32 ±1 used to undo the V
	// rotation on attnOut after the fused kernel returns.
	signs *Tensor
	// locs is the [nCells]i32 physical-slot index tensor used by indexed
	// (non-contiguous) caches. nil = contiguous fast path (cell = firstCell+c).
	locs *Tensor
}

// Permute propagates the tqVTensor through the key permutation in SDPA.
// Only the embedded packed-V tensor is permuted; all metadata fields are
// carried through unchanged.
func (t *tqVTensor) Permute(ctx ml.Context, shape ...int) ml.Tensor {
	return &tqVTensor{
		Tensor:    t.Tensor.Permute(ctx, shape...).(*Tensor),
		vScales:   t.vScales,
		vCodebook: t.vCodebook,
		vBits:     t.vBits,
		headDim:   t.headDim,
		nKVHeads:  t.nKVHeads,
		nCells:    t.nCells,
		firstCell: t.firstCell,
		signs:     t.signs,
		locs:      t.locs,
	}
}

// tqVOnlyFlashAttention creates a GGML_OP_TQ_FLASH_ATTN_EXT node for the
// V-only fused path: raw f16 K (bits=0 sentinel) + packed V decoded inline.
// query: permuted Q [D, ncols, nHeadsQ, nSeq] f32 (after Permute(0,2,1,3))
// key:   UN-permuted raw f16 K [D, nKVHeads, nCells, nSeq] f16 (rawKey before SDPA permute)
// tqv:   V-only packed wrapper (vPacked + scales + codebook + signs)
func (b *Backend) tqVOnlyFlashAttention(
	ctx ml.Context,
	query *Tensor,
	key *Tensor,
	tqv *tqVTensor,
	mask ml.Tensor,
	scale float64,
	logitSoftcap float64,
) ml.Tensor {
	var maskT *C.struct_ggml_tensor
	if mask != nil {
		maskT = mask.(*Tensor).t
	}
	var locsT *C.struct_ggml_tensor
	vFirstCell := tqv.firstCell
	if tqv.locs != nil {
		locsT = tqv.locs.t
		vFirstCell = 0 // DequantKAt output is dense starting at slot 0
	}
	// bits=0 is the sentinel the CUDA dispatch uses to identify V-only fused.
	// v_scales != NULL signals V is packed; K scales/codebook/outliers are NULL.
	t := C.ggml_tq_flash_attn_ext(
		ctx.(*Context).ctx,
		query.t,
		key.t,           // raw f16 K (src[1])
		tqv.Tensor.t,    // packed V bytes (src[2])
		maskT,
		nil,             // K scales (NULL — K is raw f16)
		nil,             // K codebook (NULL)
		C.float(scale),
		C.float(logitSoftcap),
		C.int32_t(0),   // bits=0 sentinel: K is raw f16, not TQ-packed
		C.int32_t(vFirstCell),
		tqv.vScales.t,   // v_scales (non-NULL → V is packed)
		tqv.vCodebook.t, // v_codebook
		C.int32_t(tqv.vBits),
		nil,            // zeros (NULL — K is not asymmetric)
		C.int32_t(0),   // asymmetric=0
		nil, nil, nil, nil, // outlier fields (NULL — no K outlier)
		C.int32_t(0),   // outlier_bits
		C.int32_t(0),   // outlier_count
		C.int32_t(0),   // outlier_packed_bytes
		locsT,          // locs: nil=contiguous, [nCells]i32=indexed physical slots
		nil,            // wht_signs (no WHT-Q fusion for V-only path)
	)
	return &Tensor{b: b, t: t}
}

// TQFlashAttention creates a GGML_OP_TQ_FLASH_ATTN_EXT graph node.
// query: permuted+rotated Q [D, nTokensQ, nHeadsQ, nSeq] f32
// tqk:   TQ packed-K wrapper (may also carry V packed fields for K+V fused)
// value: permuted f16 V for K-only fused, OR packed i8 V for K+V fused
func (b *Backend) tqFlashAttention(
	ctx ml.Context,
	query *Tensor,
	tqk *tqTensor,
	value *Tensor,
	mask ml.Tensor,
	scale float64,
	logitSoftcap float64,
) ml.Tensor {
	var maskT *C.struct_ggml_tensor
	if mask != nil {
		maskT = mask.(*Tensor).t
	}

	// K+V fused: pass V packed tensors to the C API.
	// K-only fused: pass NULL for v_scales (backward compat).
	var vScalesT, vCodebookT *C.struct_ggml_tensor
	vBits := C.int32_t(0)
	if tqk.vPacked != nil {
		vScalesT = tqk.vScales.t
		vCodebookT = tqk.vCodebook.t
		vBits = C.int32_t(tqk.vBits)
	}

	var zerosT *C.struct_ggml_tensor
	if tqk.asymmetric && tqk.zeros != nil {
		zerosT = tqk.zeros.t
	}

	asymmetricFlag := C.int32_t(0)
	if tqk.asymmetric {
		asymmetricFlag = 1
	}

	var outlierPackedT, outlierScalesT, outlierIndicesT, outlierZerosT *C.struct_ggml_tensor
	if tqk.outlierCount > 0 && tqk.outlierPacked != nil {
		outlierPackedT = tqk.outlierPacked.t
		outlierScalesT = tqk.outlierScales.t
		outlierIndicesT = tqk.outlierIndices.t
		if tqk.asymmetric && tqk.outlierZeros != nil {
			outlierZerosT = tqk.outlierZeros.t
		}
	}

	var locsT *C.struct_ggml_tensor
	if tqk.locs != nil {
		locsT = tqk.locs.t
	}

	t := C.ggml_tq_flash_attn_ext(
		ctx.(*Context).ctx,
		query.t,
		tqk.Tensor.t,
		value.t,
		maskT,
		tqk.scales.t,
		tqk.codebook.t,
		C.float(scale),
		C.float(logitSoftcap),
		C.int32_t(tqk.bits),
		C.int32_t(tqk.firstCell),
		vScalesT,
		vCodebookT,
		vBits,
		zerosT,
		asymmetricFlag,
		outlierPackedT,
		outlierScalesT,
		outlierIndicesT,
		outlierZerosT,
		C.int32_t(tqk.outlierBits),
		C.int32_t(tqk.outlierCount),
		C.int32_t(tqk.outlierPackedBytes),
		locsT,
		nil, // wht_signs: external WHT applied by caller (ggml.go TQApplyWHT); no internal WHT
	)
	return &Tensor{b: b, t: t}
}

