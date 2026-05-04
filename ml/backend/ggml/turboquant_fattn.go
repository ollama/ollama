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
	// QJL residual sketch fields
	qjlRows       int
	qjlPacked     *Tensor // QJL sign bits (nil if no QJL)
	qjlNorm       *Tensor // QJL residual L2 norm [nKVHeads, capacity] f32 (nil if no QJL)
	qjlProjection *Tensor // [qjlRows, headDim] f32 projection matrix (nil if no QJL)
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
		qjlRows:            t.qjlRows,
		qjlPacked:          t.qjlPacked,
		qjlNorm:            t.qjlNorm,
		qjlProjection:      t.qjlProjection,
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

	var zerosT, qjlPackedT, qjlNormT, qjlProjectionT *C.struct_ggml_tensor
	if tqk.asymmetric && tqk.zeros != nil {
		zerosT = tqk.zeros.t
	}
	if tqk.qjlRows > 0 {
		if tqk.qjlPacked != nil {
			qjlPackedT = tqk.qjlPacked.t
		}
		if tqk.qjlNorm != nil {
			qjlNormT = tqk.qjlNorm.t
		}
		if tqk.qjlProjection != nil {
			qjlProjectionT = tqk.qjlProjection.t
		}
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
		qjlPackedT,
		qjlNormT,
		qjlProjectionT,
		C.int32_t(tqk.qjlRows),
		asymmetricFlag,
		outlierPackedT,
		outlierScalesT,
		outlierIndicesT,
		outlierZerosT,
		C.int32_t(tqk.outlierBits),
		C.int32_t(tqk.outlierCount),
		C.int32_t(tqk.outlierPackedBytes),
	)
	return &Tensor{b: b, t: t}
}
