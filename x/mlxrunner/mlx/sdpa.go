package mlx

// #include "generated.h"
import "C"

import (
	"math"
	"unsafe"
)

// KVHistory carries sequence metadata alongside K/V buffers for SDPA.
// Page table and seq lens travel together — SDPA always needs both.
type KVHistory struct {
	// PageTable maps (seqIdx, position) → slot index in the K/V buffer.
	// Shape: [numSeqs, maxSeqLen], int32. Unused entries are 0.
	PageTable *Array

	// SeqLens is the history length per sequence (number of valid
	// entries in each row of PageTable).
	SeqLens []int
}

// SDPAOption configures ScaledDotProductAttention.
type SDPAOption func(*sdpaConfig)

type sdpaConfig struct {
	kvHistory    *KVHistory
	mask         *Array
	querySeqLens []int
}

// WithKVHistory provides sequence metadata for multi-sequence attention.
// SDPA uses the page table and seq lens to build an isolation mask that
// prevents cross-sequence attention and masks gaps in the buffer.
//
// querySeqLens specifies how many query tokens belong to each sequence
// (matching the order of KVHistory.SeqLens).
func WithKVHistory(kv KVHistory, querySeqLens []int) SDPAOption {
	return func(c *sdpaConfig) {
		c.kvHistory = &kv
		c.querySeqLens = querySeqLens
	}
}

// WithMask provides a model-built attention mask for custom attention
// patterns (e.g., bidirectional regions for VLMs). When used with
// WithKVHistory, masks are composed additively.
func WithMask(mask *Array) SDPAOption {
	return func(c *sdpaConfig) { c.mask = mask }
}

// ScaledDotProductAttention performs scaled dot-product attention.
//
// Without options: uses MLX's built-in causal SDPA (L>1) or no mask (L==1).
// With WithKVHistory: builds isolation+causal mask from page table.
// With WithMask: composes isolation mask + model mask additively.
func ScaledDotProductAttention(q, k, v *Array, scale float32, opts ...SDPAOption) *Array {
	var cfg sdpaConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	// Fast path: single-sequence KVHistory with no user mask.
	// With only one sequence and an identity page table, there is nothing to
	// isolate from — the isolation mask reduces to plain causal masking.
	if cfg.mask == nil {
		if cfg.kvHistory == nil || cfg.kvHistory.PageTable == nil {
			return sdpaCausal(q, k, v, scale)
		} else {
			if len(cfg.kvHistory.SeqLens) == 1 && cfg.kvHistory.SeqLens[0] == k.Dim(2) && isIdentityMapped(cfg.kvHistory.PageTable) {
				return sdpaCausal(q, k, v, scale)
			}
		}
	}

	// Build the final mask
	var finalMask *Array

	if cfg.kvHistory != nil && cfg.kvHistory.PageTable != nil {
		includeCausality := cfg.mask == nil
		isolationMask := buildIsolationMask(q, k, cfg.kvHistory, cfg.querySeqLens, includeCausality)
		if cfg.mask != nil {
			finalMask = isolationMask.Add(cfg.mask)
		} else {
			finalMask = isolationMask
		}
	} else {
		finalMask = cfg.mask
	}

	// Masks are built from float32 slices but Q/K/V may be bfloat16/float16.
	// MLX SDPA requires matching dtypes.
	if finalMask.DType() != q.DType() {
		finalMask = finalMask.AsType(q.DType())
	}

	sinks := New("")
	cMode := C.CString("")
	defer C.free(unsafe.Pointer(cMode))

	out := New("FAST_SDPA_BATCHED")
	C.mlx_fast_scaled_dot_product_attention(
		&out.ctx, q.ctx, k.ctx, v.ctx,
		C.float(scale), cMode, finalMask.ctx, sinks.ctx, DefaultStream().ctx,
	)
	return out
}

// sdpaCausal uses MLX's built-in causal mode.
func sdpaCausal(q, k, v *Array, scale float32) *Array {
	mask := New("")
	sinks := New("")
	cMode := C.CString("causal")
	defer C.free(unsafe.Pointer(cMode))

	out := New("FAST_SDPA")
	C.mlx_fast_scaled_dot_product_attention(
		&out.ctx, q.ctx, k.ctx, v.ctx,
		C.float(scale), cMode, mask.ctx, sinks.ctx, DefaultStream().ctx,
	)
	return out
}

// buildIsolationMask creates a mask from KVHistory metadata.
// Mask width = k.Dim(2) (full buffer size, not just populated slots).
func buildIsolationMask(q, k *Array, kv *KVHistory, querySeqLens []int, includeCausality bool) *Array {
	totalQ := q.Dim(2)
	totalSlots := k.Dim(2)

	pageTableData := kv.PageTable.Ints()
	numSeqs := len(kv.SeqLens)
	maxSeqLen := 0
	if numSeqs > 0 {
		maxSeqLen = kv.PageTable.Dim(1)
	}

	negInf := float32(math.Inf(-1))
	mask := make([]float32, totalQ*totalSlots)
	for i := range mask {
		mask[i] = negInf
	}

	qi := 0
	for seqIdx := range numSeqs {
		qCount := querySeqLens[seqIdx]
		historyLen := kv.SeqLens[seqIdx]

		// When queries exceed history (e.g., prefill > sliding window),
		// only the last historyLen queries have corresponding K/V.
		// Earlier queries' K/V were dropped — they see nothing.
		droppedQueries := max(qCount-historyLen, 0)

		// priorHistory is the number of KV entries already in the cache
		// before this batch of queries. Clamped to 0 for the sliding window
		// case where queries exceed history (droppedQueries > 0).
		priorHistory := max(historyLen-qCount, 0)

		for qLocal := range qCount {
			allowed := 0
			if qLocal >= droppedQueries {
				allowed = historyLen
				if includeCausality {
					// Query at qLocal sees all prior history plus causal
					// entries within the new queries up to its own position.
					allowed = min(priorHistory+qLocal-droppedQueries+1, historyLen)
				}
			}
			for j := range allowed {
				slot := pageTableData[seqIdx*maxSeqLen+j]
				mask[qi*totalSlots+slot] = 0
			}
			qi++
		}
	}

	return FromValues(mask, 1, 1, totalQ, totalSlots)
}

// isIdentityMapped returns true when the page table maps every
// position i to slot i with no gaps.
func isIdentityMapped(pt *Array) bool {
	ptData := pt.Ints()
	maxSeqLen := pt.Dim(1)
	for i := range maxSeqLen {
		if ptData[i] != i {
			return false
		}
	}
	return true
}
