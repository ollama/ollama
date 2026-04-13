package kvcache

import (
	"log/slog"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

// TurboQuantWrapper applies TurboQuant (arxiv 2504.19874, ICLR 2026) to the KV cache.
//
// # Algorithm (TurboQuant_mse — MSE-optimized, Algorithm 1 from paper)
//
//  1. Normalize x to unit sphere, store ||x||_2
//  2. Rotate: y = Pi * x_normalized (Pi = randomized Hadamard via FWHT)
//  3. Scalar quantize each coordinate of y with Lloyd-Max codebook (b bits)
//  4. Dequant: y_hat[i] = centroid[idx[i]], x_hat = Pi^T * y_hat * ||x||
//
// All bits are used for Lloyd-Max (no QJL residual). Community benchmarks
// (llama.cpp #20969) show QJL adds no benefit for KV cache inference.
//
// Both keys and values go through the same pipeline:
//
//	FWHT rotation → L2 norm preservation → Lloyd-Max quantization
//
// Single Causal cache (F32):
//
//	key slot:   packed Lloyd-Max key indices + L2 norm per head
//	value slot: packed Lloyd-Max value indices + L2 norm per head
//
// On Put(key, value):
//
//	Keys:   norm → normalize → FWHT → LloydMaxQuantize → concat(packed, norm) → cache key slot
//	Values: norm → normalize → FWHT → LloydMaxQuantize → concat(packed, norm) → cache value slot
//
// On Get():
//
//	Keys:   split(packed, norm) → LloydMaxDequantize → inverse FWHT → rescale by norm
//	Values: split(packed, norm) → LloydMaxDequantize → inverse FWHT → rescale by norm
type TurboQuantWrapper struct {
	cache *Causal // single cache: key=compressed_key, value=compressed_value

	bitWidth int
	mseBits  int // quantization bits for Lloyd-Max (= bitWidth when QJL is not used)
	backend  ml.Backend
	layer    int

	// Seed split into hi/lo 32-bit halves for the FWHT GGML op
	keySeedHi uint32
	keySeedLo uint32
	valSeedHi uint32
	valSeedLo uint32

	// Track which head dims we've logged initialization for
	loggedDims map[int]bool
}

var _ CheckpointCache = (*TurboQuantWrapper)(nil)

func noopShift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return key, nil
}

func NewTurboQuantWrapper(tqDType ml.DType) *TurboQuantWrapper {
	bitWidth := turboquant.BitWidthTQ4
	switch tqDType {
	case ml.DTypeTQ2:
		bitWidth = turboquant.BitWidthTQ2
	case ml.DTypeTQ3:
		bitWidth = turboquant.BitWidthTQ3
	}

	keySeed := turboquant.TurboQuantSeed
	valSeed := turboquant.TurboQuantValueSeed
	return &TurboQuantWrapper{
		cache:      NewCausalCache(noopShift),
		bitWidth:   bitWidth,
		mseBits:    bitWidth,
		keySeedHi:  uint32(keySeed >> 32),
		keySeedLo:  uint32(keySeed & 0xFFFFFFFF),
		valSeedHi:  uint32(valSeed >> 32),
		valSeedLo:  uint32(valSeed & 0xFFFFFFFF),
		loggedDims: make(map[int]bool),
	}
}

func (w *TurboQuantWrapper) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	w.backend = backend
	// Cache stores packed I32 bit patterns as F32 (for ggml_set_rows compatibility).
	w.cache.Init(backend, ml.DTypeF32, maxSequences, capacity, maxBatch)
}

func (w *TurboQuantWrapper) Close() {
	w.cache.Close()
}

func (w *TurboQuantWrapper) SetLayer(layer int) {
	w.layer = layer
	w.cache.SetLayer(layer)
}

func (w *TurboQuantWrapper) SetConfig(config ml.CacheConfig) {
	// Packed indices — PermutedV doesn't apply.
	plainConfig := config
	plainConfig.PermutedV = false
	w.cache.SetConfig(plainConfig)
}

func (w *TurboQuantWrapper) CopyPrefix(src, dst int, l int32) {
	w.cache.CopyPrefix(src, dst, l)
}

func (w *TurboQuantWrapper) CanResume(seq int, pos int32) bool {
	return w.cache.CanResume(seq, pos)
}

func (w *TurboQuantWrapper) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	return w.cache.StartForward(ctx, batch, reserve)
}

func (w *TurboQuantWrapper) Remove(seq int, beginIndex, endIndex int32) error {
	return w.cache.Remove(seq, beginIndex, endIndex)
}

func (w *TurboQuantWrapper) PrepareRestore(seq int, targetPos int32) (int32, bool) {
	if w.cache.CanResume(seq, targetPos) {
		return targetPos, true
	}
	return 0, false
}

// compressTensor applies the TurboQuant pipeline using a single fused GGML op.
// TQCompress combines L2 norm + normalize + FWHT + LloydMaxQ + concat norm in one kernel,
// replacing 8 separate graph ops (Sqr, SumRows, Sqrt, Clamp, Div, FWHT, LloydMaxQ, Concat)
// and eliminating ~500 kernel launches per generated token.
func (w *TurboQuantWrapper) compressTensor(ctx ml.Context, t ml.Tensor, seedHi, seedLo uint32) ml.Tensor {
	headDim := t.Dim(0)
	return t.TQCompress(ctx, w.mseBits, headDim, seedHi, seedLo)
}

// decompressTensor reverses the TurboQuant pipeline using a single fused GGML op.
// TQDecompress combines LloydMaxDQ + inverse FWHT + norm rescale in one kernel,
// eliminating ALL intermediate tensors from the GGML graph. This is critical because
// Q4_0's flash attention reads compressed data on-the-fly with zero graph overhead,
// while separate DQ→FWHT→Mul ops create ~1 GiB of intermediates at 128k context.
func (w *TurboQuantWrapper) decompressTensor(ctx ml.Context, withNorm ml.Tensor, seedHi, seedLo uint32) ml.Tensor {
	dim0 := withNorm.Dim(0)
	packedDim := dim0 - 1
	origDim := packedDim * 32 / w.mseBits

	return withNorm.TQDecompress(ctx, w.mseBits, origDim, seedHi, seedLo)
}

func (w *TurboQuantWrapper) Put(ctx ml.Context, key, value ml.Tensor) {
	headDim := key.Dim(0)

	// Fall back for non-power-of-2 head dims (FWHT requires power of 2)
	if !isPowerOf2(headDim) {
		w.cache.Put(ctx, key, value)
		return
	}

	keyCompressed := w.compressTensor(ctx, key, w.keySeedHi, w.keySeedLo)
	valCompressed := w.compressTensor(ctx, value, w.valSeedHi, w.valSeedLo)
	w.cache.Put(ctx, keyCompressed, valCompressed)

	w.logInit(headDim)
}

func (w *TurboQuantWrapper) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	keyWithNorm, valWithNorm, mask := w.cache.Get(ctx)

	dim0 := keyWithNorm.Dim(0)
	if dim0 == 0 {
		return keyWithNorm, valWithNorm, mask
	}

	key := w.decompressTensor(ctx, keyWithNorm, w.keySeedHi, w.keySeedLo)
	value := w.decompressTensor(ctx, valWithNorm, w.valSeedHi, w.valSeedLo)

	return key, value, mask
}

func (w *TurboQuantWrapper) logInit(headDim int) {
	if !w.loggedDims[headDim] {
		w.loggedDims[headDim] = true
		packedDim := headDim * w.mseBits / 32
		storedDim := packedDim + 1 // packed indices + norm
		bytesPerHead := storedDim * 4 // F32 = 4 bytes each
		slog.Info("TurboQuant cache initialized",
			"head_dim", headDim,
			"bit_width", w.bitWidth,
			"mse_bits", w.mseBits,
			"stored_dim", storedDim,
			"bytes_per_head_key+val", bytesPerHead*2,
			"compression_ratio", float64(headDim*2)/float64(storedDim*2))
	}
}

func isPowerOf2(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}
