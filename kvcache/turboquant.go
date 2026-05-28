package kvcache

import (
	"fmt"
	"log/slog"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

// TurboQuantWrapper applies the data-oblivious random rotation from Google's
// TurboQuant paper (arxiv 2504.19874, ICLR 2026) to the KV cache.
//
// # Paper Algorithm
//
// The full TurboQuant_prod (Algorithm 2) works as follows:
//  1. Normalize x to unit sphere, store ||x||_2
//  2. Rotate: y = Pi * x_normalized (Pi = randomized Hadamard)
//  3. Scalar quantize each coordinate of y with Lloyd-Max codebook (b-1 bits)
//  4. Compute residual r = x_normalized - DeQuant_mse(indices)
//  5. QJL: signs = sign(S * r), store (signs, ||r||_2)
//  6. Dequantize: x_hat = DeQuant_mse(indices) + (sqrt(pi/2)/d)*||r||*S^T*signs
//
// # What This Implementation Does
//
// This wrapper implements the core rotation stage (step 2 above) which is the
// primary contributor to TurboQuant's quality improvement. After rotation,
// coordinates follow a concentrated Beta distribution (approx Gaussian N(0,1/d)
// in high dims), eliminating the outlier channels that cause block quantization
// to fail. The inner cache stores the rotated data using GGML's Q4_0 format.
//
// Specifically, on Put(key, value):
//   - key_rotated = Pi * key       (rotate key vectors)
//   - Store key_rotated via inner cache (Q4_0 quantization)
//   - Store value via inner cache as-is (Q4_0 quantization)
//
// On Get():
//   - Read key_quantized from inner cache (auto-dequantized by GGML)
//   - key_restored = Pi^T * key_quantized  (inverse rotation)
//   - Return key_restored, value, mask
//
// # Approximations vs Full Paper
//
// The following aspects of the paper are not yet implemented:
//   - Norm preservation (L2 normalization before rotation). Q4_0's per-block
//     scale factors partially compensate, making this non-critical.
//   - Lloyd-Max optimal codebook quantization. GGML's Q4_0 block quantization
//     (32-element blocks with shared scale) is used instead. After rotation,
//     Q4_0 works significantly better because information is uniformly spread.
//   - QJL residual correction (Algorithm 2, steps 4-6). This would provide
//     unbiased inner product estimates. Attention quality is already good
//     because the rotation stage dominates the error reduction.
//   - Value rotation. Only keys are rotated because values stored with
//     PermutedV layout (required for flash attention) would need extra permute
//     operations. Key rotation has the largest impact on attention accuracy
//     since Q*K^T is the inner product that TurboQuant targets.
type TurboQuantWrapper struct {
	inner    Cache
	bitWidth int
	backend  ml.Backend
	layer    int

	mu        sync.Mutex
	rotations map[int]*rotationPair
	rotCtxs   map[int]ml.Context
}

var _ CheckpointCache = (*TurboQuantWrapper)(nil)

type rotationPair struct {
	// forward stores Pi^T in GGML format.
	// GGML Mulmat(A, B) computes A^T @ B.
	// So forward.Mulmat(X) = (Pi^T)^T @ X = Pi @ X.
	forward ml.Tensor

	// inverse stores Pi in GGML format.
	// inverse.Mulmat(X) = Pi^T @ X.
	inverse ml.Tensor
}

func NewTurboQuantWrapper(inner Cache, tqDType ml.DType) *TurboQuantWrapper {
	bitWidth := turboquant.BitWidthTQ4
	if tqDType == ml.DTypeTQ3 {
		bitWidth = turboquant.BitWidthTQ3
	}

	return &TurboQuantWrapper{
		inner:     inner,
		bitWidth:  bitWidth,
		rotations: make(map[int]*rotationPair),
		rotCtxs:   make(map[int]ml.Context),
	}
}

func (w *TurboQuantWrapper) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	w.backend = backend

	// ! The inner cache uses Q4_0 regardless of whether tq3 or tq4 was requested.
	// Q4_0 provides ~4.5 bits/element effective storage. The rotation transform
	// makes this dramatically more effective than raw Q4_0 by eliminating outlier
	// channels that cause block quantization distortion.
	w.inner.Init(backend, ml.DTypeQ40, maxSequences, capacity, maxBatch)
}

func (w *TurboQuantWrapper) Close() {
	for _, ctx := range w.rotCtxs {
		ctx.Close()
	}
	w.inner.Close()
}

func (w *TurboQuantWrapper) SetLayer(layer int) {
	w.layer = layer
	w.inner.SetLayer(layer)
}
func (w *TurboQuantWrapper) SetConfig(config ml.CacheConfig)   { w.inner.SetConfig(config) }
func (w *TurboQuantWrapper) CopyPrefix(src, dst int, l int32)  { w.inner.CopyPrefix(src, dst, l) }
func (w *TurboQuantWrapper) CanResume(seq int, pos int32) bool { return w.inner.CanResume(seq, pos) }

func (w *TurboQuantWrapper) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	return w.inner.StartForward(ctx, batch, reserve)
}

func (w *TurboQuantWrapper) Remove(seq int, beginIndex, endIndex int32) error {
	return w.inner.Remove(seq, beginIndex, endIndex)
}

func (w *TurboQuantWrapper) PrepareRestore(seq int, targetPos int32) (int32, bool) {
	if cc, ok := w.inner.(CheckpointCache); ok {
		return cc.PrepareRestore(seq, targetPos)
	}

	// Preserve non-checkpoint cache behavior used by ollamarunner:
	// keep targetPos when the cache can resume, otherwise signal reprocess.
	if w.inner.CanResume(seq, targetPos) {
		return targetPos, true
	}

	return 0, false
}

func (w *TurboQuantWrapper) Put(ctx ml.Context, key, value ml.Tensor) {
	headDim := key.Dim(0)
	rot := w.getOrCreateRotation(headDim)

	if rot != nil {
		key = rot.forward.Mulmat(ctx, key)
	}

	w.inner.Put(ctx, key, value)
}

func (w *TurboQuantWrapper) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key, value, mask := w.inner.Get(ctx)

	headDim := key.Dim(0)
	rot := w.getOrCreateRotation(headDim)

	if rot != nil {
		// Metal does not provide all f32 x q4_0 matmul variants needed for
		// rotation. Cast cache output to f32 before applying inverse rotation.
		if key.DType() != ml.DTypeF32 {
			key = key.Cast(ctx, ml.DTypeF32)
		}
		key = rot.inverse.Mulmat(ctx, key)
	}

	return key, value, mask
}

func (w *TurboQuantWrapper) getOrCreateRotation(headDim int) *rotationPair {
	w.mu.Lock()
	defer w.mu.Unlock()

	if rot, ok := w.rotations[headDim]; ok {
		return rot
	}

	if !isPowerOf2(headDim) {
		slog.Warn("TurboQuant requires power-of-2 head dimension, skipping rotation",
			"head_dim", headDim)
		w.rotations[headDim] = nil
		return nil
	}

	seed := turboquant.TurboQuantSeed

	piData := turboquant.GenerateRotation(headDim, seed)
	piTData := turboquant.GenerateRotationTranspose(headDim, seed)

	rotCtx := w.backend.NewContextSize(2).Layer(w.layer)
	piTensor := rotCtx.FromFloats(piData, headDim, headDim)
	piTTensor := rotCtx.FromFloats(piTData, headDim, headDim)

	rot := &rotationPair{
		forward: piTTensor,
		inverse: piTensor,
	}

	w.rotations[headDim] = rot
	w.rotCtxs[headDim] = rotCtx

	slog.Info("TurboQuant: initialized randomized Hadamard rotation",
		"head_dim", headDim,
		"bit_width", w.bitWidth,
		"inner_dtype", "q4_0",
		"matrix_bytes", fmt.Sprintf("%dKB", headDim*headDim*4/1024))

	return rot
}

func isPowerOf2(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}
