package ggml

import (
	"log/slog"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// ggmlTQCompressedK implements ml.TQCompressedKManager using ggml tensors and
// GGML_OP_TQ_ENCODE / GGML_OP_TQ_DEQUANT graph ops. All buffers live in GPU
// memory; no CPU round-trips occur during the forward pass.
type ggmlTQCompressedK struct {
	backend      *Backend
	headDim      int
	numKVHeads   int
	bits         int

	// Outlier-split config (post-rotation top-K channel split). When
	// outlierCount > 0, EnsureLayer allocates additional tensors for an
	// outlier sub-block encoded at outlierBits, and the encode/dequant
	// kernels follow the outlier-aware path. When 0, uses pure uniform
	// per-channel Lloyd-Max at `bits`.
	outlierBits  int
	outlierCount int

	mu sync.Mutex

	// Per-layer ggml tensors, allocated lazily via EnsureLayer.
	layerCtxs     map[int]ml.Context
	packedTensors map[int]*Tensor // regular sub-block: [regularPackedBytes*numKVHeads, capacity] i8
	scalesTensors map[int]*Tensor // regular scales: [numKVHeads, capacity] f32

	// Outlier sub-block per-layer tensors (populated only when outlierCount > 0).
	outlierPackedTensors  map[int]*Tensor // [outlierPackedBytes*numKVHeads, capacity] i8
	outlierScalesTensors  map[int]*Tensor // [numKVHeads, capacity] f32
	outlierIndicesTensors map[int]*Tensor // [outlierCount*numKVHeads, capacity] i8 (channel idx)

	// Rotation matrix R^T, shared across layers: [headDim, headDim] f32.
	rotCtx    ml.Context
	rotTensor *Tensor // stores R^T row-major (used for K encode and Q rotate)

	// Rotation matrix R (transpose of R^T), for undoing V rotation in SDPA.
	// mul_mat(rotInverseTensor, x) = R @ x (recovers original from R^T @ x).
	rotInverseTensor *Tensor // stores R row-major

	// Codebook and boundaries tensors, shared across layers.
	sharedCtx        ml.Context
	codebookTensor   *Tensor // regular: [1<<bits] f32
	boundariesTensor *Tensor // regular: [(1<<bits)-1] f32

	// Outlier codebook and boundaries (populated only when outlierCount > 0).
	outlierCodebookTensor   *Tensor // [1<<outlierBits] f32
	outlierBoundariesTensor *Tensor // [(1<<outlierBits)-1] f32

	vBits int

	// Per-layer V tensors, allocated lazily via EnsureVLayer.
	vLayerCtxs     map[int]ml.Context
	vPackedTensors map[int]*Tensor // [packedBytes*numKVHeads, capacity] i8
	vScalesTensors map[int]*Tensor // [numKVHeads, capacity] f32

	// V codebook and boundaries (same bit width as K for tq2/tq3).
	vCodebookTensor   *Tensor // [1<<vBits] f32
	vBoundariesTensor *Tensor // [(1<<vBits)-1] f32
}

// hasOutliers reports whether outlier-split is active for this manager.
func (m *ggmlTQCompressedK) hasOutliers() bool {
	return m.outlierCount > 0 && m.outlierBits > 0 && m.outlierCount < m.headDim
}

// regularChannelCount is the number of non-outlier channels per head.
func (m *ggmlTQCompressedK) regularChannelCount() int {
	if m.hasOutliers() {
		return m.headDim - m.outlierCount
	}
	return m.headDim
}

// regularPackedBytes is the padded per-head byte count for the regular
// sub-block. The encode kernel uses atomicOr on 4-byte words to pack bits;
// for that to stay aligned, each head's region must start on a 4-byte
// boundary. Round the raw bit-count up to the next multiple of 4 so the
// per-head stride is naturally aligned. The padding bytes are never read
// during decode, and are zeroed by the encode kernel's init loop.
func (m *ggmlTQCompressedK) regularPackedBytes() int {
	raw := (m.regularChannelCount()*m.bits + 7) / 8
	return (raw + 3) &^ 3
}

// outlierPackedBytes is the padded per-head byte count for the outlier
// sub-block. Same 4-byte alignment as regularPackedBytes() for the same
// reason: atomicOr-on-word in the encode kernel.
func (m *ggmlTQCompressedK) outlierPackedBytes() int {
	if !m.hasOutliers() {
		return 0
	}
	raw := (m.outlierCount*m.outlierBits + 7) / 8
	return (raw + 3) &^ 3
}

func (b *Backend) NewTQCompressedKManager(headDim, numKVHeads, bits int, rotationSeed uint64, vBits, outlierBits, outlierCount int) ml.TQCompressedKManager {
	// TurboQuant ops are CUDA-only and require compute capability 6.0+ (Pascal
	// or newer) because the kernels use __shfl_sync for codebook lookup. Scan
	// the scheduler buffer types, pick the first TQ-capable GPU, and warn
	// clearly if there's no suitable device or if a mixed-generation rig forces
	// us to skip older cards.
	scan := b.scanTQDevices()
	if !scan.selectedOK {
		if len(scan.Skipped) > 0 {
			slog.Warn("turboquant: no GPU with compute capability 6.0+ available; "+
				"falling back to f16 KV cache. TurboQuant requires Pascal or newer.",
				"skipped_gpus", scan.Skipped)
		} else {
			slog.Warn("turboquant: no GPU backend available, falling back to f16 KV cache")
		}
		return nil
	}
	if len(scan.Skipped) > 0 {
		slog.Warn("turboquant: skipping GPU(s) with compute capability < 6.0; "+
			"TQ tensors will be placed on the first Pascal+ device. "+
			"To silence this warning, hide older cards with CUDA_VISIBLE_DEVICES.",
			"selected", scan.SelectedName+" (cc "+scan.SelectedCC+")",
			"skipped", scan.Skipped)
	}
	if len(scan.Accepted) > 1 {
		slog.Warn("turboquant: multi-GPU detected; TQ compressed buffers live on the "+
			"primary GPU only. Layers scheduled to other GPUs will incur per-step "+
			"cross-GPU transfers. On SWA models like gemma3/gemma4 this is per "+
			"TQ-wrapped global sub-cache — the SWA sub-cache stays on its native "+
			"GPU and is unaffected. To avoid: set num_gpu so the model fits on one "+
			"GPU, or use the tq2k/tq3k (K-only) presets which let V stay on its "+
			"native GPU.",
			"selected", scan.SelectedName+" (cc "+scan.SelectedCC+")",
			"tq_capable_gpus", scan.Accepted)
	}

	// Codebook and boundaries (same for all layers). Use headDim for the
	// codebook dim parameter regardless of whether outlier split is active:
	// the CPU path does the same (scalarCodebook(dim=headDim, bits)), and
	// after per-sub-block RMS normalization the input distribution is
	// approximately unit-variance Gaussian either way. Using sub-block dims
	// here produced slightly different boundaries that caused observable
	// quality regressions on multi-head configurations.
	codebook := turboquant.ExportCodebook(headDim, bits)
	boundaries := turboquant.ExportBoundaries(headDim, bits)

	// All shared tensors (codebook, rotation) must be GPU-resident: TQ ops are
	// CUDA-only. Using newTQContext() ensures GPU buffer type is used regardless
	// of which model layers are on CPU vs GPU.
	// Size shared context for up to 6 tensors (codebook+bounds ×2 regular, ×2
	// outlier, ×2 V); newTQContext takes a hint count.
	sharedCtx := b.newTQContext(8)
	codebookT := sharedCtx.FromFloats(codebook, len(codebook)).(*Tensor)
	boundariesT := sharedCtx.FromFloats(boundaries, len(boundaries)).(*Tensor)

	// Outlier codebook/boundaries at a different bit width. Use headDim as
	// the codebook dim (same reason as regular codebook above).
	var outlierCodebookT, outlierBoundariesT *Tensor
	if outlierCount > 0 && outlierBits > 0 && outlierCount < headDim {
		oCodebook := turboquant.ExportCodebook(headDim, outlierBits)
		oBoundaries := turboquant.ExportBoundaries(headDim, outlierBits)
		outlierCodebookT = sharedCtx.FromFloats(oCodebook, len(oCodebook)).(*Tensor)
		outlierBoundariesT = sharedCtx.FromFloats(oBoundaries, len(oBoundaries)).(*Tensor)
	}

	// V codebook and boundaries
	vCodebook := turboquant.ExportCodebook(headDim, vBits)
	vBoundaries := turboquant.ExportBoundaries(headDim, vBits)
	vCodebookT := sharedCtx.FromFloats(vCodebook, len(vCodebook)).(*Tensor)
	vBoundariesT := sharedCtx.FromFloats(vBoundaries, len(vBoundaries)).(*Tensor)

	// Rotation matrix R^T: rotData[i*headDim+j] = R[j][i]. Built via
	// Householder QR on a random Gaussian matrix per TurboQuant paper (arXiv
	// 2504.19874) Algorithm 1.
	rot := turboquant.BuildRotation(headDim, rotationSeed)
	rotData := make([]float32, headDim*headDim)
	for i := 0; i < headDim; i++ {
		for j := 0; j < headDim; j++ {
			rotData[i*headDim+j] = rot.Matrix[j*headDim+i]
		}
	}
	rotInverseData := make([]float32, headDim*headDim)
	copy(rotInverseData, rot.Matrix)
	rotCtx := b.newTQContext(2)
	rotTensor := rotCtx.FromFloats(rotData, headDim, headDim).(*Tensor)
	rotInverseTensor := rotCtx.FromFloats(rotInverseData, headDim, headDim).(*Tensor)

	m := &ggmlTQCompressedK{
		backend:                 b,
		headDim:                 headDim,
		numKVHeads:              numKVHeads,
		bits:                    bits,
		outlierBits:             outlierBits,
		outlierCount:            outlierCount,
		layerCtxs:               make(map[int]ml.Context),
		packedTensors:           make(map[int]*Tensor),
		scalesTensors:           make(map[int]*Tensor),
		outlierPackedTensors:    make(map[int]*Tensor),
		outlierScalesTensors:    make(map[int]*Tensor),
		outlierIndicesTensors:   make(map[int]*Tensor),
		rotCtx:                  rotCtx,
		rotTensor:               rotTensor,
		rotInverseTensor:        rotInverseTensor,
		sharedCtx:               sharedCtx,
		codebookTensor:          codebookT,
		boundariesTensor:        boundariesT,
		outlierCodebookTensor:   outlierCodebookT,
		outlierBoundariesTensor: outlierBoundariesT,
		vBits:                   vBits,
		vLayerCtxs:              make(map[int]ml.Context),
		vPackedTensors:          make(map[int]*Tensor),
		vScalesTensors:          make(map[int]*Tensor),
		vCodebookTensor:         vCodebookT,
		vBoundariesTensor:       vBoundariesT,
	}
	if m.hasOutliers() {
		slog.Info("turboquant: outlier split enabled",
			"outlier_bits", outlierBits, "outlier_count", outlierCount,
			"regular_bits", bits, "regular_channels", m.regularChannelCount(),
			"effective_bits", float32(outlierCount*outlierBits+m.regularChannelCount()*bits)/float32(headDim))
	}
	return m
}

// EnsureLayer allocates per-layer packed and scales tensors on first use.
// When outlier split is active, also allocates the outlier sub-block tensors.
func (m *ggmlTQCompressedK) EnsureLayer(layer, capacity int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.packedTensors[layer]; ok {
		return
	}
	packedBytes := m.regularPackedBytes()

	// Size the per-layer context hint by how many tensors we're allocating.
	// 2 (regular) + 3 (outlier) = 5 tensors when outlier split is on.
	ctxHint := 2
	if m.hasOutliers() {
		ctxHint = 5
	}
	// TQ tensors must always be GPU-resident; use newTQContext, not Layer(layer),
	// which would allocate CPU memory for layers assigned to CPU.
	ctx := m.backend.newTQContext(ctxHint)
	// packed: interleaved as (cell*numKVHeads+head)*packedBytes — matches encode kernel layout.
	packed := ctx.Zeros(ml.DTypeI8, packedBytes*m.numKVHeads, capacity).(*Tensor)
	// scales: scales[cell*numKVHeads+head] — cell-major.
	scales := ctx.Zeros(ml.DTypeF32, m.numKVHeads, capacity).(*Tensor)

	m.layerCtxs[layer] = ctx
	m.packedTensors[layer] = packed
	m.scalesTensors[layer] = scales

	if m.hasOutliers() {
		oPackedBytes := m.outlierPackedBytes()
		m.outlierPackedTensors[layer] = ctx.Zeros(ml.DTypeI8, oPackedBytes*m.numKVHeads, capacity).(*Tensor)
		m.outlierScalesTensors[layer] = ctx.Zeros(ml.DTypeF32, m.numKVHeads, capacity).(*Tensor)
		m.outlierIndicesTensors[layer] = ctx.Zeros(ml.DTypeI8, m.outlierCount*m.numKVHeads, capacity).(*Tensor)
	}
}

// EncodeK creates a GGML_OP_TQ_ENCODE graph node.
// EnsureLayer must have been called for this layer before EncodeK;
// the forward pass is single-threaded per cache so the map reads below
// race only with concurrent EnsureLayer calls, which the contract forbids.
// firstCell is the index of the first cache slot being written
// (cells are sequential: firstCell+0, firstCell+1, ...).
// Returns a view of the packed buffer (use as encodeResult in DequantK).
func (m *ggmlTQCompressedK) EncodeK(ctx ml.Context, layer int, key ml.Tensor, firstCell int) ml.Tensor {
	packed := m.packedTensors[layer]
	if packed == nil {
		return nil
	}
	scales := m.scalesTensors[layer]
	if m.hasOutliers() {
		if oPacked := m.outlierPackedTensors[layer]; oPacked != nil {
			return packed.TQEncodeOutlier(ctx, scales, key, m.rotTensor, firstCell, m.boundariesTensor, m.bits,
				oPacked, m.outlierScalesTensors[layer], m.outlierIndicesTensors[layer], m.outlierBoundariesTensor,
				m.outlierBits, m.outlierCount)
		}
	}
	return packed.TQEncode(ctx, scales, key, m.rotTensor, firstCell, m.boundariesTensor, m.bits)
}

// DequantK creates a GGML_OP_TQ_DEQUANT graph node. encodeResult is the
// view returned by EncodeK (establishes encode→dequant ordering). See
// EncodeK for the single-threaded access contract.
func (m *ggmlTQCompressedK) DequantK(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int) ml.Tensor {
	scales := m.scalesTensors[layer]
	if scales == nil || encodeResult == nil || nCells <= 0 {
		return nil
	}
	if m.hasOutliers() {
		if oPacked := m.outlierPackedTensors[layer]; oPacked != nil {
			return encodeResult.(*Tensor).TQDequantOutlier(ctx, scales, m.codebookTensor,
				m.headDim, m.numKVHeads, nCells, firstCell, m.bits,
				oPacked, m.outlierScalesTensors[layer], m.outlierIndicesTensors[layer], m.outlierCodebookTensor,
				m.outlierBits, m.outlierCount)
		}
	}
	return encodeResult.(*Tensor).TQDequant(ctx, scales, m.codebookTensor,
		m.headDim, m.numKVHeads, nCells, firstCell, m.bits)
}

// fusedKernelSupports reports whether the fused TQ flash-attention kernel
// should be used.
//
// The fused path is the default for supported configurations (headDim=128,
// bits=2 or 3). It decodes packed K and V bits inline during flash attention
// as a fallback for configurations where DequantKV is unsupported.  The
// inline-decode path is slower than DequantKV + stock FA on all measured
// hardware — DequantKV is always preferred when available.
func (m *ggmlTQCompressedK) fusedKernelSupports() bool {
	if m.headDim != 128 {
		return false
	}
	if m.bits != 2 && m.bits != 3 {
		return false
	}
	// Outlier split changes the packed layout: the fused inline-decode FA
	// kernel reads the packed buffer directly and doesn't know about outlier
	// sub-blocks. Route to path 5 (separate dequant + stock FA) when outliers
	// are active. Extending the fused kernel to handle outliers is deliberately
	// NOT done — the fused inline-decode path is already documented as 17.6x
	// slower than separate dequant + stock FA (feedback_cuda_kernel_optimization.md),
	// so adding more ALU work (outlier scan / popcount / dual codebook shuffles)
	// to that inner loop moves it further from the correct architecture.
	if m.hasOutliers() {
		return false
	}
	return true
}

// GetAsTQTensor wraps the packed K buffer for the given layer as a tqTensor
// so that ScaledDotProductAttention can dispatch to the fused kernel.
// Returns (nil, false) when the fused path is not supported.
func (m *ggmlTQCompressedK) GetAsTQTensor(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int) (ml.Tensor, bool) {
	if !m.fusedKernelSupports() {
		return nil, false
	}
	scales := m.scalesTensors[layer]
	if scales == nil || encodeResult == nil || nCells <= 0 {
		return nil, false
	}
	return &tqTensor{
		Tensor:    encodeResult.(*Tensor),
		scales:    scales,
		codebook:  m.codebookTensor,
		bits:      m.bits,
		headDim:   m.headDim,
		nKVHeads:  m.numKVHeads,
		nCells:    nCells,
		firstCell: firstCell,
	}, true
}

// GetAsTQTensorKV wraps both packed K and packed V buffers as a tqTensor for
// the fully fused K+V TQ flash-attention path. Returns (nil, false) when
// fused is not supported or V compression is not yet active for this layer.
func (m *ggmlTQCompressedK) GetAsTQTensorKV(ctx ml.Context, layer int, kEncodeResult, vEncodeResult ml.Tensor, firstCell, nCells int) (ml.Tensor, bool) {
	if !m.fusedKernelSupports() {
		return nil, false
	}
	kScales := m.scalesTensors[layer]
	vScales := m.vScalesTensors[layer]
	if kScales == nil || kEncodeResult == nil || nCells <= 0 {
		return nil, false
	}
	if vScales == nil || vEncodeResult == nil {
		return nil, false
	}
	return &tqTensor{
		Tensor:    kEncodeResult.(*Tensor),
		scales:    kScales,
		codebook:  m.codebookTensor,
		bits:      m.bits,
		headDim:   m.headDim,
		nKVHeads:  m.numKVHeads,
		nCells:    nCells,
		firstCell: firstCell,
		vPacked:   vEncodeResult.(*Tensor),
		vScales:   vScales,
		vCodebook: m.vCodebookTensor,
		vBits:     m.vBits,
	}, true
}

func (m *ggmlTQCompressedK) RotationMatrix(_ ml.Context, _ int) ml.Tensor {
	return m.rotTensor
}

// RotationMatrixR returns R (not R^T) for use as the V rotation undo matrix.
// mul_mat(R, R^T @ v) = v (recovers original from rotated V).
func (m *ggmlTQCompressedK) RotationMatrixR() ml.Tensor {
	return m.rotInverseTensor
}

// EnsureVLayer allocates per-layer V packed and scales tensors on first use.
func (m *ggmlTQCompressedK) EnsureVLayer(layer, capacity int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.vPackedTensors[layer]; ok {
		return
	}
	packedBytes := (m.headDim*m.vBits + 7) / 8

	ctx := m.backend.newTQContext(2)
	packed := ctx.Zeros(ml.DTypeI8, packedBytes*m.numKVHeads, capacity).(*Tensor)
	scales := ctx.Zeros(ml.DTypeF32, m.numKVHeads, capacity).(*Tensor)

	m.vLayerCtxs[layer] = ctx
	m.vPackedTensors[layer] = packed
	m.vScalesTensors[layer] = scales
}

// EncodeV creates a GGML_OP_TQ_ENCODE_V graph node.
// EnsureVLayer must have been called for this layer before EncodeV.
func (m *ggmlTQCompressedK) EncodeV(ctx ml.Context, layer int, value ml.Tensor, firstCell int) ml.Tensor {
	packed := m.vPackedTensors[layer]
	if packed == nil {
		return nil
	}
	// Pass the K rotation matrix so V outlier energy spreads evenly before
	// quantization. SDPA's post-attention R @ output step undoes the rotation.
	return packed.TQEncodeV(ctx, m.vScalesTensors[layer], value, m.rotTensor, firstCell, m.vBoundariesTensor, m.vBits)
}

// EncodeKV creates a single GGML_OP_TQ_ENCODE_KV graph node encoding both
// K and V, halving scheduler overhead vs separate EncodeK + EncodeV.
// Returns (kEncodeResult, vEncodeResult) — both reference the same op for
// graph dependency tracking.
//
// When outlier-split is active, the combined encode kernel is not used
// because it only understands the uniform packed layout. Falls back to
// separate EncodeK (outlier-aware) + EncodeV (uniform) calls.
func (m *ggmlTQCompressedK) EncodeKV(ctx ml.Context, layer int, key, value ml.Tensor, firstCell int) (ml.Tensor, ml.Tensor) {
	if m.hasOutliers() {
		return m.EncodeK(ctx, layer, key, firstCell), m.EncodeV(ctx, layer, value, firstCell)
	}
	kPacked := m.packedTensors[layer]
	vPacked := m.vPackedTensors[layer]
	if kPacked == nil || vPacked == nil {
		return nil, nil
	}
	kResult := kPacked.TQEncodeKV(ctx,
		m.scalesTensors[layer], key, m.rotTensor, m.boundariesTensor,
		vPacked, m.vScalesTensors[layer], value, m.vBoundariesTensor,
		firstCell, m.bits, m.vBits)

	// kResult is the EncodeKV op output (K packed view); the scheduler uses it
	// to order DequantKV after EncodeKV.  V packed buffer was written as a side
	// effect by the combined kernel.  Return the V packed tensor directly —
	// DequantKV reads its data pointer (which now contains the encoded V).
	// Graph ordering is still correct: DequantKV depends on kResult (src[0]),
	// and both kernels run on the same CUDA stream.
	return kResult, vPacked
}

// DequantV creates a GGML_OP_TQ_DEQUANT graph node for V.
// encodeResult is the view returned by EncodeV (establishes encode→dequant ordering).
func (m *ggmlTQCompressedK) DequantV(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int) ml.Tensor {
	scales := m.vScalesTensors[layer]
	if scales == nil || encodeResult == nil || nCells <= 0 {
		return nil
	}
	return encodeResult.(*Tensor).TQDequant(ctx, scales, m.vCodebookTensor,
		m.headDim, m.numKVHeads, nCells, firstCell, m.vBits)
}

// DequantKV creates a single GGML_OP_TQ_DEQUANT_KV graph node that dequants
// both K and V in one op, halving scheduler overhead vs separate DequantK+DequantV.
// Returns (kTensor, vTensor) as views into the combined output.
//
// When outlier-split is active, the combined kernel cannot be used because
// its K reader assumes the uniform packed layout. Returns (nil, nil) to
// force Get() to fall through to the separate DequantK + DequantV path.
func (m *ggmlTQCompressedK) DequantKV(ctx ml.Context, layer int, kEncodeResult, vEncodeResult ml.Tensor, firstCell, nCells int) (ml.Tensor, ml.Tensor) {
	if m.hasOutliers() {
		return nil, nil
	}
	kScales := m.scalesTensors[layer]
	vScales := m.vScalesTensors[layer]
	if kScales == nil || kEncodeResult == nil || nCells <= 0 {
		return nil, nil
	}
	if vScales == nil || vEncodeResult == nil {
		return nil, nil
	}
	combined := TQDequantKV(ctx, m.backend,
		kEncodeResult.(*Tensor), kScales, m.codebookTensor,
		vEncodeResult.(*Tensor), vScales, m.vCodebookTensor,
		m.rotInverseTensor, // R matrix for fused V rotation undo
		m.headDim, m.numKVHeads, nCells, firstCell, m.bits, m.vBits)

	// Split the [headDim, numKVHeads, nCells, 2] output into K and V views.
	planeBytes := m.headDim * m.numKVHeads * nCells * 2 // f16 = 2 bytes
	kView := combined.View(ctx, 0, m.headDim, combined.Stride(1), m.numKVHeads, combined.Stride(2), nCells)
	vView := combined.View(ctx, planeBytes, m.headDim, combined.Stride(1), m.numKVHeads, combined.Stride(2), nCells)

	return kView, vView
}

func (m *ggmlTQCompressedK) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, ctx := range m.layerCtxs {
		ctx.Close()
	}
	for _, ctx := range m.vLayerCtxs {
		ctx.Close()
	}
	if m.rotCtx != nil {
		m.rotCtx.Close()
	}
	if m.sharedCtx != nil {
		m.sharedCtx.Close()
	}
	m.packedTensors = nil
	m.scalesTensors = nil
	m.layerCtxs = nil
	m.vPackedTensors = nil
	m.vScalesTensors = nil
	m.vLayerCtxs = nil
	m.rotCtx = nil
	m.sharedCtx = nil
}
