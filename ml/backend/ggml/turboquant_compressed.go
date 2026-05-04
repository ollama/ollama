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
	backend    *Backend
	headDim    int
	numKVHeads int
	bits       int

	// Outlier-split config (post-rotation top-K channel split). When
	// outlierCount > 0, EnsureLayer allocates additional tensors for an
	// outlier sub-block encoded at outlierBits, and the encode/dequant
	// kernels follow the outlier-aware path. When 0, uses pure uniform
	// per-channel Lloyd-Max at `bits`.
	outlierBits  int
	outlierCount int

	// Asymmetric primary quantization: when true, each per-head rotated vector
	// is centred by its mean before scalar quantization and that mean is stored
	// alongside the RMS scale. Decoding unconditionally adds the mean back.
	asymmetricPrimary bool

	// QJL residual sketch dimension. When > 0, EnsureLayer allocates qjlPacked
	// and qjlNorm tensors, and the encode/dequant kernels compute a random-
	// Gaussian projection of the primary-quantization residual, storing sign
	// bits and the residual L2 norm.
	qjlRows int

	mu sync.Mutex

	// Per-layer ggml tensors, allocated lazily via EnsureLayer.
	layerCtxs     map[int]ml.Context
	packedTensors map[int]*Tensor // regular sub-block: [regularPackedBytes*numKVHeads, capacity] i8
	scalesTensors map[int]*Tensor // regular scales: [numKVHeads, capacity] f32

	// Asymmetric zero tensors (populated only when asymmetricPrimary is true).
	zerosTensors        map[int]*Tensor // regular zeros: [numKVHeads, capacity] f32
	outlierZerosTensors map[int]*Tensor // outlier zeros: [numKVHeads, capacity] f32

	// Outlier sub-block per-layer tensors (populated only when outlierCount > 0).
	outlierPackedTensors  map[int]*Tensor // [outlierPackedBytes*numKVHeads, capacity] i8
	outlierScalesTensors  map[int]*Tensor // [numKVHeads, capacity] f32
	outlierIndicesTensors map[int]*Tensor // [outlierCount*numKVHeads, capacity] i8 (channel idx)

	// QJL per-layer tensors (populated only when qjlRows > 0).
	qjlPackedTensors map[int]*Tensor // [qjlPackedBytes*numKVHeads, capacity] i8
	qjlNormTensors   map[int]*Tensor // [numKVHeads, capacity] f32

	// Per-layer K projection bias tensors (populated when the model has K bias,
	// e.g. Qwen2). Shape: [numKVHeads * headDim] f32. Subtracted from K before
	// rotation in the CUDA encoder; the decoder requires no change (bias cancels in softmax).
	kBiasTensors map[int]*Tensor

	// Rotation matrix R^T, shared across layers: [headDim, headDim] f32.
	rotCtx    ml.Context
	rotTensor *Tensor // stores R^T row-major (used for K encode and Q rotate)

	// Rotation matrix R (transpose of R^T), for undoing V rotation in SDPA.
	// mul_mat(rotInverseTensor, x) = R @ x (recovers original from R^T @ x).
	rotInverseTensor *Tensor // stores R row-major

	// QJL projection matrix, shared across layers: [qjlRows, headDim] f32.
	// Generated on CPU from the preset's QJL seed and uploaded once.
	qjlProjectionCtx    ml.Context
	qjlProjectionTensor *Tensor

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

	// preferFusedAttention is true on Metal. The DequantKV → stock FA path
	// writes a full f16 intermediate buffer before attention, doubling KV
	// bandwidth vs reading packed data directly. On Metal at long context the
	// fused kernel (kernel_tq_fattn_vec_packed) is dramatically faster because
	// it reads packed K+V once and never materialises the f16 intermediate.
	// On CUDA, DequantKV + stock FA is faster because cuDNN/cuBLAS flash
	// attention is highly tuned and the intermediate buffer stays in L2.
	preferFusedAttention bool
}

// hasOutliers reports whether outlier-split is active for this manager.
func (m *ggmlTQCompressedK) hasOutliers() bool {
	return m.outlierCount > 0 && m.outlierBits > 0 && m.outlierCount < m.headDim
}

// hasQJL reports whether QJL residual sketch is active.
func (m *ggmlTQCompressedK) hasQJL() bool {
	return m.qjlRows > 0
}

// SetLayerKBias stores the K projection bias tensor for the given layer.
// Called once per layer at model init when the model has a K bias (e.g. Qwen2).
// The bias is subtracted from K before rotation in the CUDA encoder; the
// decoder requires no change because the constant bias cancels in softmax.
func (m *ggmlTQCompressedK) SetLayerKBias(layer int, bias ml.Tensor) {
	if bias == nil {
		return
	}
	if t, ok := bias.(*Tensor); ok {
		m.kBiasTensors[layer] = t
	}
}

// qjlPackedBytes is the padded per-head byte count for QJL sign bits.
func (m *ggmlTQCompressedK) qjlPackedBytes() int {
	if !m.hasQJL() {
		return 0
	}
	raw := (m.qjlRows + 7) / 8
	return (raw + 3) &^ 3
}

// PreferFusedAttention reports whether the fused flash-attention path
// (packed K+V decoded inline) should be tried before DequantKV + stock FA.
// True on Metal: the DequantKV path writes a full f16 intermediate buffer that
// doubles KV bandwidth at long context. False on CUDA/ROCm where DequantKV +
// stock FA is faster due to large L2 caches and highly-tuned flash attention.
func (m *ggmlTQCompressedK) PreferFusedAttention() bool {
	return m.preferFusedAttention
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
//
// Must match the stride computation in ggml-cuda/tq-encode.cu and
// tq-dequant.cu (they recompute from regularCount and bits); changing
// the formula here requires matching edits in those kernels.
func (m *ggmlTQCompressedK) regularPackedBytes() int {
	raw := (m.regularChannelCount()*m.bits + 7) / 8
	return (raw + 3) &^ 3
}

// outlierPackedBytes is the padded per-head byte count for the outlier
// sub-block. Same 4-byte alignment as regularPackedBytes() for the same
// reason: atomicOr-on-word in the encode kernel.
//
// Must match the stride computation in ggml-cuda/tq-encode.cu and
// tq-dequant.cu (they recompute from outlierCount and outlierBits).
func (m *ggmlTQCompressedK) outlierPackedBytes() int {
	if !m.hasOutliers() {
		return 0
	}
	raw := (m.outlierCount*m.outlierBits + 7) / 8
	return (raw + 3) &^ 3
}

func (b *Backend) NewTQCompressedKManager(headDim, numKVHeads, bits int, rotationSeed uint64, vBits, outlierBits, outlierCount int, asymmetricPrimary bool, qjlRows int) ml.TQCompressedKManager {
	// TurboQuant ops run on CUDA (NVIDIA Pascal+), ROCm/HIP (AMD RDNA1+,
	// gfx1010+), or Metal (Apple Silicon). The gate is wave32: the kernels
	// hard-code a 32-lane shuffle for codebook lookup. On wave64 AMD (Vega/
	// GCN/CDNA) the HIP shim's __shfl(…, 32) sub-partitions the 64-lane warp
	// and the upper 32 lanes return garbage — those are rejected. Metal SIMD
	// groups are always 32-wide on Apple Silicon, so Metal is unconditionally
	// admitted. Scan the scheduler buffer types, pick the first TQ-capable
	// GPU, and warn clearly if there's no suitable device.
	scan := b.scanTQDevices()
	if !scan.selectedOK {
		if len(scan.Skipped) > 0 {
			slog.Warn("turboquant: no TQ-capable GPU found; falling back to f16 KV cache. "+
				"TurboQuant requires NVIDIA Pascal (cc 6.0+), AMD RDNA1+ (gfx1010+, wave32), or Apple Silicon (Metal).",
				"skipped_gpus", scan.Skipped)
		} else {
			slog.Warn("turboquant: no GPU backend available, falling back to f16 KV cache")
		}
		return nil
	}
	if len(scan.Skipped) > 0 {
		slog.Warn("turboquant: skipping unsupported GPU(s); TQ tensors will be placed on the "+
			"first wave32 device (NVIDIA Pascal+, AMD RDNA1+, or Apple Silicon). To silence "+
			"this warning, hide the unsupported cards with CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES.",
			"selected", scan.SelectedName+" (cc "+scan.SelectedCC+")",
			"skipped", scan.Skipped)
	}
	if qjlRows > 0 && outlierCount == 0 {
		slog.Warn("turboquant: QJL residual sketch requires outlier-split kernels "+
			"and cannot be used without them. Falling back to f16 KV cache.",
			"qjl_rows", qjlRows, "outlier_count", outlierCount)
		return nil
	}
	// Metal's kernel_tq_encode_outlier supports asymmetric+outlier (since
	// the asymmetric port). QJL is still not implemented on Metal; the host
	// gates qjl-on configurations to f16. No ship preset uses QJL — only
	// test fixtures opt into it directly.
	if scan.SelectedLibrary == "Metal" && qjlRows > 0 {
		slog.Warn("turboquant: QJL not yet implemented on Metal; falling back to f16 KV cache",
			"qjl_rows", qjlRows, "outlier_count", outlierCount)
		return nil
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

	// When outlier split is active, concatenate the outlier codebook into the
	// regular codebook tensor so the fused kernel can access both via a single
	// src[] slot. Layout: [regular (1<<bits), outlier (1<<outlierBits)].
	// Non-outlier paths read only the first 1<<bits entries — backwards compatible.
	// Dequant/encode paths still take the separate outlierCodebookTensor below;
	// only the fused flash-attn kernel consumes the concatenated layout.
	var codebookT *Tensor
	if outlierCount > 0 && outlierBits > 0 && outlierCount < headDim {
		oCodebook := turboquant.ExportCodebook(headDim, outlierBits)
		combined := make([]float32, len(codebook)+len(oCodebook))
		copy(combined, codebook)
		copy(combined[len(codebook):], oCodebook)
		codebookT = sharedCtx.FromFloats(combined, len(combined)).(*Tensor)
	} else {
		codebookT = sharedCtx.FromFloats(codebook, len(codebook)).(*Tensor)
	}
	boundariesT := sharedCtx.FromFloats(boundaries, len(boundaries)).(*Tensor)

	// Outlier codebook/boundaries at a different bit width. Use headDim as
	// the codebook dim (same reason as regular codebook above).
	// Note: outlierCodebookTensor remains a separate tensor for the
	// encode/dequant kernels (they take a distinct src slot). The fused
	// flash-attention kernel uses the concatenated codebookT above instead.
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
	for i := range headDim {
		for j := range headDim {
			rotData[i*headDim+j] = rot.Matrix[j*headDim+i]
		}
	}
	rotInverseData := make([]float32, headDim*headDim)
	copy(rotInverseData, rot.Matrix)
	rotCtx := b.newTQContext(2)
	rotTensor := rotCtx.FromFloats(rotData, headDim, headDim).(*Tensor)
	rotInverseTensor := rotCtx.FromFloats(rotInverseData, headDim, headDim).(*Tensor)

	// QJL projection matrix: [qjlRows, headDim] f32, generated on CPU from the
	// preset's QJL seed and uploaded once.  The seed matches the CPU reference
	// (rotationSeed ^ 0x9e3779b97f4a7c15).
	var qjlProjectionCtx ml.Context
	var qjlProjectionTensor *Tensor
	if qjlRows > 0 {
		qjlSeed := rotationSeed ^ 0x9e3779b97f4a7c15
		qjlProjData := turboquant.BuildQJLProjection(headDim, qjlRows, qjlSeed)
		qjlProjectionCtx = b.newTQContext(1)
		qjlProjectionTensor = qjlProjectionCtx.FromFloats(qjlProjData, headDim, qjlRows).(*Tensor)
	}

	m := &ggmlTQCompressedK{
		backend:                 b,
		headDim:                 headDim,
		numKVHeads:              numKVHeads,
		bits:                    bits,
		outlierBits:             outlierBits,
		outlierCount:            outlierCount,
		asymmetricPrimary:       asymmetricPrimary,
		qjlRows:                 qjlRows,
		kBiasTensors:            make(map[int]*Tensor),
		layerCtxs:               make(map[int]ml.Context),
		packedTensors:           make(map[int]*Tensor),
		scalesTensors:           make(map[int]*Tensor),
		zerosTensors:            make(map[int]*Tensor),
		outlierZerosTensors:     make(map[int]*Tensor),
		outlierPackedTensors:    make(map[int]*Tensor),
		outlierScalesTensors:    make(map[int]*Tensor),
		outlierIndicesTensors:   make(map[int]*Tensor),
		qjlPackedTensors:        make(map[int]*Tensor),
		qjlNormTensors:          make(map[int]*Tensor),
		rotCtx:                  rotCtx,
		rotTensor:               rotTensor,
		rotInverseTensor:        rotInverseTensor,
		qjlProjectionCtx:        qjlProjectionCtx,
		qjlProjectionTensor:     qjlProjectionTensor,
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
		preferFusedAttention:    scan.SelectedLibrary == "Metal",
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
	// Base: 2 (packed + scales). Outlier adds 3. Asymmetric adds 2 (zeros).
	// QJL adds 2 (qjlPacked + qjlNorm).
	ctxHint := 2
	if m.hasOutliers() {
		ctxHint += 3
	}
	if m.asymmetricPrimary {
		ctxHint += 2
	}
	if m.hasQJL() {
		ctxHint += 2
	}
	// TQ tensors must always be GPU-resident; use newTQContext, not Layer(layer),
	// which would allocate CPU memory for layers assigned to CPU.
	ctx := m.backend.newTQContext(ctxHint)
	// Opt this layer's TQ persistent buffers into the scheduler's per-layer
	// Cache accounting so the (packed K, scales, optional outlier sub-block)
	// tensors below flow into btDeviceMemory.Cache[layer] via newTensor, not
	// the anonymous Graph bucket. Without this, scanTQDevices' scheduler can't
	// see TQ's real KV footprint and may mis-plan context-fit decisions.
	ctx.layer = layer
	// packed: interleaved as (cell*numKVHeads+head)*packedBytes — matches encode kernel layout.
	packed := ctx.Zeros(ml.DTypeI8, packedBytes*m.numKVHeads, capacity).(*Tensor)
	// scales: scales[cell*numKVHeads+head] — cell-major.
	scales := ctx.Zeros(ml.DTypeF32, m.numKVHeads, capacity).(*Tensor)

	m.layerCtxs[layer] = ctx
	m.packedTensors[layer] = packed
	m.scalesTensors[layer] = scales

	if m.asymmetricPrimary {
		m.zerosTensors[layer] = ctx.Zeros(ml.DTypeF32, m.numKVHeads, capacity).(*Tensor)
		if m.hasOutliers() {
			m.outlierZerosTensors[layer] = ctx.Zeros(ml.DTypeF32, m.numKVHeads, capacity).(*Tensor)
		}
	}

	if m.hasOutliers() {
		oPackedBytes := m.outlierPackedBytes()
		m.outlierPackedTensors[layer] = ctx.Zeros(ml.DTypeI8, oPackedBytes*m.numKVHeads, capacity).(*Tensor)
		m.outlierScalesTensors[layer] = ctx.Zeros(ml.DTypeF32, m.numKVHeads, capacity).(*Tensor)
		m.outlierIndicesTensors[layer] = ctx.Zeros(ml.DTypeI8, m.outlierCount*m.numKVHeads, capacity).(*Tensor)
	}

	if m.hasQJL() {
		qjlPackedBytes := m.qjlPackedBytes()
		m.qjlPackedTensors[layer] = ctx.Zeros(ml.DTypeI8, qjlPackedBytes*m.numKVHeads, capacity).(*Tensor)
		m.qjlNormTensors[layer] = ctx.Zeros(ml.DTypeF32, m.numKVHeads, capacity).(*Tensor)
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
			var zeros, outlierZeros ml.Tensor
			if m.asymmetricPrimary {
				zeros = m.zerosTensors[layer]
				outlierZeros = m.outlierZerosTensors[layer]
			}
			var qjlPacked, qjlNorm, qjlProjection ml.Tensor
			if m.hasQJL() {
				qjlPacked = m.qjlPackedTensors[layer]
				qjlNorm = m.qjlNormTensors[layer]
				qjlProjection = m.qjlProjectionTensor
			}
			var kBias ml.Tensor
			if t := m.kBiasTensors[layer]; t != nil {
				kBias = t
			}
			return packed.TQEncodeOutlier(ctx, scales, key, m.rotTensor, firstCell, m.boundariesTensor, m.bits,
				oPacked, m.outlierScalesTensors[layer], m.outlierIndicesTensors[layer], m.outlierBoundariesTensor,
				m.outlierBits, m.outlierCount,
				zeros, outlierZeros,
				qjlPacked, qjlNorm, qjlProjection, m.qjlRows,
				m.codebookTensor, m.outlierCodebookTensor,
				kBias)
		}
	}
	var zeros ml.Tensor
	if m.asymmetricPrimary {
		zeros = m.zerosTensors[layer]
	}
	var kBias ml.Tensor
	if t := m.kBiasTensors[layer]; t != nil {
		kBias = t
	}
	return packed.TQEncode(ctx, scales, key, m.rotTensor, firstCell, m.boundariesTensor, m.bits, zeros, kBias, m.codebookTensor)
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
			var zeros, outlierZeros ml.Tensor
			if m.asymmetricPrimary {
				zeros = m.zerosTensors[layer]
				outlierZeros = m.outlierZerosTensors[layer]
			}
			var qjlPacked, qjlNorm, qjlProjection ml.Tensor
			if m.hasQJL() {
				qjlPacked = m.qjlPackedTensors[layer]
				qjlNorm = m.qjlNormTensors[layer]
				qjlProjection = m.qjlProjectionTensor
			}
			return encodeResult.(*Tensor).TQDequantOutlier(ctx, scales, m.codebookTensor,
				m.headDim, m.numKVHeads, nCells, firstCell, m.bits,
				oPacked, m.outlierScalesTensors[layer], m.outlierIndicesTensors[layer], m.outlierCodebookTensor,
				m.outlierBits, m.outlierCount,
				zeros, outlierZeros,
				qjlPacked, qjlNorm, qjlProjection, m.qjlRows)
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
	// D=128 on all backends; D=256 only on Metal (kernel_tq_fattn_vec_*{,_d256}).
	// CUDA still has only the D=128 kernel, so gemma3 (D=256) stays off the
	// fused path on CUDA.
	switch m.headDim {
	case 128:
	case 256:
		if !m.preferFusedAttention {
			return false
		}
	default:
		return false
	}
	if m.bits != 2 && m.bits != 3 && m.bits != 4 {
		return false
	}
	// K-only fused path with outlier split is disabled: GetAsTQTensor with
	// outlier data produces wrong results. K+V fused via GetAsTQTensorKV works.
	// tq*kqa falls back to DequantK as a result.
	if m.hasOutliers() && m.vBits == 0 {
		return false
	}
	// Metal does not yet have outlier-aware fattn kernels (kernel_tq_fattn_vec*
	// read only the regular packed buffer; there is no kernel_tq_fattn_vec_outlier
	// or kernel_tq_fattn_vec_packed_outlier on Metal). Force outlier presets to
	// the DequantK + stock-FA slow path, which is correct after the
	// kernel_tq_dequant_outlier asymmetric port. Once outlier-aware fattn is
	// ported to Metal, drop this guard.
	if m.hasOutliers() && m.preferFusedAttention {
		return false
	}
	// K+V outlier path (tq*qa): GetAsTQTensorKV handles outlier decode inline.
	// On Pascal (P40, cc 6.1) it is slower than DequantK + stockFA for single-
	// token decode due to shared-memory pressure from the dual-stream loop.
	// On Ampere+ the performance gap narrows. For PPL measurement
	// (prefill-dominated) the decode throughput does not matter.
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
	t := &tqTensor{
		Tensor:    encodeResult.(*Tensor),
		scales:    scales,
		codebook:  m.codebookTensor,
		bits:      m.bits,
		headDim:   m.headDim,
		nKVHeads:  m.numKVHeads,
		nCells:    nCells,
		firstCell: firstCell,
	}
	if m.asymmetricPrimary {
		t.asymmetric = true
		t.zeros = m.zerosTensors[layer]
	}
	if m.hasQJL() {
		t.qjlRows = m.qjlRows
		t.qjlPacked = m.qjlPackedTensors[layer]
		t.qjlNorm = m.qjlNormTensors[layer]
		t.qjlProjection = m.qjlProjectionTensor
	}
	if m.hasOutliers() {
		t.outlierPacked = m.outlierPackedTensors[layer]
		t.outlierScales = m.outlierScalesTensors[layer]
		t.outlierIndices = m.outlierIndicesTensors[layer]
		t.outlierCodebook = m.outlierCodebookTensor
		t.outlierBits = m.outlierBits
		t.outlierCount = m.outlierCount
		t.outlierPackedBytes = m.outlierPackedBytes()
		if m.asymmetricPrimary {
			t.outlierZeros = m.outlierZerosTensors[layer]
		}
	}
	return t, true
}

// GetAsTQTensorKV wraps both packed K and packed V buffers as a tqTensor for
// the fully fused K+V TQ flash-attention path. Returns (nil, false) when
// fused is not supported or V compression is not enabled for this layer.
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
	t := &tqTensor{
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
	}
	if m.asymmetricPrimary {
		t.asymmetric = true
		t.zeros = m.zerosTensors[layer]
	}
	if m.hasQJL() {
		t.qjlRows = m.qjlRows
		t.qjlPacked = m.qjlPackedTensors[layer]
		t.qjlNorm = m.qjlNormTensors[layer]
		t.qjlProjection = m.qjlProjectionTensor
	}
	if m.hasOutliers() {
		t.outlierPacked = m.outlierPackedTensors[layer]
		t.outlierScales = m.outlierScalesTensors[layer]
		t.outlierIndices = m.outlierIndicesTensors[layer]
		t.outlierCodebook = m.outlierCodebookTensor
		t.outlierBits = m.outlierBits
		t.outlierCount = m.outlierCount
		t.outlierPackedBytes = m.outlierPackedBytes()
		if m.asymmetricPrimary {
			t.outlierZeros = m.outlierZerosTensors[layer]
		}
	}
	return t, true
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
	// 4-byte alignment — matches regularPackedBytes() so the scheduler and the
	// Go-side allocator agree on padded bytes per head. The encode/dequant
	// kernels read the raw bits; padding is never touched.
	raw := (m.headDim*m.vBits + 7) / 8
	packedBytes := (raw + 3) &^ 3

	ctx := m.backend.newTQContext(2)
	// Same per-layer Cache accounting as EnsureLayer — see that comment for why.
	ctx.layer = layer
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
	return packed.TQEncodeV(ctx, m.vScalesTensors[layer], value, m.rotTensor, firstCell, m.vBoundariesTensor, m.vBits, m.vCodebookTensor)
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
	// Asymmetric presets write per-cell zeros (mean offset) into a separate tensor
	// during K encode — TQEncodeKV has no zeros output, so it silently drops the
	// mean-centering step. Route asymmetric presets through separate EncodeK + EncodeV.
	if m.hasOutliers() || m.asymmetricPrimary {
		return m.EncodeK(ctx, layer, key, firstCell), m.EncodeV(ctx, layer, value, firstCell)
	}
	kPacked := m.packedTensors[layer]
	vPacked := m.vPackedTensors[layer]
	if kPacked == nil || vPacked == nil {
		return nil, nil
	}
	var kBias ml.Tensor
	if t := m.kBiasTensors[layer]; t != nil {
		kBias = t
	}
	kResult := kPacked.TQEncodeKV(ctx,
		m.scalesTensors[layer], key, m.rotTensor, m.boundariesTensor,
		vPacked, m.vScalesTensors[layer], value, m.vBoundariesTensor,
		firstCell, m.bits, m.vBits, kBias, m.codebookTensor, m.vCodebookTensor)

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
// When outlier-split is active the K plane is dequanted via the
// regular+outlier overwrite kernel; V is always plain dequant (no V outliers
// in any ship preset).
func (m *ggmlTQCompressedK) DequantKV(ctx ml.Context, layer int, kEncodeResult, vEncodeResult ml.Tensor, firstCell, nCells int) (ml.Tensor, ml.Tensor) {
	kScales := m.scalesTensors[layer]
	vScales := m.vScalesTensors[layer]
	if kScales == nil || kEncodeResult == nil || nCells <= 0 {
		return nil, nil
	}
	if vScales == nil || vEncodeResult == nil {
		return nil, nil
	}

	var (
		kOutlierPacked   *Tensor
		kOutlierScales   *Tensor
		kOutlierIndices  *Tensor
		kOutlierCodebook *Tensor
		kZeros           *Tensor
		kOutlierZeros    *Tensor
		outlierBits      int
		outlierCount     int
	)
	if m.hasOutliers() {
		kOutlierPacked = m.outlierPackedTensors[layer]
		kOutlierScales = m.outlierScalesTensors[layer]
		kOutlierIndices = m.outlierIndicesTensors[layer]
		kOutlierCodebook = m.outlierCodebookTensor
		kZeros = m.zerosTensors[layer]
		kOutlierZeros = m.outlierZerosTensors[layer]
		outlierBits = m.outlierBits
		outlierCount = m.outlierCount
		if kOutlierPacked == nil || kOutlierScales == nil || kOutlierIndices == nil || kOutlierCodebook == nil {
			return nil, nil
		}
	}

	combined := TQDequantKV(ctx, m.backend,
		kEncodeResult.(*Tensor), kScales, m.codebookTensor,
		vEncodeResult.(*Tensor), vScales, m.vCodebookTensor,
		m.rotInverseTensor, // R matrix for fused V rotation undo
		m.headDim, m.numKVHeads, nCells, firstCell, m.bits, m.vBits,
		kOutlierPacked, kOutlierScales, kOutlierIndices, kOutlierCodebook,
		kZeros, kOutlierZeros,
		outlierBits, outlierCount)

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
	if m.qjlProjectionCtx != nil {
		m.qjlProjectionCtx.Close()
	}
	if m.sharedCtx != nil {
		m.sharedCtx.Close()
	}
	m.packedTensors = nil
	m.scalesTensors = nil
	m.zerosTensors = nil
	m.outlierZerosTensors = nil
	m.layerCtxs = nil
	m.outlierPackedTensors = nil
	m.outlierScalesTensors = nil
	m.outlierIndicesTensors = nil
	m.qjlPackedTensors = nil
	m.qjlNormTensors = nil
	m.vPackedTensors = nil
	m.vScalesTensors = nil
	m.vLayerCtxs = nil
	m.rotCtx = nil
	m.qjlProjectionCtx = nil
	m.sharedCtx = nil
}
