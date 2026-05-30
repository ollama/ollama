package ggml

import (
	"log/slog"
	"os"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// forceVOnlyFused overrides the per-backend default and enables the V-only
// fused decode path (Path 8-fused) even on backends where DequantV + stock FA
// is faster (currently ROCm). Set OLLAMA_TQ_FORCE_VONLY_FUSED=1 to benchmark
// fused vs DequantV on new AMD hardware before updating the routing default.
// Not for production use — same class as OLLAMA_TQ_FORCE_DEQUANT_V.
var forceVOnlyFused = os.Getenv("OLLAMA_TQ_FORCE_VONLY_FUSED") != ""

// dequantKFallbackWarnOnce guards a one-time warning when the plain DequantK
// materialize path is taken for a WHT-rotated or asymmetric preset without
// outliers (only the outlier-split kernel undoes WHT / applies the mean).
var dequantKFallbackWarnOnce sync.Once

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
	outlierIndicesTensors map[int]*Tensor // [outlierCount*numKVHeads, capacity] i16 (channel idx)

	// Per-layer K projection bias tensors (populated when the model has K bias,
	// e.g. Qwen2). Shape: [numKVHeads * headDim] f32. Subtracted from K before
	// rotation in the CUDA encoder; the decoder requires no change (bias cancels in softmax).
	kBiasTensors map[int]*Tensor

	// rotationSeed is the per-manager WHT seed. Each layer's sign vector is
	// derived as BuildRotation(headDim, rotationSeed ^ uint64(layer+1)), giving
	// an independent rotation per model layer. Nil (zero-value) headDim or
	// non-power-of-2 headDim means no rotation is applied.
	rotationSeed uint64

	// Per-layer WHT sign tensors: [headDim] f32 ±1, one per model layer.
	// Allocated lazily in EnsureLayer alongside the K packed/scales tensors.
	// Stored in the same per-layer context as packed/scales so Close() frees them.
	rotTensors map[int]*Tensor

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

	// preferFusedAttention is true on Metal. For K+V WHT presets the
	// DequantKV → stock FA path (Path 1) writes a full f16 intermediate
	// buffer before attention, doubling KV bandwidth vs the fused
	// inline-decode kernel which reads packed K+V once. At long context on
	// Metal the fused path is dramatically faster; on CUDA/ROCm the
	// intermediate buffer stays in L2 and stock FA is highly tuned, so
	// DequantKV wins. The flag is read in TurboQuantCache.Get() to route
	// K+V WHT presets to Path 2 (fused inline-decode K+V) on Metal while
	// CUDA/ROCm fall through to Path 1.
	preferFusedAttention bool

	// useROCm is true when the selected backend is ROCm/HIP. The V-only
	// fused decode path (Path 8-fused) is disabled by default on ROCm:
	// measured on RX 7600 gfx1102 (RDNA3, GDDR6 ~288 GB/s), fused is
	// 13–40% slower than DequantV + stock ROCm FA at ctx 2048–16384, and
	// the gap widens monotonically with context. Root cause: the fused
	// kernel's warp-shuffle V decode is tuned for NVIDIA shuffle latency;
	// stock ROCm FA is more tightly optimised for the AMD memory hierarchy.
	// Override with OLLAMA_TQ_FORCE_VONLY_FUSED=1 for future hardware sweeps.
	useROCm bool

	// ccMajor is the compute-capability major of the selected CUDA device
	// (0 on non-CUDA). Gates the fused inline-V-decode paths (Path 1.5, Path
	// 8-fused) on cp.async availability (Ampere cc 8.0+) — see SupportsVOnlyFused.
	ccMajor int
}

// hasOutliers reports whether K outlier-split is active for this manager.
// The current outlier path is K-only — V-only mode (bits == 0) means no
// K compression at all, and there are no V outliers in any ship preset.
func (m *ggmlTQCompressedK) hasOutliers() bool {
	return m.bits > 0 && m.outlierCount > 0 && m.outlierBits > 0 && m.outlierCount < m.headDim
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


// PreferFusedAttention reports whether the fused flash-attention path
// (packed K+V decoded inline) should be tried before DequantKV + stock FA
// for K+V WHT presets. True on Metal: the DequantKV path materialises a
// full f16 intermediate that doubles KV bandwidth at long context. False
// on CUDA/ROCm where DequantKV + stock FA is faster (intermediate stays
// in L2 and stock flash attention is highly tuned).
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

func (b *Backend) NewTQCompressedKManager(headDim, numKVHeads, bits int, rotationSeed uint64, vBits, outlierBits, outlierCount int, asymmetricPrimary bool) ml.TQCompressedKManager {
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

	// V-only mode: bits == 0 means K stays as raw f16 in the inner Causal
	// cache and the manager skips all K-side codebook/boundaries/per-layer
	// state. The K methods (EncodeK, DequantK, EnsureLayer) all guard on
	// packedTensors[layer] == nil and return nil cleanly when K is absent.
	kSkipped := bits == 0

	// Codebook and boundaries (same for all layers). Use headDim for the
	// codebook dim parameter regardless of whether outlier split is active:
	// the CPU path does the same (scalarCodebook(dim=headDim, bits)), and
	// after per-sub-block RMS normalization the input distribution is
	// approximately unit-variance Gaussian either way. Using sub-block dims
	// here produced slightly different boundaries that caused observable
	// quality regressions on multi-head configurations.
	var codebook, boundaries []float32
	if !kSkipped {
		codebook = turboquant.ExportCodebook(headDim, bits)
		boundaries = turboquant.ExportBoundaries(headDim, bits)
	}

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
	if !kSkipped {
		if outlierCount > 0 && outlierBits > 0 && outlierCount < headDim {
			oCodebook := turboquant.ExportCodebook(headDim, outlierBits)
			combined := make([]float32, len(codebook)+len(oCodebook))
			copy(combined, codebook)
			copy(combined[len(codebook):], oCodebook)
			codebookT = sharedCtx.FromFloats(combined, len(combined)).(*Tensor)
		} else {
			codebookT = sharedCtx.FromFloats(codebook, len(codebook)).(*Tensor)
		}
	}
	var boundariesT *Tensor
	if !kSkipped {
		boundariesT = sharedCtx.FromFloats(boundaries, len(boundaries)).(*Tensor)
	}

	// Outlier codebook/boundaries at a different bit width. Use headDim as
	// the codebook dim (same reason as regular codebook above).
	// Note: outlierCodebookTensor remains a separate tensor for the
	// encode/dequant kernels (they take a distinct src slot). The fused
	// flash-attention kernel uses the concatenated codebookT above instead.
	var outlierCodebookT, outlierBoundariesT *Tensor
	if !kSkipped && outlierCount > 0 && outlierBits > 0 && outlierCount < headDim {
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

	// WHT sign vectors are allocated per-layer in EnsureLayer using
	// BuildRotation(headDim, rotationSeed ^ uint64(layer+1)). The seed is
	// stored here for later use.

	m := &ggmlTQCompressedK{
		backend:                 b,
		headDim:                 headDim,
		numKVHeads:              numKVHeads,
		bits:                    bits,
		outlierBits:             outlierBits,
		outlierCount:            outlierCount,
		asymmetricPrimary:       asymmetricPrimary,
		kBiasTensors:            make(map[int]*Tensor),
		layerCtxs:               make(map[int]ml.Context),
		packedTensors:           make(map[int]*Tensor),
		scalesTensors:           make(map[int]*Tensor),
		zerosTensors:            make(map[int]*Tensor),
		outlierZerosTensors:     make(map[int]*Tensor),
		outlierPackedTensors:    make(map[int]*Tensor),
		outlierScalesTensors:    make(map[int]*Tensor),
		outlierIndicesTensors:   make(map[int]*Tensor),
		rotationSeed:            rotationSeed,
		rotTensors:              make(map[int]*Tensor),
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
		useROCm:                 scan.SelectedLibrary == "ROCm",
		ccMajor:                 scan.SelectedComputeMajor,
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
//
// In V-only mode (bits == 0) this is a no-op: K isn't compressed and the
// downstream encode/dequant K paths short-circuit on packedTensors[layer]
// being absent.
func (m *ggmlTQCompressedK) EnsureLayer(layer, capacity int) {
	if m.bits == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.packedTensors[layer]; ok {
		return
	}
	packedBytes := m.regularPackedBytes()

	// Size the per-layer context hint by how many tensors we're allocating.
	// Base: 2 (packed + scales). Outlier adds 3. Asymmetric adds 2 (zeros).
	// WHT rotation adds 1 (signs vector). Channel scale adds 1.
	ctxHint := 2
	if m.hasOutliers() {
		ctxHint += 3
	}
	if m.asymmetricPrimary {
		ctxHint += 2
	}
	if m.HasRotation() {
		ctxHint++
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
		bmapWords := m.headDim / 32
		prefixInt16s := (bmapWords + 1) / 2
		outlierIdxStride := m.outlierCount + 2*bmapWords + prefixInt16s
		m.outlierIndicesTensors[layer] = ctx.Zeros(ml.DTypeI16, outlierIdxStride*m.numKVHeads, capacity).(*Tensor)
	}

	// Per-layer WHT sign vector. Seed is XOR'd with layer+1 so each layer
	// gets an independent random rotation (same as HH generated per-layer).
	if m.HasRotation() {
		rot := turboquant.BuildRotation(m.headDim, m.rotationSeed^uint64(layer+1))
		m.rotTensors[layer] = ctx.FromFloats(rot.Signs, m.headDim).(*Tensor)
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
	return m.encodeKInternal(ctx, layer, key, firstCell, nil)
}

// EncodeKAt is the indexed counterpart of EncodeK. Token i is written to
// physical cache slot locs[i]. Used by sliding-window caches where eviction
// fragments free cells.
func (m *ggmlTQCompressedK) EncodeKAt(ctx ml.Context, layer int, key, locs ml.Tensor) ml.Tensor {
	return m.encodeKInternal(ctx, layer, key, 0, locs)
}

func (m *ggmlTQCompressedK) encodeKInternal(ctx ml.Context, layer int, key ml.Tensor, firstCell int, locs ml.Tensor) ml.Tensor {
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
			var kBias ml.Tensor
			if t := m.kBiasTensors[layer]; t != nil {
				kBias = t
			}
			return packed.TQEncodeOutlier(ctx, scales, key, m.rotFor(layer), firstCell, m.boundariesTensor, m.bits,
				oPacked, m.outlierScalesTensors[layer], m.outlierIndicesTensors[layer], m.outlierBoundariesTensor,
				m.outlierBits, m.outlierCount,
				zeros, outlierZeros,
				m.codebookTensor, m.outlierCodebookTensor,
				kBias, locs)
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
	return packed.TQEncode(ctx, scales, key, m.rotFor(layer), firstCell, m.boundariesTensor, m.bits, zeros, kBias, m.codebookTensor, locs)
}

// DequantK creates a GGML_OP_TQ_DEQUANT graph node. encodeResult is the
// view returned by EncodeK (establishes encode→dequant ordering). See
// EncodeK for the single-threaded access contract.
func (m *ggmlTQCompressedK) DequantK(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int) ml.Tensor {
	return m.dequantKInternal(ctx, layer, encodeResult, firstCell, nCells, nil)
}

// DequantKAt is the indexed counterpart of DequantK. Output[i] is populated
// from physical cache slot locs[i]. locs.Shape() = [nCells]i32.
func (m *ggmlTQCompressedK) DequantKAt(ctx ml.Context, layer int, encodeResult, locs ml.Tensor) ml.Tensor {
	if locs == nil {
		return nil
	}
	nCells := locs.Shape()[0]
	return m.dequantKInternal(ctx, layer, encodeResult, 0, nCells, locs)
}

func (m *ggmlTQCompressedK) dequantKInternal(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int, locs ml.Tensor) ml.Tensor {
	scales := m.scalesTensors[layer]
	if scales == nil || encodeResult == nil || nCells <= 0 {
		return nil
	}
	// Wrap encode side-effect buffers with proxy nodes so gallocr does not
	// recycle their memory as scratch before the dequant reads them.
	// (getAsTQTensorInternal applies the same proxyTo pattern for the FA path.)
	pScales := proxyTo(ctx, encodeResult, scales)
	if m.hasOutliers() {
		if oPacked := m.outlierPackedTensors[layer]; oPacked != nil {
			var zeros, outlierZeros ml.Tensor
			if m.asymmetricPrimary {
				zeros = proxyTo(ctx, encodeResult, m.zerosTensors[layer])
				outlierZeros = proxyTo(ctx, encodeResult, m.outlierZerosTensors[layer])
			}
			// Pass WHT signs when rotation is active — the kernel fuses WHT
			// undo into the dequant so the output is plain (unrotated) f16.
			var whtSigns ml.Tensor
			if m.HasRotation() {
				whtSigns = m.rotFor(layer)
			}
			pOPacked := proxyTo(ctx, encodeResult, oPacked)
			pOScales := proxyTo(ctx, encodeResult, m.outlierScalesTensors[layer])
			pOIndices := proxyTo(ctx, encodeResult, m.outlierIndicesTensors[layer])
			k := encodeResult.(*Tensor).TQDequantOutlier(ctx, pScales, m.codebookTensor,
				m.headDim, m.numKVHeads, nCells, firstCell, m.bits,
				pOPacked, pOScales, pOIndices, m.outlierCodebookTensor,
				m.outlierBits, m.outlierCount,
				zeros, outlierZeros,
				whtSigns, locs)
			return k
		}
	}
	// The plain dequant kernel cannot undo the WHT rotation or apply the
	// asymmetric mean offset — only TQDequantOutlier does. This branch is reached
	// only when outliers are disabled (e.g. OLLAMA_TQ_DISABLE_OUTLIERS=1); for a
	// rotated/asymmetric preset the materialized K is approximate. Decode itself
	// uses the fused FA kernel (which handles WHT/zeros inline), so this affects
	// only the non-fused materialize fallback and warmup — warn once rather than
	// silently return wrong values.
	if m.HasRotation() || m.asymmetricPrimary {
		dequantKFallbackWarnOnce.Do(func() {
			slog.Warn("turboquant: DequantK materialize fallback omits WHT-undo/asymmetric-mean when outliers are disabled; K is approximate (fused decode path is unaffected)")
		})
	}
	k := encodeResult.(*Tensor).TQDequant(ctx, pScales, m.codebookTensor,
		m.headDim, m.numKVHeads, nCells, firstCell, m.bits, locs)
	return k
}

// SupportsFusedEncDequantKWHT reports whether DequantKFusedEncode is usable
// for this preset. True when HasRotation, hasOutliers, asymmetricPrimary, and
// no codebooks (EDEN refinement not needed so the kernel can skip it).
// Currently covers tq2k/tq3k/tq4k presets.
func (m *ggmlTQCompressedK) SupportsFusedEncDequantKWHT() bool {
	// Fused encode+decode kernel requires: WHT (HasRotation), outlier split
	// (hasOutliers), and asymmetric primary quantization.  The standard
	// uniform codebooks (always present when !kSkipped) are fine — they're
	// passed to the decode half of the kernel as usual.
	return m.HasRotation() && m.hasOutliers() && m.asymmetricPrimary
}

// PackedTensor returns the per-layer packed K tensor (persistent buffer).
// Used by kvcache as a non-nil sentinel in encodeResults when the fused encode
// path skips the separate TQ_ENCODE op.
func (m *ggmlTQCompressedK) PackedTensor(layer int) ml.Tensor {
	if t := m.packedTensors[layer]; t != nil {
		return t
	}
	return nil
}

// DequantKFusedEncode creates a GGML_OP_TQ_DEQUANT node that encodes k_new into
// the cell at encCell AND decodes all nCells in one Metal dispatch, eliminating
// the TQ_ENCODE→DQ cross-kernel barrier.  The caller must NOT separately call
// EncodeK for this layer/step — the encode is handled inside the fused kernel.
// key is the raw K tensor (f16 or f32) produced by the current K projection.
// encCell is the absolute cache cell index being written this step.
func (m *ggmlTQCompressedK) DequantKFusedEncode(ctx ml.Context, layer int, key ml.Tensor, encCell, firstCell, nCells int) ml.Tensor {
	if m.outlierPackedTensors[layer] == nil || !m.HasRotation() || !m.asymmetricPrimary {
		return nil
	}
	packed := m.packedTensors[layer]
	if packed == nil {
		return nil
	}
	scales := m.scalesTensors[layer]
	if scales == nil || nCells <= 0 {
		return nil
	}

	var zeros, outlierZeros ml.Tensor
	if m.asymmetricPrimary {
		zeros = m.zerosTensors[layer]
		outlierZeros = m.outlierZerosTensors[layer]
	}
	var whtSigns ml.Tensor
	if m.HasRotation() {
		whtSigns = m.rotFor(layer)
	}

	return packed.TQDequantOutlierFusedEnc(ctx,
		scales, m.codebookTensor,
		m.headDim, m.numKVHeads, nCells, firstCell, m.bits,
		m.outlierPackedTensors[layer], m.outlierScalesTensors[layer],
		m.outlierIndicesTensors[layer], m.outlierCodebookTensor,
		m.outlierBits, m.outlierCount,
		zeros, outlierZeros,
		whtSigns, nil,
		key, m.boundariesTensor, m.outlierBoundariesTensor,
		encCell)
}

// SupportsFusedDequantKWHT reports whether DequantK will produce plain
// (unrotated) f16 K via the fused dequant+WHT kernel for this preset.
// True when the preset has rotation, outliers, and asymmetric primary —
// i.e. the ship K-only WHT presets (tq2k/tq3k/tq4k). Enabled on CUDA,
// ROCm, and Metal: each backend dispatches GGML_OP_TQ_DEQUANT with
// op->src[12]=signs to its fused dequant+WHT kernel
// (tq_dequant_k_outlier_wht_kernel on CUDA/ROCm,
// kernel_tq_dequant_outlier_wht on Metal). Callers gate path 2c on this
// to confirm DequantK's output won't need a separate WHT-undo node.
func (m *ggmlTQCompressedK) SupportsFusedDequantKWHT() bool {
	return m.HasRotation() && m.hasOutliers() && m.asymmetricPrimary
}

// SupportsVOnlyFused reports whether the fused inline-V-decode paths are
// expected to beat materialised-V + stock FA on this device. It gates BOTH:
//   - Path 8-fused (V-only presets): packed V decoded inline, raw f16 K.
//   - Path 1.5 (K+V hybrid): DequantK→f16 K + the same inline-V fused kernel.
//
// Both read packed V and decode it inline instead of materialising f16 V, a
// trade that only wins if the fused kernel can hide the packed-V load latency
// with cp.async pipelining rings — and cp.async exists only on Ampere (cc 8.0)+
// (CP_ASYNC_AVAILABLE in common.cuh, gated !GGML_USE_HIP). Without it the loads
// stall and the materialise-V paths (Path 8 / Path 1) win. Hence the cc>=8 gate.
//
// Measured head-to-head, qwen2.5:7b decode, materialise-V vs fused:
//   - Pascal P40 (cc 6.1, no cp.async): materialise wins — tq3 K+V +3.3%/+14.5%
//     (ctx 2048/8192); tq3v V-only +3.2%/tie/+8.3% (ctx 512/2048/8192).
//   - ROCm RX 7600 (HIP → no cp.async): materialise wins — tq3 K+V +15.1%/+38.5%.
//   - Blackwell sm_120 (cc 12, cp.async): fused wins — tq3 K+V +4.3%/+8.1%.
//
// Volta/Turing (cc 7.x, no cp.async) are unmeasured but follow the same rule.
// ROCm/Metal are excluded outright (no cp.async path; HIP never compiles it,
// Metal uses Path 2). OLLAMA_TQ_FORCE_VONLY_FUSED=1 bypasses every gate for
// benchmarking older/excluded hardware.
func (m *ggmlTQCompressedK) SupportsVOnlyFused() bool {
	if forceVOnlyFused {
		return m.vBits > 0 && (m.headDim == 64 || m.headDim == 128 || m.headDim == 256 || m.headDim == 512)
	}
	if m.useROCm || m.preferFusedAttention { // no cp.async on ROCm/Metal; see doc
		return false
	}
	if m.ccMajor < 8 { // cp.async pipelining needs Ampere (cc 8.0)+
		return false
	}
	if m.vBits <= 0 {
		return false
	}
	switch m.headDim {
	case 64, 128, 256, 512:
		return true
	}
	return false
}

// GetAsTQVTensor wraps the packed V buffer for the given layer as a tqVTensor
// for the V-only fused TQ flash-attention path.
func (m *ggmlTQCompressedK) GetAsTQVTensor(ctx ml.Context, layer int, vEncodeResult ml.Tensor, firstCell, nCells int) (ml.Tensor, bool) {
	if !m.SupportsVOnlyFused() {
		return nil, false
	}
	vPacked := m.vPackedTensors[layer]
	vScales := m.vScalesTensors[layer]
	if vPacked == nil || vScales == nil || vEncodeResult == nil || nCells <= 0 {
		return nil, false
	}
	t := &tqVTensor{
		Tensor:    proxyTo(ctx, vEncodeResult, vPacked),
		vScales:   proxyTo(ctx, vEncodeResult, vScales),
		vCodebook: m.vCodebookTensor,
		vBits:     m.vBits,
		headDim:   m.headDim,
		nKVHeads:  m.numKVHeads,
		nCells:    nCells,
		firstCell: firstCell,
		signs:     m.rotFor(layer),
	}
	return t, true
}

// GetAsTQVTensorAt is the indexed variant of GetAsTQVTensor.
// locs is a [nCells]i32 tensor mapping dense cell positions to physical slots.
// The returned tqVTensor carries locs and uses firstCell=0 (physical addressing).
func (m *ggmlTQCompressedK) GetAsTQVTensorAt(ctx ml.Context, layer int, vEncodeResult ml.Tensor, locs ml.Tensor, nCells int) (ml.Tensor, bool) {
	if !m.SupportsVOnlyFused() {
		return nil, false
	}
	vPacked := m.vPackedTensors[layer]
	vScales := m.vScalesTensors[layer]
	if vPacked == nil || vScales == nil || vEncodeResult == nil || nCells <= 0 || locs == nil {
		return nil, false
	}
	t := &tqVTensor{
		Tensor:    proxyTo(ctx, vEncodeResult, vPacked),
		vScales:   proxyTo(ctx, vEncodeResult, vScales),
		vCodebook: m.vCodebookTensor,
		vBits:     m.vBits,
		headDim:   m.headDim,
		nKVHeads:  m.numKVHeads,
		nCells:    nCells,
		firstCell: 0,
		signs:     m.rotFor(layer),
		locs:      locs.(*Tensor),
	}
	return t, true
}

// fusedKernelSupports reports whether the fused TQ flash-attention kernel
// should be used. Supports headDim ∈ {64,128,256,512} and bits ∈ {2,3,4}.
// The inline-decode path is the default on Metal (avoids the f16 K+V
// intermediate) and for all backends at D≥512. DequantKV wins on CUDA/ROCm
// at D≤256.
func (m *ggmlTQCompressedK) fusedKernelSupports() bool {
	// D=64/128/256/512 on all backends.
	switch m.headDim {
	case 64, 128, 256, 512:
	default:
		return false
	}
	if m.bits != 2 && m.bits != 3 && m.bits != 4 {
		return false
	}
	return true
}

// GetAsTQTensor wraps the packed K buffer for the given layer as a tqTensor
// so that ScaledDotProductAttention can dispatch to the fused kernel.
// Returns (nil, false) when the fused path is not supported.
func (m *ggmlTQCompressedK) GetAsTQTensor(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int) (ml.Tensor, bool) {
	return m.getAsTQTensorInternal(ctx, layer, encodeResult, firstCell, nCells, nil)
}

// GetAsTQTensorAt is the indexed counterpart of GetAsTQTensor.
// locs.Shape() = [nCells]i32.
func (m *ggmlTQCompressedK) GetAsTQTensorAt(ctx ml.Context, layer int, encodeResult, locs ml.Tensor) (ml.Tensor, bool) {
	if locs == nil {
		return nil, false
	}
	nCells := locs.Shape()[0]
	return m.getAsTQTensorInternal(ctx, layer, encodeResult, 0, nCells, locs)
}

// proxyTo wraps `dep` (a tensor written by `encodeResult`'s op as a side
// effect) with a TQ_ENCODED_PROXY view so the ggml graph allocator sees
// `dep` as a live tensor with a consumer downstream of encodeResult. Without
// this, gallocr considers `dep` dead immediately after the encode op (since
// nothing reads `dep` via a graph edge) and recycles its memory as scratch
// before downstream consumers (e.g. Phase D's TQ FA op) read it. See
// docs/superpowers/specs/... or the proxy commit message for the full
// diagnosis. nil-in / nil-out for optional tensors.
func proxyTo(ctx ml.Context, encodeResult ml.Tensor, dep *Tensor) *Tensor {
	if dep == nil || encodeResult == nil {
		return dep
	}
	return TQEncodedProxy(ctx, encodeResult, dep).(*Tensor)
}

func (m *ggmlTQCompressedK) getAsTQTensorInternal(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int, locs ml.Tensor) (ml.Tensor, bool) {
	if !m.fusedKernelSupports() {
		return nil, false
	}
	scales := m.scalesTensors[layer]
	if scales == nil || encodeResult == nil || nCells <= 0 {
		return nil, false
	}
	t := &tqTensor{
		Tensor:    encodeResult.(*Tensor),
		scales:    proxyTo(ctx, encodeResult, scales),
		codebook:  m.codebookTensor,
		bits:      m.bits,
		headDim:   m.headDim,
		nKVHeads:  m.numKVHeads,
		nCells:    nCells,
		firstCell: firstCell,
		signs:     m.rotFor(layer),
		// vIsWHT=true when V is WHT-encoded (K+V preset); K-only fused path
		// must apply WHT undo to attnOut so Σwᵢ·WHT(Vᵢ) → Σwᵢ·Vᵢ.
		vIsWHT:       m.HasRotation() && m.vBits > 0,
	}
	if locs != nil {
		t.locs = locs.(*Tensor)
	}
	if m.asymmetricPrimary {
		t.asymmetric = true
		t.zeros = proxyTo(ctx, encodeResult, m.zerosTensors[layer])
	}
	if m.hasOutliers() {
		t.outlierPacked = proxyTo(ctx, encodeResult, m.outlierPackedTensors[layer])
		t.outlierScales = proxyTo(ctx, encodeResult, m.outlierScalesTensors[layer])
		t.outlierIndices = proxyTo(ctx, encodeResult, m.outlierIndicesTensors[layer])
		t.outlierCodebook = m.outlierCodebookTensor
		t.outlierBits = m.outlierBits
		t.outlierCount = m.outlierCount
		t.outlierPackedBytes = m.outlierPackedBytes()
		if m.asymmetricPrimary {
			t.outlierZeros = proxyTo(ctx, encodeResult, m.outlierZerosTensors[layer])
		}
	}
	return t, true
}

// GetAsTQTensorKV wraps both packed K and packed V buffers as a tqTensor for
// the fully fused K+V TQ flash-attention path. Returns (nil, false) when
// fused is not supported or V compression is not enabled for this layer.
func (m *ggmlTQCompressedK) GetAsTQTensorKV(ctx ml.Context, layer int, kEncodeResult, vEncodeResult ml.Tensor, firstCell, nCells int) (ml.Tensor, bool) {
	return m.getAsTQTensorKVInternal(ctx, layer, kEncodeResult, vEncodeResult, firstCell, nCells, nil)
}

// GetAsTQTensorKVAt is the indexed counterpart of GetAsTQTensorKV.
func (m *ggmlTQCompressedK) GetAsTQTensorKVAt(ctx ml.Context, layer int, kEncodeResult, vEncodeResult, locs ml.Tensor) (ml.Tensor, bool) {
	if locs == nil {
		return nil, false
	}
	nCells := locs.Shape()[0]
	return m.getAsTQTensorKVInternal(ctx, layer, kEncodeResult, vEncodeResult, 0, nCells, locs)
}

func (m *ggmlTQCompressedK) getAsTQTensorKVInternal(ctx ml.Context, layer int, kEncodeResult, vEncodeResult ml.Tensor, firstCell, nCells int, locs ml.Tensor) (ml.Tensor, bool) {
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
	// EncodeKV writes both K and V side effects from a SINGLE op (kEncodeResult
	// is its output, vEncodeResult is the V packed buffer written as a side
	// effect). Wrap both K and V scales / zeros / outlier_* via kEncodeResult
	// to establish gallocr graph edges. vEncodeResult itself is already wrapped
	// implicitly via being src[1] of the FA op + kEncodeResult ordering.
	t := &tqTensor{
		Tensor:       kEncodeResult.(*Tensor),
		scales:       proxyTo(ctx, kEncodeResult, kScales),
		codebook:     m.codebookTensor,
		bits:         m.bits,
		headDim:      m.headDim,
		nKVHeads:     m.numKVHeads,
		nCells:       nCells,
		firstCell:    firstCell,
		vPacked:      vEncodeResult.(*Tensor),
		vScales:      proxyTo(ctx, kEncodeResult, vScales),
		vCodebook:    m.vCodebookTensor,
		vBits:        m.vBits,
		signs:        m.rotFor(layer),
	}
	if locs != nil {
		t.locs = locs.(*Tensor)
	}
	if m.asymmetricPrimary {
		t.asymmetric = true
		t.zeros = proxyTo(ctx, kEncodeResult, m.zerosTensors[layer])
	}
	if m.hasOutliers() {
		t.outlierPacked = proxyTo(ctx, kEncodeResult, m.outlierPackedTensors[layer])
		t.outlierScales = proxyTo(ctx, kEncodeResult, m.outlierScalesTensors[layer])
		t.outlierIndices = proxyTo(ctx, kEncodeResult, m.outlierIndicesTensors[layer])
		t.outlierCodebook = m.outlierCodebookTensor
		t.outlierBits = m.outlierBits
		t.outlierCount = m.outlierCount
		t.outlierPackedBytes = m.outlierPackedBytes()
		if m.asymmetricPrimary {
			t.outlierZeros = proxyTo(ctx, kEncodeResult, m.outlierZerosTensors[layer])
		}
	}
	return t, true
}

// rotFor returns the per-layer WHT sign tensor (nil for non-power-of-2 headDim).
// EnsureLayer must have been called for this layer before rotFor is used.
func (m *ggmlTQCompressedK) rotFor(layer int) *Tensor {
	return m.rotTensors[layer]
}

// HasRotation reports whether WHT rotation is active (headDim is a power of 2).
func (m *ggmlTQCompressedK) HasRotation() bool {
	return m.headDim > 0 && m.headDim&(m.headDim-1) == 0
}

// WHTUndo applies the self-inverse Walsh-Hadamard transform to undo the
// rotation stored in DequantK / DequantV output. No-op when HasRotation()
// is false or t is nil. Used by the prefill path to produce unrotated f16
// K/V that stock flash attention can consume directly.
func (m *ggmlTQCompressedK) WHTUndo(ctx ml.Context, layer int, t ml.Tensor) ml.Tensor {
	if t == nil || !m.HasRotation() {
		return t
	}
	signs := m.rotFor(layer)
	if signs == nil {
		return t
	}
	// The WHT kernel accepts both f16 and f32 input; output type matches input.
	return t.(*Tensor).TQApplyWHT(ctx, signs)
}

// EnsureVLayer allocates per-layer V packed and scales tensors on first use.
// Also allocates the per-layer WHT rotation tensor when this is the first
// layer-allocation call to touch this layer (V-only presets bypass
// EnsureLayer entirely, so rotation allocation can't live there alone).
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

	ctxHint := 2
	if m.HasRotation() {
		if _, ok := m.rotTensors[layer]; !ok {
			ctxHint++
		}
	}
	ctx := m.backend.newTQContext(ctxHint)
	// Same per-layer Cache accounting as EnsureLayer — see that comment for why.
	ctx.layer = layer
	packed := ctx.Zeros(ml.DTypeI8, packedBytes*m.numKVHeads, capacity).(*Tensor)
	scales := ctx.Zeros(ml.DTypeF32, m.numKVHeads, capacity).(*Tensor)

	m.vLayerCtxs[layer] = ctx
	m.vPackedTensors[layer] = packed
	m.vScalesTensors[layer] = scales

	// Per-layer WHT signs. EnsureLayer would have created this for the K plane,
	// but in V-only mode EnsureLayer is a no-op. Allocate here when missing.
	if m.HasRotation() {
		if _, ok := m.rotTensors[layer]; !ok {
			rot := turboquant.BuildRotation(m.headDim, m.rotationSeed^uint64(layer+1))
			m.rotTensors[layer] = ctx.FromFloats(rot.Signs, m.headDim).(*Tensor)
		}
	}
}

// EncodeV creates a GGML_OP_TQ_ENCODE_V graph node.
// EnsureVLayer must have been called for this layer before EncodeV.
func (m *ggmlTQCompressedK) EncodeV(ctx ml.Context, layer int, value ml.Tensor, firstCell int) ml.Tensor {
	return m.encodeVInternal(ctx, layer, value, firstCell, nil)
}

// EncodeVAt is the indexed counterpart of EncodeV.
func (m *ggmlTQCompressedK) EncodeVAt(ctx ml.Context, layer int, value, locs ml.Tensor) ml.Tensor {
	return m.encodeVInternal(ctx, layer, value, 0, locs)
}

func (m *ggmlTQCompressedK) encodeVInternal(ctx ml.Context, layer int, value ml.Tensor, firstCell int, locs ml.Tensor) ml.Tensor {
	packed := m.vPackedTensors[layer]
	if packed == nil {
		return nil
	}
	// Pass the K rotation matrix so V outlier energy spreads evenly before
	// quantization. SDPA's post-attention R @ output step undoes the rotation.
	return packed.TQEncodeV(ctx, m.vScalesTensors[layer], value, m.rotFor(layer), firstCell, m.vBoundariesTensor, m.vBits, m.vCodebookTensor, locs)
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
	return m.encodeKVInternal(ctx, layer, key, value, firstCell, nil)
}

// EncodeKVAt is the indexed counterpart of EncodeKV.
func (m *ggmlTQCompressedK) EncodeKVAt(ctx ml.Context, layer int, key, value, locs ml.Tensor) (ml.Tensor, ml.Tensor) {
	return m.encodeKVInternal(ctx, layer, key, value, 0, locs)
}

func (m *ggmlTQCompressedK) encodeKVInternal(ctx ml.Context, layer int, key, value ml.Tensor, firstCell int, locs ml.Tensor) (ml.Tensor, ml.Tensor) {
	// Asymmetric presets write per-cell zeros (mean offset) into a separate tensor
	// during K encode — TQEncodeKV has no zeros output, so it silently drops the
	// mean-centering step. Route asymmetric presets through separate EncodeK + EncodeV.
	if m.hasOutliers() || m.asymmetricPrimary {
		return m.encodeKInternal(ctx, layer, key, firstCell, locs), m.encodeVInternal(ctx, layer, value, firstCell, locs)
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
		m.scalesTensors[layer], key, m.rotFor(layer), m.boundariesTensor,
		vPacked, m.vScalesTensors[layer], value, m.vBoundariesTensor,
		firstCell, m.bits, m.vBits, kBias, m.codebookTensor, m.vCodebookTensor, locs)

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
	return m.dequantVInternal(ctx, layer, encodeResult, firstCell, nCells, nil)
}

// DequantVAt is the indexed counterpart of DequantV.
func (m *ggmlTQCompressedK) DequantVAt(ctx ml.Context, layer int, encodeResult, locs ml.Tensor) ml.Tensor {
	if locs == nil {
		return nil
	}
	nCells := locs.Shape()[0]
	return m.dequantVInternal(ctx, layer, encodeResult, 0, nCells, locs)
}

func (m *ggmlTQCompressedK) dequantVInternal(ctx ml.Context, layer int, encodeResult ml.Tensor, firstCell, nCells int, locs ml.Tensor) ml.Tensor {
	scales := m.vScalesTensors[layer]
	if scales == nil || encodeResult == nil || nCells <= 0 {
		return nil
	}
	return encodeResult.(*Tensor).TQDequant(ctx, scales, m.vCodebookTensor,
		m.headDim, m.numKVHeads, nCells, firstCell, m.vBits, locs)
}

// DequantKV creates a single GGML_OP_TQ_DEQUANT_KV graph node that dequants
// both K and V in one op, halving scheduler overhead vs separate DequantK+DequantV.
// Returns (kTensor, vTensor) as views into the combined output.
//
// When outlier-split is active the K plane is dequanted via the
// regular+outlier overwrite kernel; V is always plain dequant (no V outliers
// in any ship preset).
func (m *ggmlTQCompressedK) DequantKV(ctx ml.Context, layer int, kEncodeResult, vEncodeResult ml.Tensor, firstCell, nCells int) (ml.Tensor, ml.Tensor) {
	return m.dequantKVInternal(ctx, layer, kEncodeResult, vEncodeResult, firstCell, nCells, nil)
}

// DequantKVAt is the indexed counterpart of DequantKV.
func (m *ggmlTQCompressedK) DequantKVAt(ctx ml.Context, layer int, kEncodeResult, vEncodeResult, locs ml.Tensor) (ml.Tensor, ml.Tensor) {
	if locs == nil {
		return nil, nil
	}
	nCells := locs.Shape()[0]
	return m.dequantKVInternal(ctx, layer, kEncodeResult, vEncodeResult, 0, nCells, locs)
}

func (m *ggmlTQCompressedK) dequantKVInternal(ctx ml.Context, layer int, kEncodeResult, vEncodeResult ml.Tensor, firstCell, nCells int, locs ml.Tensor) (ml.Tensor, ml.Tensor) {
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

	// Asymmetric primary quantization stores a per-block mean offset in
	// zerosTensors (allocated in EnsureLayer whenever asymmetricPrimary, so it
	// exists independent of outliers). The dequant kernel needs it to undo the
	// centering, so populate it whenever asymmetric — including the
	// outliers-disabled ablation (e.g. OLLAMA_TQ_DISABLE_OUTLIERS=1).
	if m.asymmetricPrimary {
		kZeros = m.zerosTensors[layer]
	}

	if m.hasOutliers() {
		kOutlierPacked = m.outlierPackedTensors[layer]
		kOutlierScales = m.outlierScalesTensors[layer]
		kOutlierIndices = m.outlierIndicesTensors[layer]
		kOutlierCodebook = m.outlierCodebookTensor
		kOutlierZeros = m.outlierZerosTensors[layer]
		outlierBits = m.outlierBits
		outlierCount = m.outlierCount
		if kOutlierPacked == nil || kOutlierScales == nil || kOutlierIndices == nil || kOutlierCodebook == nil {
			return nil, nil
		}
	}

	var locsT *Tensor
	if locs != nil {
		locsT = locs.(*Tensor)
	}

	combined := TQDequantKV(ctx, m.backend,
		kEncodeResult.(*Tensor), kScales, m.codebookTensor,
		vEncodeResult.(*Tensor), vScales, m.vCodebookTensor,
		m.rotFor(layer), // WHT sign vector [headDim] for V dequant undo
		m.headDim, m.numKVHeads, nCells, firstCell, m.bits, m.vBits,
		kOutlierPacked, kOutlierScales, kOutlierIndices, kOutlierCodebook,
		kZeros, kOutlierZeros,
		outlierBits, outlierCount,
		locsT)

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
	m.vPackedTensors = nil
	m.vScalesTensors = nil
	m.vLayerCtxs = nil
	m.rotTensors = nil
	m.kBiasTensors = nil
	m.sharedCtx = nil
}

// OutlierIndicesSnapshot reads back the GPU outlier-index tensor for layer as a flat
// []int16 of length capacity × outlierCount × numKVHeads. Layout: element [slot, cell]
// is at int16 index slot + cell*outlierCount*numKVHeads, so cell c's indices are
// consecutive at [c*outlierCount .. (c+1)*outlierCount) for numKVHeads=1.
// Must be called after ctx.Compute() has returned (GPU work already completed).
// Returns nil when outliers are not enabled or the layer has not been initialised.
func (m *ggmlTQCompressedK) OutlierIndicesSnapshot(layer int) []int16 {
	t := m.outlierIndicesTensors[layer]
	if t == nil {
		return nil
	}
	raw := t.BackendGetBytes()
	if raw == nil {
		return nil
	}
	result := make([]int16, len(raw)/2)
	for i := range result {
		result[i] = int16(uint16(raw[i*2]) | uint16(raw[i*2+1])<<8)
	}
	return result
}

// TQOutlierCount returns the number of outlier channels per KV head (32 for tq4).
func (m *ggmlTQCompressedK) TQOutlierCount() int { return m.outlierCount }

// TQNumKVHeads returns the number of KV heads.
func (m *ggmlTQCompressedK) TQNumKVHeads() int { return m.numKVHeads }
