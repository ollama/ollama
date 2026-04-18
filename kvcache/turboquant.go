package kvcache

import (
	"log/slog"
	"math"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

type TurboQuantCache struct {
	meta      *Causal
	preset    turboquant.Preset
	isReserve bool

	compressedK ml.TQCompressedKManager

	// phase2Checked ensures GPU encode is activated at most once.
	phase2Checked bool

	headDim    int
	numKVHeads int

	// encodeResults stores per-layer EncodeK result tensors for the current
	// forward pass. DequantK uses them as src[0] to establish the graph
	// dependency (encode before dequant in the ggml scheduler).
	encodeResults map[int]ml.Tensor

	// vEncodeResults stores per-layer EncodeV result tensors for the current
	// forward pass. DequantV uses them to establish the encode→dequant ordering.
	vEncodeResults map[int]ml.Tensor

	// logPathOnce ensures each active Get() path is logged at most once per
	// cache instance (avoids log spam: Get() is called every layer every step).
	logPathOnce [5]sync.Once

	// fusedFallbackEligible gates the inline-decode fused-FA fallback paths
	// (Get paths 2 and 4). Those paths dispatch to a CUDA kernel that is
	// template-instantiated only at D=128, so any model with a larger head
	// dim (gemma3 D=256, gemma4 D=512) must skip them to avoid a kernel-side
	// GGML_ASSERT. The DequantK + stock FA path (Get path 0/1/5) works at
	// any head dim — this gate is specific to the inline-decode variants.
	// Remove once the fused kernels gain D=256/512 template instantiations.
	fusedFallbackEligible bool

	// rotMatrix is the R^T rotation matrix sized for this cache's headDim.
	// Set in activateGPUEncode. Get() sets the backend's tqRotationMatrix to
	// this value per-call (consume-once) right before returning rotated K.
	rotMatrix ml.Tensor

	// vRotMatrix is the R (inverse) matrix used to undo V rotation after
	// attention in K+V presets (tq3/tq2). Nil for K-only presets. Get() sets
	// the backend's tqVRotationMatrix to this value per-call along with
	// rotMatrix so SDPA applies R @ attn_out after flash attention.
	vRotMatrix ml.Tensor

	// rotSetter is the cached type assertion of c.meta.backend onto the TQ
	// rotation setter interface, populated once in activateGPUEncode. nil
	// when the backend doesn't support TQ rotation hooks (the fallback case).
	rotSetter tqRotSetter
}

// tqRotSetter is the backend hook TurboQuantCache uses to arm the per-call
// rotation matrices SDPA consumes. Implemented by ml/backend/ggml.Backend.
type tqRotSetter interface {
	SetTQRotationMatrix(ml.Tensor)
	SetTQVRotationMatrix(ml.Tensor)
}

// isSWACausal reports whether a *Causal has sliding-window attention
// active. Plain Causal caches have swaWindowSize either 0 (before Init
// normalizes the default) or math.MaxInt32 (after); SWA constructors set
// it to the actual window size.
func isSWACausal(c *Causal) bool {
	return c.swaWindowSize > 0 && c.swaWindowSize != math.MaxInt32
}

// WrapWithTurboQuant returns a cache that applies TurboQuant compression to
// global-attention Causal layers and a bool reporting whether any wrapping
// took effect. For a top-level *Causal (non-SWA), it returns a new
// *TurboQuantCache. For a *WrapperCache, it mutates the caches slice in
// place, replacing every non-SWA *Causal sub-cache with a *TurboQuantCache,
// and returns the same *WrapperCache pointer. This enables TQ on SWA models
// like gemma3/gemma4 where the global attention layers dominate KV memory
// at long context. Returns (cache, false) if no eligible sub-caches were
// found.
func WrapWithTurboQuant(cache Cache, preset turboquant.Preset) (Cache, bool) {
	switch c := cache.(type) {
	case *Causal:
		// Reject SWA caches. Plain NewCausalCache leaves swaWindowSize=0
		// until Init() normalizes it to math.MaxInt32; SWA constructors set
		// it to the actual window size. "Plain causal" means the field is
		// either 0 (uninitialized default) or math.MaxInt32 (post-Init).
		if isSWACausal(c) {
			slog.Warn("turboquant: top-level Causal is sliding-window, cannot wrap")
			return cache, false
		}
		return &TurboQuantCache{
			meta:           c,
			preset:         preset,
			encodeResults:  make(map[int]ml.Tensor),
			vEncodeResults: make(map[int]ml.Tensor),
		}, true

	case *WrapperCache:
		// Mutate sub-caches in place: replace every non-SWA *Causal with a
		// *TurboQuantCache wrapping it. SWA sub-caches (SWACache, SWAMemCache)
		// are left untouched — they still allocate f16 K/V as before.
		wrapped := 0
		for i, sub := range c.caches {
			inner, ok := sub.(*Causal)
			if !ok || isSWACausal(inner) {
				continue
			}
			c.caches[i] = &TurboQuantCache{
				meta:           inner,
				preset:         preset,
				encodeResults:  make(map[int]ml.Tensor),
				vEncodeResults: make(map[int]ml.Tensor),
			}
			wrapped++
		}
		if wrapped == 0 {
			slog.Warn("turboquant: no eligible Causal sub-caches in WrapperCache, falling back to unwrapped cache")
			return cache, false
		}
		slog.Info("turboquant: wrapped Causal sub-caches inside WrapperCache",
			"count", wrapped, "preset", preset.Name)
		return cache, true

	default:
		slog.Warn("turboquant: underlying cache is not *Causal or *WrapperCache, falling back to unwrapped cache")
		return cache, false
	}
}

func (c *TurboQuantCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	// K is always compressed; suppress inner Causal from allocating it.
	c.meta.SkipK = true
	// V is compressed only when ValueBits > 0 (tq2/tq3). K-only presets
	// (tq2k/tq3k) have ValueBits=0 and keep V in the f16 Causal cache.
	if c.preset.ValueBits > 0 {
		c.meta.SkipV = true
	}
	c.meta.Init(backend, ml.DTypeF16, maxSequences, capacity, maxBatch)
	slog.Info("turboquant cache initialized", "preset", c.preset.Name,
		"K_bits", c.preset.KeyPrimaryBits, "V_bits", c.preset.ValueBits)
}

func (c *TurboQuantCache) Close() {
	if c.compressedK != nil {
		c.compressedK.Close()
		c.compressedK = nil
	}
	c.meta.Close()
}

func (c *TurboQuantCache) SetLayer(layer int)              { c.meta.SetLayer(layer) }
func (c *TurboQuantCache) SetConfig(config ml.CacheConfig) { c.meta.SetConfig(config) }

func (c *TurboQuantCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	c.isReserve = reserve
	clear(c.encodeResults)
	clear(c.vEncodeResults)
	return c.meta.StartForward(ctx, batch, reserve)
}

func (c *TurboQuantCache) Put(ctx ml.Context, key, value ml.Tensor) {
	// Capture headDim early — needed even during reserve to size placeholders.
	if c.headDim == 0 && key != nil {
		c.headDim = key.Dim(0)
		c.numKVHeads = key.Dim(1)
	}

	if c.isReserve {
		c.meta.Put(ctx, key, value)
		return
	}

	if !c.phase2Checked {
		c.phase2Checked = true
		c.activateGPUEncode()
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		capacity := len(c.meta.cells)

		c.compressedK.EnsureLayer(layer, capacity)
		if c.preset.ValueBits > 0 {
			c.compressedK.EnsureVLayer(layer, capacity)
		}

		// Cells are allocated sequentially; pass the first cell index.
		firstCell := 0
		if len(c.meta.curLocs) > 0 {
			firstCell = int(c.meta.curLocs[0])
		}

		if c.preset.ValueBits > 0 {
			// Combined K+V encode: single GGML op, two back-to-back kernels.
			kResult, vResult := c.compressedK.EncodeKV(ctx, layer, key, value, firstCell)
			if kResult != nil {
				ctx.Forward(kResult)
				c.encodeResults[layer] = kResult
				c.vEncodeResults[layer] = vResult
			}
		} else {
			// K-only presets (tq2k/tq3k): V stays as f16 in the Causal cache.
			encodeResult := c.compressedK.EncodeK(ctx, layer, key, firstCell)
			if encodeResult != nil {
				ctx.Forward(encodeResult)
				c.encodeResults[layer] = encodeResult
			}
		}

		// Inner Causal.Put() tracks cell metadata (positions, masks).
		// SkipK and SkipV suppress the actual K/V tensor writes.
		c.meta.Put(ctx, key, value)
		return
	}

	c.meta.Put(ctx, key, value)
}

func (c *TurboQuantCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	if c.isReserve {
		key, value, mask := c.meta.Get(ctx)
		// SkipK: synthesize zero f16 K placeholder for graph sizing.
		if key == nil && c.headDim > 0 {
			nCells := 1
			if c.meta.curMask != nil {
				nCells = c.meta.curMask.Dim(0)
			}
			key = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
		}
		// SkipV: synthesize zero f16 V placeholder for graph sizing.
		if value == nil && c.headDim > 0 {
			nCells := 1
			if c.meta.curMask != nil {
				nCells = c.meta.curMask.Dim(0)
			}
			value = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
		}
		return key, value, mask
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		firstCell := c.meta.curCellRange.min
		nCells := c.meta.curMask.Dim(0)

		encodeResult := c.encodeResults[layer]
		vEncodeResult := c.vEncodeResults[layer]

		// 0. K-only presets (tq2k/tq3k): DequantK + f16 V from Causal → stock FA.
		//    Skips the fused FA kernel (slower than stock FA on Pascal).
		if c.preset.ValueBits == 0 && encodeResult != nil {
			key := c.compressedK.DequantK(ctx, layer, encodeResult, firstCell, nCells)
			if key != nil {
				c.logPathOnce[0].Do(func() {
					slog.Info("turboquant: using K-only DequantK + f16 V path")
				})
				_, value, mask := c.meta.Get(ctx)
				c.armRotationForNextSDPA()
				return key, value, mask
			}
		}

		// 1. Combined K+V dequant → stock FA (default, fastest path).
		//    Single GGML op dequants both K and V to f16, then stock FA
		//    handles the bandwidth-bound attention with no decode ALU.
		if vEncodeResult != nil {
			key, value := c.compressedK.DequantKV(ctx, layer, encodeResult, vEncodeResult, firstCell, nCells)
			if key != nil && value != nil {
				c.logPathOnce[1].Do(func() {
					slog.Info("turboquant: using combined DequantKV + stock FA path")
				})
				_, _, mask := c.meta.Get(ctx)
				c.armRotationForNextSDPA()
				return key, value, mask
			}
		}

		// 2. K+V fused inline-decode fallback: used only when DequantKV is
		//    unsupported (rare).  The inline-decode kernel is slower than the
		//    DequantKV + stock FA path on all measured hardware, and is only
		//    instantiated at D=128 — skip it for larger head dims.
		if vEncodeResult != nil && c.fusedFallbackEligible {
			if tqkv, ok := c.compressedK.GetAsTQTensorKV(ctx, layer, encodeResult, vEncodeResult, firstCell, nCells); ok {
				c.logPathOnce[2].Do(func() {
					slog.Warn("turboquant: falling back to K+V inline-decode fused kernel (slower)")
				})
				_, _, mask := c.meta.Get(ctx)
				c.armRotationForNextSDPA()
				return tqkv, nil, mask
			}
		}

		// 3. V-only dequant for K-only fused or separate K dequant fallback.
		var value ml.Tensor
		if vEncodeResult != nil {
			value = c.compressedK.DequantV(ctx, layer, vEncodeResult, firstCell, nCells)
		}

		// 4. Try K-only fused: K decoded inline, V is dequanted f16. Gated on
		//    fusedFallbackEligible for the same D=128 reason as path 2.
		if c.fusedFallbackEligible {
			if tqk, ok := c.compressedK.GetAsTQTensor(ctx, layer, encodeResult, firstCell, nCells); ok {
				c.logPathOnce[3].Do(func() {
					slog.Warn("turboquant: falling back to K-only inline-decode fused kernel")
				})
				_, metaValue, mask := c.meta.Get(ctx)
				if value == nil {
					value = metaValue
				}
				c.armRotationForNextSDPA()
				return tqk, value, mask
			}
		}

		// 5. Separate K + V dequant fallback (last resort).
		c.logPathOnce[4].Do(func() {
			slog.Warn("turboquant: falling back to separate K + V dequant path")
		})
		key := c.compressedK.DequantK(ctx, layer, encodeResult, firstCell, nCells)
		_, metaValue, mask := c.meta.Get(ctx)
		if value == nil {
			value = metaValue
		}
		c.armRotationForNextSDPA()
		return key, value, mask
	}

	return c.meta.Get(ctx)
}

// armRotationForNextSDPA sets the backend's tqRotationMatrix (and
// tqVRotationMatrix for K+V presets) so the next SDPA call — which happens
// immediately after this Get returns — rotates Q to match the TQ-rotated K
// and applies the V rotation undo after attention. SDPA reads-and-clears the
// flags, so non-TQ sub-cache layers in a WrapperCache (e.g. gemma3 SWA
// layers) are unaffected.
func (c *TurboQuantCache) armRotationForNextSDPA() {
	if c.rotMatrix == nil || c.rotSetter == nil {
		return
	}
	c.rotSetter.SetTQRotationMatrix(c.rotMatrix)
	// For K+V presets (tq3/tq2), also arm the V rotation undo on the next
	// SDPA call. For K-only presets (tq3k/tq2k), c.vRotMatrix is nil so the
	// V rotation is not armed and SDPA's consumed vRot stays nil.
	if c.vRotMatrix != nil {
		c.rotSetter.SetTQVRotationMatrix(c.vRotMatrix)
	}
}

// activateGPUEncode initialises the TQ compressed-K manager if the backend
// supports it and re-enables Q rotation (stored K is in rotated space).
func (c *TurboQuantCache) activateGPUEncode() {
	// fallbackToF16 un-skips K/V on the inner Causal so subsequent Put/Get
	// on this cache store and read f16 tensors like an ordinary non-TQ cache.
	// Init() sets SkipK/SkipV unconditionally, so any failure to activate GPU
	// encode must reverse those flags before the current Put() continues into
	// c.meta.Put — otherwise SDPA receives a nil K/V from Get and segfaults.
	// K allocation in Causal.Put is lazy (keyed on presence of c.keys[layer])
	// so flipping the flag on the first Put is sufficient.
	fallbackToF16 := func() {
		c.meta.SkipK = false
		c.meta.SkipV = false
	}

	tqb, ok := c.meta.backend.(ml.TQCompressedKBackend)
	if !ok {
		fallbackToF16()
		return
	}

	// Pass the preset's outlier config so the manager can enable post-rotation
	// outlier split on the GPU encode path. This is required for correct
	// output on models with learned K bias (e.g. qwen2 family) and matches
	// the TurboQuant paper's validated experimental setup.
	mgr := tqb.NewTQCompressedKManager(
		c.headDim, c.numKVHeads, c.preset.KeyPrimaryBits, c.preset.RotationSeed,
		c.preset.ValueBits, c.preset.OutlierBits, c.preset.OutlierCount,
	)
	if mgr == nil {
		slog.Info("turboquant: GPU encode not available, using f16 K fallback")
		fallbackToF16()
		return
	}
	c.compressedK = mgr

	// The inline-decode fused-FA fallback paths (Get paths 2 and 4) dispatch
	// to a CUDA kernel template instantiated only at D=128. Models with a
	// larger head dim (e.g. gemma4 global layers at headDim=512) must skip
	// those fallbacks to avoid a kernel-side GGML_ASSERT; path 5 (separate
	// K+V dequant) handles them correctly.
	c.fusedFallbackEligible = (c.headDim == 128)
	if !c.fusedFallbackEligible {
		slog.Info("turboquant: inline-decode fused-FA fallback paths disabled",
			"reason", "headDim != 128", "headDim", c.headDim)
	}

	// Cache the rotation matrices and the backend's rotation-setter hook on
	// TurboQuantCache so Get() can arm them per-call without re-running a
	// type assertion every layer. We do NOT set them at activate time — a
	// sticky backend-global rotation would corrupt attention on unwrapped
	// SWA layers in mixed-head-dim models like gemma3.
	c.rotMatrix = mgr.RotationMatrix(nil, 0)
	if rs, ok := c.meta.backend.(tqRotSetter); ok {
		c.rotSetter = rs
	}

	type vRotFusedSetter interface {
		SetTQVRotFusedInDequant(bool)
	}
	type vRotProvider interface {
		RotationMatrixR() ml.Tensor
	}
	if c.preset.ValueBits > 0 {
		if vp, ok := mgr.(vRotProvider); ok {
			c.vRotMatrix = vp.RotationMatrixR()
		}
		// DequantKV outputs R^T @ v (still rotated). SDPA applies the rotation
		// undo as R @ attn_out via mulmat, which is dramatically faster than
		// the per-cell matmul of the fused dequant kernel.
		if rs, ok := c.meta.backend.(vRotFusedSetter); ok {
			rs.SetTQVRotFusedInDequant(false)
		}
	}

	slog.Info("turboquant: GPU-native encode active",
		"headDim", c.headDim, "numKVHeads", c.numKVHeads,
		"K_bits", c.preset.KeyPrimaryBits, "V_bits", c.preset.ValueBits)
}

func (c *TurboQuantCache) CopyPrefix(srcSeq, dstSeq int, prefixLen int32) {
	c.meta.CopyPrefix(srcSeq, dstSeq, prefixLen)
}

func (c *TurboQuantCache) CanResume(seq int, pos int32) bool {
	return c.meta.CanResume(seq, pos)
}

// Remove returns ErrNotSupported for partial evictions when GPU-compressed K is
// active. The compressed buffer cannot be RoPE-shifted in-place; callers fall
// back to full reprocessing via ErrReprocessInputs.
func (c *TurboQuantCache) Remove(seq int, beginIndex, endIndex int32) error {
	if c.compressedK != nil && beginIndex != 0 {
		return ErrNotSupported
	}
	return c.meta.Remove(seq, beginIndex, endIndex)
}

func (c *TurboQuantCache) SetCausal(ctx ml.Context, opts CausalOptions) {
	c.meta.SetCausal(ctx, opts)
}

func PresetFromDType(dtype ml.DType) (turboquant.Preset, bool) {
	switch dtype {
	case ml.DTypeTQ2:
		return turboquant.PresetTQ2, true
	case ml.DTypeTQ3:
		return turboquant.PresetTQ3, true
	case ml.DTypeTQ3K:
		return turboquant.PresetTQ3K, true
	case ml.DTypeTQ2K:
		return turboquant.PresetTQ2K, true
	default:
		return turboquant.Preset{}, false
	}
}

var _ Cache = (*TurboQuantCache)(nil)

// Note: TurboQuantCache intentionally does NOT implement CheckpointCache.
// CheckpointCache is for recurrent caches that need special per-sequence
// state restoration. The inner *Causal cache supports plain CanResume, which
// is the right semantics for TQ-wrapped caches too — the runner will fall
// through to the CanResume branch when TurboQuantCache is in use (see
// runner/ollamarunner/cache.go:163-170). An earlier implementation had a
// stub PrepareRestore that always returned (0, false), which forced a full
// prompt reprocess on every resume for long-context gemma/llama runs.
