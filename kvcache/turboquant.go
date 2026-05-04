package kvcache

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

// forceFused opts back into the K+V fused inline-decode path (path 2). The
// default is the combined DequantKV + stock FA path (path 1), which is
//   - faster on Pascal P40 (4.4× decode, 8.95× prefill measured),
//   - the only correct option on RDNA3/HIP — the fused kernel there is
//     producing wrong-but-low-PPL output (PPL below f16, which lossy
//     quantization cannot achieve), and
//   - functionally equivalent to path 2 on Pascal CUDA (PPL preserved
//     bit-for-bit at ctx=16384 across llama3.1:8b, llama3.2:3b, qwen2.5:7b).
//
// OLLAMA_TQ_FORCE_FUSED=1 re-engages the fused kernel for benchmarking and
// for resuming the path-2 correctness investigation on HIP. The kernel stays
// in tree until that bug is rooted.
var forceFused = os.Getenv("OLLAMA_TQ_FORCE_FUSED") != ""

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

	// fusedFallbackEligible gates the inline-decode fused-FA fallback paths (D=64, 128, 256)
	// (Get paths 2 and 4). The CUDA fused kernel is template-instantiated at
	// D=64, 128, 256 — covering llama3.2 (64), llama3.1/qwen2.5 (128), and
	// Gemma (256). Models with D=512 or other unsupported dims fall back to the
	// DequantK + stock FA path (Get paths 0/1/5), which works at any head dim.
	fusedFallbackEligible bool

	// preferFusedAttn is true on Metal. At long context, DequantKV + stock FA
	// writes a full f16 intermediate buffer (doubling KV bandwidth) before
	// attention reads it. The fused kernel (kernel_tq_fattn_vec_packed) reads
	// packed K+V directly and skips the intermediate write — dramatically
	// faster on Metal for *decode* (Q=1). On CUDA the DequantKV path is
	// preferred because the intermediate buffer stays in L2 and stock flash
	// attention is highly tuned.
	//
	// For prefill (Q>1, see curQueryLen) DequantKV wins on every backend
	// because it decodes each packed cell once then runs stock FA with full
	// batch amortisation, whereas the fused kernel re-decodes each cell per
	// Q-token. Get() uses the combination of preferFusedAttn and curQueryLen
	// to route prefill vs decode independently.
	preferFusedAttn bool

	// curQueryLen is the number of query tokens in the current forward pass,
	// captured at StartForward from len(batch.Positions). curQueryLen==1
	// means decode; curQueryLen>1 means prefill (or a batched prompt chunk).
	curQueryLen int

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

	// pendingKBiases buffers K projection biases set by SetLayerKBias before
	// activateGPUEncode has initialised compressedK. Applied to compressedK
	// once the manager is created.
	pendingKBiases map[int]ml.Tensor

	// reserveSkipVPlaceholder, when true, suppresses the SkipV synthesis block
	// in the reserve graph for the current Get() call. Set when the K reserve
	// branch placed a K+V fused tensor (which carries V internally as vPacked,
	// so the inference-time value return is nil). The reserve graph must
	// mirror that — synthesising a Zeros f16 V here would re-introduce the
	// scratch buffer the K+V fused path was meant to eliminate. Read-and-cleared
	// once per Get() call.
	reserveSkipVPlaceholder bool
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

// AttentionKVWrapper is implemented by caches that embed *Recurrent and
// expose the attention half of a hybrid (SSM/recurrent + attention) cache.
// WrapWithTurboQuant uses it to inject TurboQuant compression into the
// attention KV path without disturbing conv/recurrent state buffers.
// *kvcache.Recurrent implements this interface, so any model HybridCache that
// embeds *Recurrent satisfies it automatically via Go method promotion.
type AttentionKVWrapper interface {
	AttentionKV() *Causal
	SetAttentionKV(Cache)
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

	case AttentionKVWrapper:
		inner := c.AttentionKV()
		if inner == nil {
			slog.Warn("turboquant: hybrid cache inner kv is not *Causal (already wrapped?), leaving as-is")
			return cache, false
		}
		if isSWACausal(inner) {
			slog.Warn("turboquant: hybrid cache inner *Causal is sliding-window, cannot wrap")
			return cache, false
		}
		c.SetAttentionKV(&TurboQuantCache{
			meta:           inner,
			preset:         preset,
			encodeResults:  make(map[int]ml.Tensor),
			vEncodeResults: make(map[int]ml.Tensor),
		})
		slog.Info("turboquant: wrapped attention KV in hybrid recurrent cache",
			"preset", preset.Name)
		return cache, true

	default:
		slog.Warn("turboquant: underlying cache is not *Causal or *WrapperCache, falling back to unwrapped cache")
		return cache, false
	}
}

func (c *TurboQuantCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	// K is always compressed; suppress inner Causal from allocating it.
	c.meta.SkipK = true
	// V is compressed when ValueBits > 0; skip the inner Causal V allocation.
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

// FusedEligible reports whether the inline-decode fused kernel path is active.
// Only valid after the first Put() call (which triggers activateGPUEncode).
// Returns false if headDim is not in the supported set — DequantK slow path is active.
func (c *TurboQuantCache) FusedEligible() bool { return c.fusedFallbackEligible }

// SetLayerKBias passes the K projection bias tensor for the given layer to the
// TQ compression manager. Called once per layer at model init for architectures
// that include a bias on the K projection (e.g. Qwen2). The bias is subtracted
// from K before rotation in the encoder; attention correctness is preserved
// because a constant shift in K cancels out in softmax.
func (c *TurboQuantCache) SetLayerKBias(layer int, bias ml.Tensor) {
	if bias == nil {
		return
	}
	if c.compressedK == nil {
		// activateGPUEncode hasn't run yet — buffer for later application.
		if c.pendingKBiases == nil {
			c.pendingKBiases = make(map[int]ml.Tensor)
		}
		c.pendingKBiases[layer] = bias
		return
	}
	type kBiasSetter interface {
		SetLayerKBias(layer int, bias ml.Tensor)
	}
	if kbs, ok := c.compressedK.(kBiasSetter); ok {
		kbs.SetLayerKBias(layer, bias)
	}
}

func (c *TurboQuantCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	c.isReserve = reserve
	c.curQueryLen = len(batch.Positions)
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

	// Activate the GPU encode path on the first Put (reserve or not) once we
	// know headDim/numKVHeads. Doing this during reserve lets EnsureLayer run
	// on the reserve pass, which books TQ's per-layer persistent K/V buffers
	// into btDeviceMemory.Cache[layer] (via newTensor's ctx.layer accounting
	// in EnsureLayer). Without this, the scheduler's fit/alloc probe sees a
	// 0-byte Cache for tq2/tq3 and only the f16 V half for tq2k/tq3k, which
	// is wrong for both headline-footprint reporting and long-context fit math.
	if !c.phase2Checked && c.headDim > 0 {
		c.phase2Checked = true
		c.activateGPUEncode()
	}

	if c.isReserve {
		c.meta.Put(ctx, key, value)
		// Eagerly allocate TQ/i4 persistent buffers during reserve so the
		// scheduler's per-layer Cache totals reflect the real post-compression
		// footprint. Also create the encode graph node so the reserve graph has
		// the same K-branch structure as the inference graph. Without the encode
		// node, gallocr's live-range analysis sees a truncated graph and
		// over-allocates scratch (measured: +222 MiB for asym+outliers presets).
		if c.compressedK != nil {
			layer := c.meta.curLayer
			capacity := len(c.meta.cells)
			c.compressedK.EnsureLayer(layer, capacity)
			if c.preset.ValueBits > 0 {
				c.compressedK.EnsureVLayer(layer, capacity)
				kResult, vResult := c.compressedK.EncodeKV(ctx, layer, key, value, 0)
				if kResult != nil {
					ctx.Forward(kResult)
					c.encodeResults[layer] = kResult
					c.vEncodeResults[layer] = vResult
				}
			} else {
				encodeResult := c.compressedK.EncodeK(ctx, layer, key, 0)
				if encodeResult != nil {
					ctx.Forward(encodeResult)
					c.encodeResults[layer] = encodeResult
				}
			}
		}
		return
	}

	// Compute firstCell from contiguous curLocs (same invariant for both paths).
	firstCell := 0
	if len(c.meta.curLocs) > 0 {
		firstCell = c.meta.curLocs[0]
		for i := 1; i < len(c.meta.curLocs); i++ {
			if c.meta.curLocs[i] != firstCell+i {
				panic(fmt.Sprintf("turboquant: non-contiguous cache slots %v — findLocs invariant violated", c.meta.curLocs))
			}
		}
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		capacity := len(c.meta.cells)

		c.compressedK.EnsureLayer(layer, capacity)
		if c.preset.ValueBits > 0 {
			c.compressedK.EnsureVLayer(layer, capacity)
		}

		// The TQ encode kernels (CUDA + Metal) write a contiguous run of
		// cells starting at firstCell. Causal.findLocs guarantees a
		// contiguous run when SkipK/SkipV is set (otherwise it returns
		// ErrKvCacheFull before we reach here); this loop is a defensive
		// invariant check that should never fire in practice.
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
		// SkipK: synthesize K for graph sizing, matching the node type that the
		// inference path produces so gallocr pre-reserves the right scratch size.
		// Each branch mirrors exactly the inference routing in Get() below.
		if key == nil && c.headDim > 0 {
			nCells := 1
			if c.meta.curMask != nil {
				nCells = c.meta.curMask.Dim(0)
			}
			if c.compressedK != nil {
				layer := c.meta.curLayer
				enc := c.encodeResults[layer]
				switch {
				case c.preset.ValueBits > 0:
					// All tq* K+V presets: prefer fused K+V (no f16 scratch).
					// GetAsTQTensorKV succeeds for non-outlier and K+V-outlier presets;
					// returns false for K-only-outlier (blocked in fusedKernelSupports).
					// Falls back to DequantKV only when fused is unavailable (D!=128).
					vEnc := c.vEncodeResults[layer]
					if enc != nil && vEnc != nil {
						if gpuKV, ok := c.compressedK.GetAsTQTensorKV(ctx, layer, enc, vEnc, 0, nCells); ok {
							key = gpuKV
							// V carried inline — suppress f16 Zeros fallback in SkipV block.
							c.reserveSkipVPlaceholder = true
						}
						if key == nil {
							k, v := c.compressedK.DequantKV(ctx, layer, enc, vEnc, 0, nCells)
							if k != nil {
								key = k
							}
							if v != nil {
								value = v
							}
						}
						if key == nil {
							key = c.compressedK.DequantK(ctx, layer, enc, 0, nCells)
						}
						if value == nil {
							value = c.compressedK.DequantV(ctx, layer, vEnc, 0, nCells)
						}
					}
				default:
					// K-only (tq2k/tq3k/tq4k, with or without asymmetric): prefer
					// fused K-only path (no f16 scratch); fall back to DequantK
					// when fused is unavailable.
					if enc != nil {
						if gpuKey, ok := c.compressedK.GetAsTQTensor(ctx, layer, enc, 0, nCells); ok {
							key = gpuKey
						} else {
							key = c.compressedK.DequantK(ctx, layer, enc, 0, nCells)
						}
					}
				}
			}
			if key == nil {
				key = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
			}
		}
		// SkipV: synthesize V placeholder when the K branch did not place a
		// K+V fused tensor. For K+V TQ presets value was already set above by
		// DequantKV, so this block is a no-op for those. For K-only presets
		// value comes from Causal. If the K block placed a K+V fused tensor
		// (value=nil, V carried inline), skip — otherwise we'd reintroduce
		// the f16 V scratch.
		skipVPlaceholder := c.reserveSkipVPlaceholder
		c.reserveSkipVPlaceholder = false
		if !skipVPlaceholder && value == nil && c.headDim > 0 {
			nCells := 1
			if c.meta.curMask != nil {
				nCells = c.meta.curMask.Dim(0)
			}
			if value == nil {
				value = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
			}
		}
		return key, value, mask
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		firstCell := c.meta.curCellRange.min
		nCells := c.meta.curMask.Dim(0)

		encodeResult := c.encodeResults[layer]
		vEncodeResult := c.vEncodeResults[layer]

		// 0. K-only presets without asymmetric primary (tq2k/tq3k/tq4k under
		//    OLLAMA_TQ_DISABLE_ASYMMETRIC=1): fused K-only (no scratch).
		//    Falls back to DequantK only when fused is unavailable (D!=128).
		if c.preset.ValueBits == 0 && !c.preset.AsymmetricPrimary && encodeResult != nil {
			if c.fusedFallbackEligible {
				if tqk, ok := c.compressedK.GetAsTQTensor(ctx, layer, encodeResult, firstCell, nCells); ok {
					c.logPathOnce[0].Do(func() {
						slog.Info("turboquant: using K-only fused path")
					})
					_, value, mask := c.meta.Get(ctx)
					c.armRotationForNextSDPA()
					return tqk, value, mask
				}
			}
			key := c.compressedK.DequantK(ctx, layer, encodeResult, firstCell, nCells)
			if key != nil {
				c.logPathOnce[0].Do(func() {
					slog.Info("turboquant: using K-only DequantK + f16 V path (fused unavailable)")
				})
				_, value, mask := c.meta.Get(ctx)
				c.armRotationForNextSDPA()
				return key, value, mask
			}
		}

		// 1. Combined K+V dequant → stock FA. Default for K+V outlier presets
		//    on all backends. Outlier-aware: K plane uses the regular+outlier
		//    overwrite kernel when outliers are present, V plane is plain
		//    dequant. Single ggml op per layer; output is one [headDim,
		//    numKVHeads, nCells, 2] f16 tensor split into K and V views.
		//
		//    Replaces the K+V fused inline-decode path (path 2 below) as the
		//    default: faster on Pascal P40 (4.4× decode, 8.95× prefill measured
		//    on llama3.1:8b ctx=16384) and correctness fix on RDNA3 — the HIP
		//    build of the fused kernel produces wrong-but-low-PPL output (PPL
		//    < f16, which lossy quant cannot achieve). PPL is preserved
		//    bit-for-bit on Pascal CUDA between path 2 and path 1 here.
		if vEncodeResult != nil && !forceFused {
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

		// 2. K+V fused inline-decode: opt-in via OLLAMA_TQ_FORCE_FUSED=1 only.
		//    Kept in tree for benchmarking and for resuming the HIP correctness
		//    investigation; the path-1 fallback above is the default since
		//    the fused kernel is slower on Pascal and broken on RDNA3.
		if vEncodeResult != nil && c.fusedFallbackEligible && forceFused {
			if tqkv, ok := c.compressedK.GetAsTQTensorKV(ctx, layer, encodeResult, vEncodeResult, firstCell, nCells); ok {
				c.logPathOnce[2].Do(func() {
					slog.Warn("turboquant: using K+V fused inline-decode path (forced via OLLAMA_TQ_FORCE_FUSED)")
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
		//    fusedFallbackEligible for the same D=128 reason as path 2. This
		//    path uses the V_PACKED=false specialisation of the TQ FA kernel
		//    (V is f16); the V_PACKED=true specialisation that path 2 used is
		//    where the HIP correctness regression lives, so K-only is unaffected.
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

	// QJL residual is only algorithmically active when the preset also uses
	// outlier split (see turboquant.encodeVectorWithOutliers). The 6 ship
	// presets all have QJLRowsDivisor=0 (no QJL); the runtime computation
	// stays here so test-only Preset values that opt into QJL still work
	// through this path.
	qjlRows := 0
	if c.preset.HasOutlierSplit() {
		qjlRows = c.preset.KeyQJLRows(c.headDim)
	}
	// The user requested a measurement-grade preset if either asymmetric
	// primary quantization or the QJL residual sketch is on. If we can't
	// activate those on the GPU we produce exactly f16 output and any PPL
	// measurement is a lie. Log that failure at ERROR with a distinctive
	// marker so smoke tests and humans can both notice instead of silently
	// reading bit-identical-to-f16 numbers and calling them success.
	measurementRequested := c.preset.AsymmetricPrimary || qjlRows > 0
	logActivation := func(gpuActive bool, path string) {
		level := slog.LevelInfo
		if measurementRequested && !gpuActive {
			level = slog.LevelError
		}
		slog.Log(context.TODO(), level, "TQ_ACTIVATION",
			"preset", c.preset.Name,
			"asymmetric_requested", c.preset.AsymmetricPrimary,
			"qjl_rows_requested", qjlRows,
			"outlier_count", c.preset.OutlierCount,
			"gpu_active", gpuActive,
			"measurement_requested", measurementRequested,
			"path", path,
		)
	}

	// Architectural compatibility gate: the TQ encode kernel and fused fattn
	// templates only handle headDim ∈ {64, 128, 256}. Models outside this set
	// (e.g. glm-4.7 / DeepSeek-MLA at headDim=576) must skip TQ entirely and
	// stay on f16 — the slow DequantK fallback also relies on encoder kernels
	// that assume those dims. Stock ggml's quantized KV (Q4_0/Q8_0) already
	// has the same restriction; this matches that behavior.
	if c.headDim != 64 && c.headDim != 128 && c.headDim != 256 {
		slog.Warn("turboquant: headDim not supported, falling back to f16 KV cache",
			"preset", c.preset.Name,
			"headDim", c.headDim,
			"supported", "{64, 128, 256}")
		logActivation(false, "f16-fallback-headdim-unsupported")
		fallbackToF16()
		return
	}

	tqb, ok := c.meta.backend.(ml.TQCompressedKBackend)
	if !ok {
		logActivation(false, "f16-fallback-no-tq-backend")
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
		c.preset.AsymmetricPrimary, qjlRows,
	)
	if mgr == nil {
		if measurementRequested {
			slog.Error("turboquant: GPU manager unavailable — SILENT F16 FALLBACK ACTIVE. Run on CUDA, or set OLLAMA_TQ_DISABLE_ASYMMETRIC=1 / OLLAMA_TQ_DISABLE_OUTLIERS=1 to drop to a backend-supported configuration.",
				"preset", c.preset.Name,
				"asymmetric", c.preset.AsymmetricPrimary,
				"qjl_rows", qjlRows)
		} else {
			slog.Info("turboquant: GPU encode not available, using f16 K fallback")
		}
		logActivation(false, "f16-fallback-mgr-nil")
		fallbackToF16()
		return
	}
	c.compressedK = mgr
	// Apply any K projection biases buffered before the manager was ready.
	if len(c.pendingKBiases) > 0 {
		type kBiasSetter interface {
			SetLayerKBias(layer int, bias ml.Tensor)
		}
		if kbs, ok := mgr.(kBiasSetter); ok {
			for layer, bias := range c.pendingKBiases {
				kbs.SetLayerKBias(layer, bias)
			}
		}
		c.pendingKBiases = nil
	}
	logActivation(true, "gpu-native")

	type fusedAttnPreferrer interface {
		PreferFusedAttention() bool
	}
	if ff, ok := mgr.(fusedAttnPreferrer); ok {
		c.preferFusedAttn = ff.PreferFusedAttention()
		if c.preferFusedAttn {
			slog.Info("turboquant: preferring fused flash-attention path (Metal: avoids f16 intermediate buffer)")
		}
	}

	// The headDim ∈ {64, 128, 256} gate at the top of this function ensures
	// we only get here on supported dims, so the fused-FA fallback paths
	// (Get paths 2 and 4) are always eligible.
	c.fusedFallbackEligible = true

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

// Remove returns ErrNotSupported for any partial eviction when GPU-compressed
// K is active: the compressed buffer cannot be RoPE-shifted in-place, and the
// inner Causal.Remove path silently skips its shiftFn loop because SkipK
// leaves c.keys empty — survivors would keep their old RoPE embeddings while
// their positions are decremented, producing wrong attention scores. Only a
// full-sequence eviction (Remove(seq, 0, MaxInt32)) avoids the shift path;
// other callers must fall back to full reprocessing via ErrReprocessInputs.
func (c *TurboQuantCache) Remove(seq int, beginIndex, endIndex int32) error {
	if c.compressedK != nil && !(beginIndex == 0 && endIndex == math.MaxInt32) {
		return ErrNotSupported
	}
	return c.meta.Remove(seq, beginIndex, endIndex)
}

func (c *TurboQuantCache) SetCausal(ctx ml.Context, opts CausalOptions) {
	c.meta.SetCausal(ctx, opts)
}

// PresetFromDType resolves a runtime KV-cache DType to a turboquant.Preset.
// Returned presets pass through ApplyEnvOverrides so OLLAMA_TQ_DISABLE_*
// takes effect in the cache encode/decode path.
func PresetFromDType(dtype ml.DType) (turboquant.Preset, bool) {
	switch dtype {
	case ml.DTypeTQ2:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ2), true
	case ml.DTypeTQ3:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ3), true
	case ml.DTypeTQ3K:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ3K), true
	case ml.DTypeTQ2K:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ2K), true
	case ml.DTypeTQ4:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ4), true
	case ml.DTypeTQ4K:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ4K), true
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
