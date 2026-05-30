package kvcache

import (
	"context"
	"log/slog"
	"math"
	"os"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

// forceFused opts the K+V CUDA path back into fused inline-decode (path 2).
// The default for K+V on CUDA is DequantKV + stock FA (path 1): faster on
// Pascal (4.4× decode, 8.95× prefill) and bit-identical in PPL across
// llama3.1/3.2 and qwen2.5 at long context. OLLAMA_TQ_FORCE_FUSED=1 is for
// benchmarking and HIP-portability work.
var forceFused = os.Getenv("OLLAMA_TQ_FORCE_FUSED") != ""

// forceDeQuantKV disables fused K+V FA on ALL backends (including Metal) so
// Path-1 (DequantKV → stock FA) can be benchmarked in isolation. The inline
// OLLAMA_TQ_FORCE_DEQUANT_KV check in activateGPUEncode only suppresses the
// D>=512 promotion on non-Metal; this var extends the override into the
// routeGet decision, where it preempts preferFusedAttn=true (Metal) as well.
var forceDeQuantKV = os.Getenv("OLLAMA_TQ_FORCE_DEQUANT_KV") != ""

// forceDeQuantV opts V-only presets out of Path 8-fused (V inline-decode) and
// back onto Path 8 (DequantV + WHTUndo + stock FA) on CUDA/ROCm. Set
// OLLAMA_TQ_FORCE_DEQUANT_V=1 to benchmark the two paths side-by-side.
var forceDeQuantV = os.Getenv("OLLAMA_TQ_FORCE_DEQUANT_V") != ""

// forceVOnlyFused overrides SupportsVOnlyFused() and the !indexed gate so
// Path 8-fused can be benchmarked on backends that opt out by default (ROCm,
// Metal) and on indexed/SWA caches. Does not bypass the decode-only
// (curQueryLen==1) gate — prefill extension requires kernel changes.
// OLLAMA_TQ_FORCE_VONLY_FUSED=1 to enable.
var forceVOnlyFused = os.Getenv("OLLAMA_TQ_FORCE_VONLY_FUSED") != ""

// disableFusedDequantK opts K-only TQ presets out of Path 2c (fused
// DequantK+WHT → stock FA op) and onto Path 4 (inline-decode → TQ FA op).
// Phase D's copy-and-patch dispatcher sits on the TQ FA op, so without this
// override tq*k presets never reach Phase D on Turing+ — Path 2c claims them
// first because SupportsFusedDequantKWHT() returns true. Set this to surface
// tq*k through the TQ FA op so Phase D's wedge can engage for verification.
// Not for production use: tq*k is a verification preset only; ship targets
// are tq (K+V) and tq*v (V-only).
var disableFusedDequantK = os.Getenv("OLLAMA_TQ_DISABLE_FUSED_DEQUANT_K") != ""

// disableFusedEncode is an ablation gate: when set, skip the fused encode+decode
// path (kernel_tq_enc_dq_wht) and fall back to the separate TQ_ENCODE+DequantK
// two-op sequence. Set OLLAMA_TQ_DISABLE_FUSED_ENCODE=1 to measure the barrier
// elimination contribution of the fused kernel.
var disableFusedEncode = os.Getenv("OLLAMA_TQ_DISABLE_FUSED_ENCODE") != ""

type TurboQuantCache struct {
	meta      *Causal
	preset    turboquant.Preset
	isReserve bool

	compressedK ml.TQCompressedKManager

	// phase2Checked ensures GPU encode is activated at most once.
	phase2Checked bool

	headDim    int
	numKVHeads int

	// CONTRACT for encodeResults / vEncodeResults: these maps are written by
	// Put and read by Get on a SINGLE goroutine per cache instance. This
	// matches the runner's per-step model-evaluation loop (Init → Put-per-layer
	// → Get-per-layer → Compute → Reset, all serial), and mirrors the same
	// goroutine-affinity contract that the inner *Causal cache already
	// assumes for its own per-layer tensor maps (c.keys, c.values, c.ctxs).
	// Violating this contract — e.g. dispatching layers across goroutines, or
	// overlapping a reserve pass with an inference pass — will trip Go's
	// runtime concurrent-map-access detector and panic.

	// encodeResults stores per-layer EncodeK result tensors for the current
	// forward pass. DequantK uses them as src[0] to establish the graph
	// dependency (encode before dequant in the ggml scheduler).
	encodeResults map[int]ml.Tensor

	// vEncodeResults stores per-layer EncodeV result tensors for the current
	// forward pass. DequantV uses them to establish the encode→dequant ordering.
	vEncodeResults map[int]ml.Tensor

	// logPathOnce ensures each active Get() path is logged at most once per
	// cache instance (avoids log spam: Get() is called every layer every step).
	logPathOnce [11]sync.Once

	// fusedFallbackEligible gates the inline-decode fused-FA fallback paths
	// (Get paths 2 and 4). CUDA/ROCm and Metal all support D=64, 128, 256, 512 —
	// covering llama3.2 (64), llama3.1/qwen2.5 (128), Gemma3/Gemma4 local-attn
	// (256), and Gemma4 global-attn (512). Other dims fall back to the DequantK
	// + stock FA path (Get paths 0/1/5).
	fusedFallbackEligible bool

	// preferFusedAttn is true on Metal. For K+V WHT presets the DequantKV →
	// stock FA path (Path 1) materialises a full f16 intermediate, doubling
	// KV bandwidth at long context. The fused inline-decode kernel (Path 2)
	// reads packed K+V once with no intermediate. On Metal the fused path is
	// dramatically faster; on CUDA/ROCm the intermediate stays in L2 and
	// stock FA is highly tuned, so DequantKV wins. When this flag is true
	// the Get() router skips Path 1 and uses Path 2 for K+V presets.
	preferFusedAttn bool

	// curQueryLen is the number of query tokens in the current forward pass,
	// captured at StartForward from len(batch.Positions). curQueryLen==1
	// means decode; curQueryLen>1 means prefill (or a batched prompt chunk).
	curQueryLen int

	// pendingKBiases buffers K projection biases set by SetLayerKBias before
	// activateGPUEncode has initialised compressedK. Applied to compressedK
	// once the manager is created.
	pendingKBiases map[int]ml.Tensor

	// rawKeys / encCells support the fused encode path (Metal + tq*k presets):
	// Put() stores the raw K tensor and the new cell index instead of calling
	// EncodeK separately; routeGet() then calls DequantKFusedEncode so the
	// single Metal dispatch handles both encode and decode, eliminating the
	// TQ_ENCODE→DQ cross-kernel barrier (~896 µs/token on gemma4:e2b D=512).
	// Both maps are cleared in StartForward like encodeResults.
	rawKeys  map[int]ml.Tensor
	encCells map[int]int
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
			rawKeys:        make(map[int]ml.Tensor),
			encCells:       make(map[int]int),
		}, true

	case *WrapperCache:
		// Mutate sub-caches in place: replace every *Causal (including SWA)
		// with a *TurboQuantCache. The TQ Put path takes the indexed
		// addressing route (EncodeKAt/EncodeKVAt) whenever curLocs is
		// fragmented, so SWA eviction's scattered free cells are written
		// correctly. All ~60 gemma3/gemma4 layers benefit from compression.
		wrapped := 0
		for i, sub := range c.caches {
			inner, ok := sub.(*Causal)
			if !ok {
				continue
			}
			c.caches[i] = &TurboQuantCache{
				meta:           inner,
				preset:         preset,
				encodeResults:  make(map[int]ml.Tensor),
				vEncodeResults: make(map[int]ml.Tensor),
				rawKeys:        make(map[int]ml.Tensor),
				encCells:       make(map[int]int),
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
			rawKeys:        make(map[int]ml.Tensor),
			encCells:       make(map[int]int),
		})
		slog.Info("turboquant: wrapped attention KV in hybrid recurrent cache",
			"preset", preset.Name)
		return cache, true

	default:
		slog.Warn("turboquant: underlying cache is not *Causal or *WrapperCache, falling back to unwrapped cache")
		return cache, false
	}
}

// SetTQModelConfig walks the cache tree and calls SetModelConfig on every
// *TurboQuantCache it finds. Must be called after WrapWithTurboQuant and
// before model.SetCache so the head dimensions are valid for encode setup.
func SetTQModelConfig(cache Cache, headDim, nKVHeads int) {
	switch c := cache.(type) {
	case *TurboQuantCache:
		c.SetModelConfig(headDim, nKVHeads)
	case *WrapperCache:
		for _, sub := range c.caches {
			if tq, ok := sub.(*TurboQuantCache); ok {
				tq.SetModelConfig(headDim, nKVHeads)
			}
		}
	}
}

func (c *TurboQuantCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	// K is compressed when KeyPrimaryBits > 0; V is compressed when ValueBits > 0.
	// Skipping the inner Causal allocation only on the compressed plane lets
	// V-only presets (KeyPrimaryBits == 0) keep K in the inner cache as raw f16
	// while V flows through the TQ manager. Likewise for K-only presets.
	if c.preset.KeyPrimaryBits > 0 {
		c.meta.SkipK = true
	}
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

// SetModelConfig seeds headDim, numKVHeads, and QK-norm detection before
// model.SetCache triggers per-layer SetLayerKWeight calls. Must be called
// before SetCache so ChannelNorms receives non-zero dimensions.
func (c *TurboQuantCache) SetModelConfig(headDim, nKVHeads int) {
	if c.headDim == 0 {
		c.headDim = headDim
	}
	if c.numKVHeads == 0 {
		c.numKVHeads = nKVHeads
	}
}

func (c *TurboQuantCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	c.isReserve = reserve
	c.curQueryLen = len(batch.Positions)
	clear(c.encodeResults)
	clear(c.vEncodeResults)
	clear(c.rawKeys)
	clear(c.encCells)
	return c.meta.StartForward(ctx, batch, reserve)
}

func (c *TurboQuantCache) Put(ctx ml.Context, key, value ml.Tensor) {
	// Pin headDim/numKVHeads to the real key on every Put until GPU encode is
	// activated (phase2Checked) — not just when c.headDim is still 0. They can be
	// set earlier from a placeholder (weight setup or a reserve pass), and the
	// dims captured here are what activateGPUEncode() below uses to build the GPU
	// manager and size per-layer buffers, so they must track the actual key.
	if key != nil && !c.phase2Checked {
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
			if c.preset.KeyPrimaryBits > 0 {
				c.compressedK.EnsureLayer(layer, capacity)
			}
			if c.preset.ValueBits > 0 {
				c.compressedK.EnsureVLayer(layer, capacity)
			}
			switch {
			case c.preset.KeyPrimaryBits > 0 && c.preset.ValueBits > 0:
				kResult, vResult := c.compressedK.EncodeKV(ctx, layer, key, value, 0)
				if kResult != nil {
					ctx.Forward(kResult)
					c.encodeResults[layer] = kResult
					c.vEncodeResults[layer] = vResult
				}
			case c.preset.KeyPrimaryBits > 0:
				encodeResult := c.compressedK.EncodeK(ctx, layer, key, 0)
				if encodeResult != nil {
					ctx.Forward(encodeResult)
					c.encodeResults[layer] = encodeResult
				}
			case c.preset.ValueBits > 0:
				// V-only: K stays raw in the inner Causal cache, V flows
				// through the TQ manager.
				vResult := c.compressedK.EncodeV(ctx, layer, value, 0)
				if vResult != nil {
					ctx.Forward(vResult)
					c.vEncodeResults[layer] = vResult
				}
			}
		}
		return
	}

	// Decide between contiguous (firstCell+i) and indexed (locs[i]) addressing.
	// Plain Causal caches hand back a contiguous run from findLocs; SWA caches
	// can return fragmented slots. The TQ kernels handle both via the *At
	// variants, which upload the locs tensor and branch block-uniformly.
	firstCell := 0
	contiguous := true
	if len(c.meta.curLocs) > 0 {
		firstCell = c.meta.curLocs[0]
		for i := 1; i < len(c.meta.curLocs); i++ {
			if c.meta.curLocs[i] != firstCell+i {
				contiguous = false
				break
			}
		}
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		capacity := len(c.meta.cells)
		if c.preset.KeyPrimaryBits > 0 {
			c.compressedK.EnsureLayer(layer, capacity)
		}
		if c.preset.ValueBits > 0 {
			c.compressedK.EnsureVLayer(layer, capacity)
		}

		var locsTensor ml.Tensor
		if !contiguous {
			locs32 := make([]int32, len(c.meta.curLocs))
			for i, v := range c.meta.curLocs {
				locs32[i] = int32(v)
			}
			locsTensor = ctx.Input().FromInts(locs32, len(locs32))
		}

		switch {
		case c.preset.KeyPrimaryBits > 0 && c.preset.ValueBits > 0:
			// Combined K+V encode: single GGML op, two back-to-back kernels.
			var kResult, vResult ml.Tensor
			if contiguous {
				kResult, vResult = c.compressedK.EncodeKV(ctx, layer, key, value, firstCell)
			} else {
				kResult, vResult = c.compressedK.EncodeKVAt(ctx, layer, key, value, locsTensor)
			}
			if kResult != nil {
				ctx.Forward(kResult)
				c.encodeResults[layer] = kResult
				c.vEncodeResults[layer] = vResult
			}
		case c.preset.KeyPrimaryBits > 0:
			// K-only presets (tq2k/tq3k/tq4k): V stays as f16 in the Causal cache.
			// Fused encode+decode path (Metal only, tq4k-style: no codebook, WHT, outliers):
			// defer encode to the DQ kernel, eliminating the TQ_ENCODE→DQ barrier (~32μs/layer).
			type fusedEncSupport interface {
				SupportsFusedEncDequantKWHT() bool
				PackedTensor(layer int) ml.Tensor
			}
			// Take the fused encode+decode path only when all four hold:
			//   !disableFusedEncode — OLLAMA_TQ_DISABLE_FUSED_ENCODE ablation is off.
			//   contiguous          — the kernel encodes one new cell inline at
			//                          firstCell; there is no indexed (*At) fused
			//                          encode, so fragmented SWA slots fall back.
			//   preferFusedAttn     — decode routes through the fused DequantK FA
			//                          kernel that the encode folds into; without it
			//                          there is no DQ kernel to absorb the encode.
			//   curQueryLen == 1    — decode only (one new cell per step). Prefill
			//                          writes many cells at once and would need
			//                          kernel changes, so it uses the batch EncodeK.
			if !disableFusedEncode && contiguous && c.preferFusedAttn && c.curQueryLen == 1 {
				if fes, ok := c.compressedK.(fusedEncSupport); ok && fes.SupportsFusedEncDequantKWHT() {
					if sent := fes.PackedTensor(layer); sent != nil {
						c.rawKeys[layer] = key
						c.encCells[layer] = firstCell
						c.encodeResults[layer] = sent
						break
					}
				}
			}
			var encodeResult ml.Tensor
			if contiguous {
				encodeResult = c.compressedK.EncodeK(ctx, layer, key, firstCell)
			} else {
				encodeResult = c.compressedK.EncodeKAt(ctx, layer, key, locsTensor)
			}
			if encodeResult != nil {
				ctx.Forward(encodeResult)
				c.encodeResults[layer] = encodeResult
			}
		case c.preset.ValueBits > 0:
			// V-only presets (tq2v/tq3v/tq4v): K stays as f16 in the Causal cache.
			var vResult ml.Tensor
			if contiguous {
				vResult = c.compressedK.EncodeV(ctx, layer, value, firstCell)
			} else {
				vResult = c.compressedK.EncodeVAt(ctx, layer, value, locsTensor)
			}
			if vResult != nil {
				ctx.Forward(vResult)
				c.vEncodeResults[layer] = vResult
			}
		}

		// Inner Causal.Put() tracks cell metadata (positions, masks).
		// SkipK and SkipV suppress the actual K/V tensor writes.
		c.meta.Put(ctx, key, value)
		return
	}

	c.meta.Put(ctx, key, value)
}

// tqGetPath enumerates the routing decisions that Get() can make.
// Both the inference and reserve branches must agree on the chosen path —
// any divergence causes gallocr to pre-reserve a different scratch shape
// than inference produces, leading to silent memory corruption on long
// contexts (we've been bitten by this before; see feedback memory
// `feedback_reserve_inference_graph_must_match`).
type tqGetPath int

const (
	tqPathNone                tqGetPath = iota // no encoded K — caller falls back to meta or placeholder
	tqPathKOnlyFusedNoAsymm                    // Path 0: K-only, no asymmetric, fused inline-decode
	tqPathKOnlyDequantNoAsymm                  // Path 0 fallback: K-only, no asymmetric, DequantK
	tqPathKVDequant                            // Path 1: K+V combined dequant → stock FA (CUDA/ROCm default)
	tqPathKVFused                              // Path 2: K+V fused inline-decode (Metal default; or OLLAMA_TQ_FORCE_FUSED)
	tqPathKVDequantWHT                         // Path 2b: K+V WHT fallback — DequantK+WHT + DequantV+WHT
	tqPathKOnlyDequantWHT                      // Path 2c: K-only WHT fused dequant+WHT undo
	tqPathKOnlyInlineDecode                    // Path 4: K-only fused inline-decode (Metal/ROCm without 2c)
	tqPathSeparateDequant                      // Path 6: separate K+V dequant — last-resort fallback
	tqPathVOnlyDequant                         // Path 8: V-only — DequantV (WHT-undone) → stock FA on raw f16 K
	tqPathVOnlyFused                           // Path 8-fused: V-only — packed V decoded inline + raw f16 K (CUDA/ROCm decode)
	tqPathKVHybrid                             // Path 1.5: K+V hybrid — DequantK f16 K + V-only fused inline-decode (CUDA decode)
)

// tqGetResult captures the outcome of a single routing decision.
type tqGetResult struct {
	path  tqGetPath
	key   ml.Tensor // K (or fused K+V tqTensor when skipMetaValue=true)
	value ml.Tensor // V tensor when produced by the path; nil when V comes from meta cache or is inline
	// skipMetaValue is set when key carries V inline (K+V fused) — caller must
	// NOT pull V from meta or synthesize an f16 V placeholder, otherwise the
	// fused-path's bandwidth win is undone (path 2 inserts a phantom f16 V scratch).
	skipMetaValue bool
}

// routeGet is the single source of truth for how a Get() call dispatches against
// the compressed-K manager. The reserve graph and the inference graph BOTH call
// this with their respective firstCell so gallocr pre-reserves the same scratch
// shape inference produces. Adding/removing/reordering a path here automatically
// keeps both branches in sync.
//
// Returns (tqPathNone, nil, nil, false) when no encoded K exists for this layer
// — caller should fall back to meta.Get / placeholder synthesis.
// routeGet dispatches a Get() call to the appropriate TQ path.
// locs is nil for contiguous cell ranges (firstCell, firstCell+1, …, firstCell+nCells-1).
// For fragmented (SWA-evicted) ranges, locs is a [nCells]i32 tensor of physical
// slot indices and firstCell is ignored — each cell is addressed via locs[i].
func (c *TurboQuantCache) routeGet(ctx ml.Context, layer, firstCell, nCells int, locs ml.Tensor) tqGetResult {
	// IMPORTANT: this routing must stay identical for reserve and inference.
	// Both branches of Get() call this function with their respective firstCell,
	// and gallocr's pre-reservation depends on it producing the same set of
	// graph nodes inference will produce. New paths go HERE only — never inline
	// a path-decision into Get() directly, or the two branches will silently
	// diverge and gallocr will under- or over-reserve KV scratch.
	if c.compressedK == nil {
		return tqGetResult{path: tqPathNone}
	}
	enc := c.encodeResults[layer]
	vEnc := c.vEncodeResults[layer]

	indexed := locs != nil

	// ── V-only presets (KeyPrimaryBits == 0, ValueBits > 0) ───────────────
	// K stays raw f16 in the inner Causal cache; the caller (Get) pulls it
	// from c.meta.Get directly. We only need to produce V.
	//
	// Path 8-fused (CUDA, D∈{64,128,256,512}, decode only): V decoded inline
	// in the fused kernel; no V scratch. Decode-only because the fused kernel
	// reads V_packed once per Q-column block — with ncols=512 prefill blocks
	// the effective V bandwidth is 512× higher than DequantV, causing a 3×
	// collapse at ctx=8192 on Blackwell (326 vs 889 tok/s). DequantV
	// materialises V once and stock FA reads it once regardless of ncols.
	// Reserve takes the same decode path so gallocr pre-reserves zero V scratch.
	// Normally contiguous CUDA only; OLLAMA_TQ_FORCE_VONLY_FUSED=1 also enables
	// indexed/SWA and non-CUDA backends (ROCm, Metal) for benchmarking.
	//
	// Path 8 (all other cases): DequantV + optional WHTUndo → stock FA.
	if c.preset.KeyPrimaryBits == 0 && c.preset.ValueBits > 0 {
		if vEnc == nil {
			return tqGetResult{path: tqPathNone}
		}
		// Path 8-fused: decode-only (curQueryLen==1). Normally CUDA contiguous only;
		// forceVOnlyFused also enables indexed and non-CUDA backends for benchmarking.
		if c.curQueryLen == 1 && !forceDeQuantV && (c.compressedK.SupportsVOnlyFused() || forceVOnlyFused) &&
			(!indexed || forceVOnlyFused) {
			var tqv ml.Tensor
			var ok bool
			if indexed {
				tqv, ok = c.compressedK.GetAsTQVTensorAt(ctx, layer, vEnc, locs, nCells)
			} else {
				tqv, ok = c.compressedK.GetAsTQVTensor(ctx, layer, vEnc, firstCell, nCells)
			}
			if ok {
				return tqGetResult{path: tqPathVOnlyFused, value: tqv}
			}
		}
		// Path 8: DequantV (WHT-undone) + raw f16 K → stock FA.
		var v ml.Tensor
		if indexed {
			v = c.compressedK.DequantVAt(ctx, layer, vEnc, locs)
		} else {
			v = c.compressedK.DequantV(ctx, layer, vEnc, firstCell, nCells)
		}
		if v == nil {
			return tqGetResult{path: tqPathNone}
		}
		if c.compressedK.HasRotation() {
			v = c.compressedK.WHTUndo(ctx, layer, v)
		}
		return tqGetResult{path: tqPathVOnlyDequant, value: v}
	}

	// All other paths require K to have been TQ-encoded.
	if enc == nil {
		return tqGetResult{path: tqPathNone}
	}

	// ── K+V presets (preset.ValueBits > 0) ──────────────────────────────
	// Order matches the original inference branch: fused first (when the
	// backend prefers it or the user forced it), then combined DequantKV
	// (CUDA/ROCm default), then the WHT-no-outlier dequant fallback, then
	// separate dequant. Reordering this changes the path taken by ablation
	// configurations like OLLAMA_TQ_DISABLE_OUTLIERS=1 on Metal.
	if c.preset.ValueBits > 0 && vEnc != nil {
		// Path 2: K+V fused inline-decode.
		// Auto-selected on Metal (preferFusedAttn=true) because materialising
		// the f16 intermediate doubles KV bandwidth at long context. Opt-in
		// elsewhere via OLLAMA_TQ_FORCE_FUSED for benchmarking. Reserve must
		// honour forceFused too or scratch sizes diverge.
		if !forceDeQuantKV && c.fusedFallbackEligible && (forceFused || c.preferFusedAttn) {
			var tqkv ml.Tensor
			var ok bool
			if indexed {
				tqkv, ok = c.compressedK.GetAsTQTensorKVAt(ctx, layer, enc, vEnc, locs)
			} else {
				tqkv, ok = c.compressedK.GetAsTQTensorKV(ctx, layer, enc, vEnc, firstCell, nCells)
			}
			if ok {
				return tqGetResult{path: tqPathKVFused, key: tqkv, skipMetaValue: true}
			}
		}
		// Path 1.5: KV hybrid — DequantK → f16 K + V-only fused inline-decode.
		// On CUDA decode: materialises f16 K via DequantK then feeds it to the
		// V-only fused kernel which decodes V inline. Saves ~256 bytes/cell-head
		// of f16 V traffic vs Path 1 (DequantKV → stock FA); predicts ~75% of
		// f16 throughput (vs ~54% for Path 1). forceDeQuantKV disables it (same
		// as Path 1 — any "force dequant" opt means stock FA with f16 K+V).
		// Path 1.5: K+V hybrid — DequantK f16 K + V-only fused inline-decode.
		// Eliminates f16 V scratch vs Path 1. Supports both contiguous and indexed
		// (SWA-evicted) caches; indexed uses DequantKAt + GetAsTQVTensorAt.
		// SupportsVOnlyFused gates on cp.async (Ampere cc 8.0+): the inline-V
		// decode only beats materialised-V + stock FA when the fused kernel can
		// hide the packed-V load latency. Pascal/Volta/Turing/ROCm lack cp.async
		// → fall through to Path 1. Same predicate gates Path 8-fused (V-only).
		if c.curQueryLen == 1 && !forceFused && !forceDeQuantKV &&
			!c.preferFusedAttn && c.compressedK.SupportsVOnlyFused() {
			var tqv ml.Tensor
			var tqvOK bool
			if indexed {
				tqv, tqvOK = c.compressedK.GetAsTQVTensorAt(ctx, layer, vEnc, locs, nCells)
			} else {
				tqv, tqvOK = c.compressedK.GetAsTQVTensor(ctx, layer, vEnc, firstCell, nCells)
			}
			if tqvOK {
				var k ml.Tensor
				if indexed {
					k = c.compressedK.DequantKAt(ctx, layer, enc, locs)
				} else {
					k = c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells)
				}
				if k != nil {
					return tqGetResult{path: tqPathKVHybrid, key: k, value: tqv}
				}
			}
		}
		// Path 1: K+V combined DequantKV → stock FA (CUDA/ROCm default for
		// non-WHT and WHT+outlier; the dispatcher fuses WHT undo into the
		// kernels when v_rotation != NULL && k_has_outliers).
		if !forceFused && (!c.preferFusedAttn || forceDeQuantKV) &&
			(!c.compressedK.HasRotation() || c.preset.HasOutlierSplit()) {
			var k, v ml.Tensor
			if indexed {
				k, v = c.compressedK.DequantKVAt(ctx, layer, enc, vEnc, locs)
			} else {
				k, v = c.compressedK.DequantKV(ctx, layer, enc, vEnc, firstCell, nCells)
			}
			if k != nil && v != nil {
				return tqGetResult{path: tqPathKVDequant, key: k, value: v}
			}
		}
		// Path 2b: WHT K+V no-outlier (test-only configuration).
		// DequantK + WHTUndo + DequantV + WHTUndo → stock FA. Specifically
		// handles HasRotation && !HasOutlier, which Path 1 skips.
		if c.compressedK.HasRotation() && !c.preset.HasOutlierSplit() {
			var k, v ml.Tensor
			if indexed {
				k = c.compressedK.DequantKAt(ctx, layer, enc, locs)
				v = c.compressedK.DequantVAt(ctx, layer, vEnc, locs)
			} else {
				k = c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells)
				v = c.compressedK.DequantV(ctx, layer, vEnc, firstCell, nCells)
			}
			k = c.compressedK.WHTUndo(ctx, layer, k)
			v = c.compressedK.WHTUndo(ctx, layer, v)
			if k != nil && v != nil {
				return tqGetResult{path: tqPathKVDequantWHT, key: k, value: v}
			}
		}
		// Path 5: separate dequant — last-resort fallback. Return tqPathNone
		// when both planes failed so the caller falls back cleanly to meta.Get
		// rather than receiving a partial (key=nil, value=non-nil) result that
		// the inference guard would silently drop while still emitting the
		// graph node for the dropped tensor.
		var k, v ml.Tensor
		if indexed {
			k = c.compressedK.DequantKAt(ctx, layer, enc, locs)
			v = c.compressedK.DequantVAt(ctx, layer, vEnc, locs)
		} else {
			k = c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells)
			v = c.compressedK.DequantV(ctx, layer, vEnc, firstCell, nCells)
		}
		if k == nil && v == nil {
			return tqGetResult{path: tqPathNone}
		}
		return tqGetResult{path: tqPathSeparateDequant, key: k, value: v}
	}

	// ── K-only presets (preset.ValueBits == 0) ──────────────────────────
	// Path 0: no asymmetric primary (ablation under OLLAMA_TQ_DISABLE_ASYMMETRIC).
	if !c.preset.AsymmetricPrimary {
		if c.fusedFallbackEligible {
			var tqk ml.Tensor
			var ok bool
			if indexed {
				tqk, ok = c.compressedK.GetAsTQTensorAt(ctx, layer, enc, locs)
			} else {
				tqk, ok = c.compressedK.GetAsTQTensor(ctx, layer, enc, firstCell, nCells)
			}
			if ok {
				return tqGetResult{path: tqPathKOnlyFusedNoAsymm, key: tqk}
			}
		}
		var k ml.Tensor
		if indexed {
			k = c.compressedK.DequantKAt(ctx, layer, enc, locs)
		} else {
			k = c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells)
		}
		if k != nil {
			return tqGetResult{path: tqPathKOnlyDequantNoAsymm, key: k}
		}
	}
	// Path 2c: WHT K-only with fused dequant+WHT support (CUDA, and Metal
	// since 2026-05-11). Single-op DequantK with WHT signs threaded through.
	// OLLAMA_TQ_DISABLE_FUSED_DEQUANT_K=1 forces tq*k off this path so it
	// falls through to Path 4 (TQ FA op), where Phase D's wedge can engage.
	// NOTE: On Metal M1 Pro (ctx=256, D=512), Path 2c (stock FA after DequantK)
	// beats WHT-Q FA at 46.8 vs 29.7 tok/s. Empirically on Blackwell sm_120,
	// path 2c also beats path 4 for tq*k decode: 52.3 vs 46.2 tok/s at ctx=2048.
	// Indexed mode skips the fused-encode subpath (no *At variant exists for it).
	if !disableFusedDequantK && c.compressedK.HasRotation() && c.compressedK.SupportsFusedDequantKWHT() {
		// Fused encode+decode: encode was deferred to the DQ kernel (Metal, tq4k-style).
		// Only available for contiguous ranges.
		if !indexed && !disableFusedEncode {
			if rawKey := c.rawKeys[layer]; rawKey != nil {
				type fusedEncDQ interface {
					DequantKFusedEncode(ctx ml.Context, layer int, key ml.Tensor, encCell, firstCell, nCells int) ml.Tensor
				}
				if fed, ok := c.compressedK.(fusedEncDQ); ok {
					if k := fed.DequantKFusedEncode(ctx, layer, rawKey, c.encCells[layer], firstCell, nCells); k != nil {
						return tqGetResult{path: tqPathKOnlyDequantWHT, key: k}
					}
				}
			}
		}
		var k ml.Tensor
		if indexed {
			k = c.compressedK.DequantKAt(ctx, layer, enc, locs)
		} else {
			k = c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells)
		}
		if k != nil {
			return tqGetResult{path: tqPathKOnlyDequantWHT, key: k}
		}
	}
	// Path 4: K-only fused inline-decode. Used when 2c isn't available
	// (some ROCm builds, or pre-2026-05-11 Metal).
	if c.fusedFallbackEligible {
		var tqk ml.Tensor
		var ok bool
		if indexed {
			tqk, ok = c.compressedK.GetAsTQTensorAt(ctx, layer, enc, locs)
		} else {
			tqk, ok = c.compressedK.GetAsTQTensor(ctx, layer, enc, firstCell, nCells)
		}
		if ok {
			return tqGetResult{path: tqPathKOnlyInlineDecode, key: tqk}
		}
	}
	// Path 5 (K-only): plain DequantK fallback.
	var k ml.Tensor
	if indexed {
		k = c.compressedK.DequantKAt(ctx, layer, enc, locs)
	} else {
		k = c.compressedK.DequantK(ctx, layer, enc, firstCell, nCells)
	}
	if k != nil {
		return tqGetResult{path: tqPathSeparateDequant, key: k}
	}
	return tqGetResult{path: tqPathNone}
}

// pathLogger maps each tqGetPath to its (logPathOnce slot, message, level).
// Inference uses this; reserve doesn't log. Slot 3 is intentionally unused
// (legacy from path renumbering) — leaving it that way to avoid churn.
func (c *TurboQuantCache) pathLogger(p tqGetPath) (slot int, msg string, warn bool) {
	switch p {
	case tqPathKOnlyFusedNoAsymm:
		return 0, "turboquant: using K-only fused path", false
	case tqPathKOnlyDequantNoAsymm:
		return 0, "turboquant: using K-only DequantK + f16 V path (fused unavailable)", false
	case tqPathKVDequant:
		if c.compressedK.HasRotation() {
			return 1, "turboquant: WHT K+V — fused DequantKV with WHT undo → stock FA", false
		}
		return 1, "turboquant: using combined DequantKV + stock FA path", false
	case tqPathKVFused:
		if c.preferFusedAttn {
			return 2, "turboquant: K+V fused inline-decode (Metal preferred — avoids f16 intermediate)", false
		}
		return 2, "turboquant: using K+V fused inline-decode path (forced via OLLAMA_TQ_FORCE_FUSED)", true
	case tqPathKVDequantWHT:
		return 5, "turboquant: WHT K+V — DequantK + DequantV + WHT undo → stock FA", false
	case tqPathKOnlyDequantWHT:
		return 6, "turboquant: WHT K-only — fused DequantK+WHT undo → stock FA (f16 V)", false
	case tqPathKOnlyInlineDecode:
		return 4, "turboquant: using K-only inline-decode fused kernel", false
	case tqPathSeparateDequant:
		return 7, "turboquant: falling back to separate K + V dequant path", true
	case tqPathVOnlyDequant:
		return 8, "turboquant: V-only — DequantV (WHT-undone) + raw f16 K → stock FA", false
	case tqPathVOnlyFused:
		return 9, "turboquant: V-only — packed V inline-decode + raw f16 K → fused FA", false
	case tqPathKVHybrid:
		return 10, "turboquant: K+V hybrid — DequantK f16 K + V-only fused inline-decode", false
	}
	return -1, "", false
}

func (c *TurboQuantCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	if c.isReserve {
		key, value, mask := c.meta.Get(ctx)
		// Both reserve and inference route through routeGet so gallocr
		// pre-reserves exactly the scratch shape inference will produce.
		// Without this lock-step routing the graph-size estimate diverges
		// from runtime allocation; gemma4:31b burned 384 MiB on this exact
		// bug before the routing was unified.
		//
		// IMPORTANT: route on c.compressedK presence, NOT on key==nil. In
		// V-only mode SkipK=false and meta.Get returns a non-nil raw-f16 K,
		// so the previous "key == nil" gate skipped routeGet entirely — and
		// the reserve graph then had no DequantV / WHTUndo nodes while
		// inference emits them. Same gallocr-divergence bug class. Mirror
		// the inference branch's c.compressedK gate here.
		if c.compressedK != nil && c.headDim > 0 {
			nCells := 1
			if c.meta.curMask != nil {
				nCells = c.meta.curMask.Dim(0)
			}
			r := c.routeGet(ctx, c.meta.curLayer, 0, nCells, nil)
			if r.key != nil {
				key = r.key
			}
			if r.value != nil {
				value = r.value
			}
			// SkipV: synthesize V placeholder unless the chosen path placed a
			// K+V fused tensor (V carried inline in `key`). Synthesising f16
			// Zeros for the fused path would re-introduce the scratch buffer
			// the fused path is specifically designed to avoid.
			if value == nil && !r.skipMetaValue {
				value = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
			}
			if key == nil {
				key = ctx.Input().Zeros(ml.DTypeF16, c.headDim, c.numKVHeads, nCells)
			}
		}
		return key, value, mask
	}

	if c.compressedK != nil {
		layer := c.meta.curLayer
		firstCell := c.meta.curCellRange.min
		nCells := c.meta.curMask.Dim(0)
		// The attention window for TQ is always a contiguous range of physical
		// cells (firstCell … firstCell+nCells-1) for both plain Causal and SWA.
		// c.meta.curLocs holds the PUT locations for the current batch (new cells
		// being encoded), not the full attention window — using it here caused
		// indexed mode to fire incorrectly on multi-sub-batch prefill, decoding
		// only the current batch's cells instead of the full history.
		var locs ml.Tensor

		// Single source of truth for path selection — see routeGet.
		// The reserve graph above calls the same helper; that lock-step is
		// load-bearing for gallocr scratch sizing. Accept any non-nil tensor
		// so a partial result (e.g. K dequant succeeded but V didn't) isn't
		// silently dropped after its graph node was already emitted.
		r := c.routeGet(ctx, layer, firstCell, nCells, locs)
		if r.path != tqPathNone && (r.key != nil || r.value != nil) {
			if slot, msg, warn := c.pathLogger(r.path); slot >= 0 {
				c.logPathOnce[slot].Do(func() {
					if warn {
						slog.Warn(msg)
					} else {
						slog.Info(msg)
					}
				})
			}
			metaKey, metaValue, mask := c.meta.Get(ctx)
			key := r.key
			// V-only paths leave r.key nil because K stays as raw f16 in
			// the inner Causal cache (SkipK=false). Fall back to metaKey
			// in that case. K-only and K+V paths always produce r.key,
			// so the fallback is a no-op for them.
			if key == nil {
				key = metaKey
			}
			value := r.value
			if value == nil && !r.skipMetaValue {
				value = metaValue
			}
			if r.skipMetaValue {
				return key, nil, mask
			}
			return key, value, mask
		}
	}

	return c.meta.Get(ctx)
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

	// The user requested a measurement-grade preset if asymmetric primary
	// quantization is on. If we can't activate it on the GPU we produce
	// exactly f16 output and any PPL measurement is a lie. Log that failure
	// at ERROR with a distinctive marker so smoke tests and humans can both
	// notice instead of silently reading bit-identical-to-f16 numbers and
	// calling them success.
	measurementRequested := c.preset.AsymmetricPrimary
	logActivation := func(gpuActive bool, path string) {
		level := slog.LevelInfo
		if measurementRequested && !gpuActive {
			level = slog.LevelError
		}
		slog.Log(context.TODO(), level, "TQ_ACTIVATION",
			"preset", c.preset.Name,
			"asymmetric_requested", c.preset.AsymmetricPrimary,
			"outlier_count", c.preset.OutlierCount,
			"gpu_active", gpuActive,
			"measurement_requested", measurementRequested,
			"path", path,
		)
	}

	// Architectural compatibility gate: the TQ encode kernel and fused fattn
	// templates handle headDim ∈ {64, 128, 256, 512}. Models outside this set
	// (e.g. glm-4.7 / DeepSeek-MLA at headDim=576) must skip TQ entirely and
	// stay on f16 — the slow DequantK fallback also relies on encoder kernels
	// that assume those dims.
	if c.headDim != 64 && c.headDim != 128 && c.headDim != 256 && c.headDim != 512 {
		slog.Warn("turboquant: headDim not supported, falling back to f16 KV cache",
			"preset", c.preset.Name,
			"headDim", c.headDim,
			"supported", "{64, 128, 256, 512}")
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
		c.preset.AsymmetricPrimary,
	)
	if mgr == nil {
		if measurementRequested {
			slog.Error("turboquant: GPU manager unavailable — SILENT F16 FALLBACK ACTIVE. Run on CUDA, or set OLLAMA_TQ_DISABLE_ASYMMETRIC=1 / OLLAMA_TQ_DISABLE_OUTLIERS=1 to drop to a backend-supported configuration.",
				"preset", c.preset.Name,
				"asymmetric", c.preset.AsymmetricPrimary)
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

	// Read the backend's K+V routing preference (Metal=true, CUDA/ROCm=false).
	// Wired into the Path 1 vs Path 2 selection in Get() — see the
	// preferFusedAttn field for rationale.
	type fusedAttnPreferrer interface {
		PreferFusedAttention() bool
	}
	if ff, ok := mgr.(fusedAttnPreferrer); ok {
		c.preferFusedAttn = ff.PreferFusedAttention()
		if c.preferFusedAttn {
			slog.Info("turboquant: K+V routing → fused inline-decode (Metal: avoids f16 intermediate)")
		}
	}

	// Large-D override: at headDim >= 512 the DequantKV f16 intermediate is
	// substantial (~16 MiB per layer at ctx=1024, growing with context) and
	// the fused inline-decode path (path 2) avoids materializing it. Route
	// K+V D>=512 through path 2 by promoting preferFusedAttn even on
	// CUDA/ROCm. The fused kernel pays a per-step decode penalty (~3× prefill
	// slowdown on Pascal P40 at D=512), partially recovered by codebook
	// pre-scaling in tq-fattn-vec.cuh.
	//
	// OLLAMA_TQ_FORCE_DEQUANT_KV reverts to path 1 for benchmarking.
	if c.headDim >= 512 && !c.preferFusedAttn {
		if forceDeQuantKV {
			slog.Info("turboquant: K+V routing → DequantKV forced via OLLAMA_TQ_FORCE_DEQUANT_KV (D>=512 default would be path 2)")
		} else {
			c.preferFusedAttn = true
			slog.Info("turboquant: K+V routing → fused inline-decode (D>=512: skip f16 K+V materialization)",
				"headDim", c.headDim)
		}
	}

	// The headDim ∈ {64, 128, 256, 512} gate at the top of this function
	// ensures we only get here on supported dims, so the fused-FA fallback
	// paths (Get paths 2 and 4) are always eligible.
	c.fusedFallbackEligible = true

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
	case ml.DTypeTQ2V:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ2V), true
	case ml.DTypeTQ3V:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ3V), true
	case ml.DTypeTQ4V:
		return turboquant.ApplyEnvOverrides(turboquant.PresetTQ4V), true
	default:
		return turboquant.Preset{}, false
	}
}

var _ Cache = (*TurboQuantCache)(nil)

// outlierReader is satisfied by ggmlTQCompressedK when outlier split is enabled.
type outlierReader interface {
	OutlierIndicesSnapshot(layer int) []int16
	TQOutlierCount() int
	TQNumKVHeads() int
}

// OutlierIndicesForLayer reads back the outlier-index GPU tensor for one layer as
// a flat []int16. For numKVHeads=1 with outlierCount=32, cell c's 32 outlier channel
// indices are at indices [c*32 .. c*32+32). Returns (nil, 0, 0) if outliers are not
// enabled, the layer is uninitialised, or the backend does not expose this method.
func (c *TurboQuantCache) OutlierIndicesForLayer(layer int) (indices []int16, outlierCount, numKVHeads int) {
	r, ok := c.compressedK.(outlierReader)
	if !ok {
		return nil, 0, 0
	}
	return r.OutlierIndicesSnapshot(layer), r.TQOutlierCount(), r.TQNumKVHeads()
}

// TurboQuantCaches returns all *TurboQuantCache instances contained within cache.
// For a bare *TurboQuantCache it returns a single-element slice.
// For a *WrapperCache it returns all *TurboQuantCache sub-caches.
// Returns nil if cache contains no TurboQuantCaches.
func TurboQuantCaches(cache Cache) []*TurboQuantCache {
	switch c := cache.(type) {
	case *TurboQuantCache:
		return []*TurboQuantCache{c}
	case *WrapperCache:
		var out []*TurboQuantCache
		for _, sub := range c.caches {
			if tqc, ok := sub.(*TurboQuantCache); ok {
				out = append(out, tqc)
			}
		}
		return out
	default:
		return nil
	}
}

// Note: TurboQuantCache intentionally does NOT implement CheckpointCache.
// CheckpointCache is for recurrent caches that need special per-sequence
// state restoration. The inner *Causal cache supports plain CanResume, which
// is the right semantics for TQ-wrapped caches too — the runner will fall
// through to the CanResume branch when TurboQuantCache is in use (see
// runner/ollamarunner/cache.go:163-170). An earlier implementation had a
// stub PrepareRestore that always returned (0, false), which forced a full
// prompt reprocess on every resume for long-context gemma/llama runs.
