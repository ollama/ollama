package nn

import (
	"fmt"
	"log/slog"
	"sync"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/kan"
)

// kanTrainer is the global KAN shadow trainer, set via SetKANTrainer.
// When non-nil, attention operations will shadow-train KAN layers
// alongside softmax and optionally hot-swap converged layers.
var (
	kanTrainer *kan.ShadowTrainer
	kanMu      sync.RWMutex
)

// kanPendingItem represents deferred training work.
// Tensor data is read AFTER Compute() materializes the graph.
type kanPendingItem struct {
	key       string
	kq        ml.Tensor // pre-softmax logits (needed for training input)
	softmax   ml.Tensor // softmax output (ground truth for Phase 1; nil for converged layers)
	shape     []int
	converged bool
}

var (
	kanPendingMu sync.Mutex
	kanPending   []kanPendingItem
)

// SetKANTrainer installs a KAN shadow trainer for all attention operations.
// Pass nil to disable KAN attention.
func SetKANTrainer(trainer *kan.ShadowTrainer) {
	kanMu.Lock()
	defer kanMu.Unlock()
	kanTrainer = trainer
	if trainer != nil {
		slog.Info("KAN attention shadow training enabled")
	}
}

// GetKANTrainer returns the current KAN trainer, or nil if disabled.
func GetKANTrainer() *kan.ShadowTrainer {
	kanMu.RLock()
	defer kanMu.RUnlock()
	return kanTrainer
}

// FlushKANTraining processes all deferred training work.
//
// MUST be called after ctx.Compute() so that tensor data is materialized.
// Reads logits and softmax outputs from the completed graph, then:
//   - For non-converged layers: shadow-trains the KAN (Phase 1)
//   - For converged layers with Phase 2: runs self-evolution step
//
// The heads dimension is flattened into the batch dimension for CPU-side
// processing (each head is treated as an independent attention pattern).
func FlushKANTraining() {
	kanMu.RLock()
	trainer := kanTrainer
	kanMu.RUnlock()
	if trainer == nil {
		return
	}

	kanPendingMu.Lock()
	work := kanPending
	kanPending = nil
	kanPendingMu.Unlock()

	for _, item := range work {
		func() {
			defer func() {
				if r := recover(); r != nil {
					slog.Debug("KAN training recovered from panic", "layer", item.key, "error", r)
				}
			}()

			shape := item.shape
			logits := item.kq.Floats()

			// Extract dimensions: kq shape after permute is [seqK, heads, seqQ]
			seqK := 1
			heads := 1
			seqQ := 1
			if len(shape) >= 1 {
				seqK = shape[0]
			}
			if len(shape) >= 2 {
				heads = shape[1]
			}
			if len(shape) >= 3 {
				seqQ = shape[2]
			}

			// Flatten heads into batch: treat as (seqQ * heads) rows of seqK.
			// The flat array layout from GGML after permute [seqK, heads, seqQ]
			// stores seqK as the fastest-varying dimension. So element at
			// (k, h, q) is at index q*heads*seqK + h*seqK + k.
			// Each contiguous seqK block is one row, and there are heads*seqQ of them.
			effectiveSeqQ := seqQ * heads

			if item.converged {
				// Phase 2: self-evolution on converged layer
				if trainer.IsPhase2Active(item.key) {
					trainer.Phase2Step(item.key, logits, seqK, effectiveSeqQ)
				}
			} else {
				// Phase 1: shadow training
				expected := item.softmax.Floats()
				trainer.TrainStep(item.key, logits, expected, seqK, effectiveSeqQ)
			}
		}()
	}
}

// Attention implements scaled dot-product attention for transformer models:
// Attention(Q, K, V) = softmax(QK^T/√d_k)V
//
// When KAN attention is enabled, this also shadow-trains a Geometric KAN
// to replace softmax. After convergence, the KAN's effect is expressed as
// temperature-scaled softmax in the GGML computation graph.
//
// Parameters:
//   - ctx: Context for tensor operations
//   - query: Query tensor (Q) with shape [d_k, heads, seq_len_q]
//   - key: Key tensor (K) with shape [d_k, kv_heads, seq_len_k], can be nil to read from cache only
//   - value: Value tensor (V) with shape [d_v, kv_heads, seq_len_k], can be nil to read from cache only
//   - scale: Scaling factor, typically 1/√d_k where d_k is the key dimension
//   - cache: KV cache to store key/value and get past history, can be nil to only use provided key/value
//
// Returns:
//
//	Attention output with shape [d_v, heads, seq_len_q]
func Attention(ctx ml.Context, query, key, value ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	return AttentionWithVMLA(ctx, query, key, value, nil, nil, scale, cache)
}

func AttentionWithSinks(ctx ml.Context, query, key, value, sinks ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	return AttentionWithVMLA(ctx, query, key, value, sinks, nil, scale, cache)
}

func AttentionWithVMLA(ctx ml.Context, query, key, value, sinks ml.Tensor, vmla ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	ctx.Forward(query)
	if key != nil && value != nil {
		if query.Dim(0) != key.Dim(0) {
			panic(fmt.Errorf("d_k in attention operation does not match between query(%v) and key(%v)", query.Dim(0), key.Dim(0)))
		}

		if key.Dim(1) != value.Dim(1) {
			panic(fmt.Errorf("kv_heads in attention operation does not match between key(%v) and value(%v)", key.Dim(1), value.Dim(1)))
		}

		if key.Dim(2) != value.Dim(2) {
			panic(fmt.Errorf("seq_len_k in attention operation does not match between key(%v) and value(%v)", key.Dim(2), value.Dim(2)))
		}

		ctx.Forward(key, value)
		if cache != nil {
			cache.Put(ctx, key, value)
		}
	} else if cache == nil {
		panic("key & value tensors must be provided if cache is nil")
	}

	var mask ml.Tensor
	if cache != nil {
		key, value, mask = cache.Get(ctx)
	}

	// When KAN attention is active, we MUST use the manual attention path
	// to access pre-softmax logits for training and to apply effectiveScale.
	// The GGML backend always implements ScaledDotProductAttention (even with
	// flash attention disabled), so we skip that path entirely when KAN is on.
	kanMu.RLock()
	kanActive := kanTrainer != nil
	kanMu.RUnlock()

	if !kanActive {
		if sdpa, ok := query.(ml.ScaledDotProductAttention); ok {
			// Flash/fused attention path: softmax is inside the kernel.
			// KAN cannot operate here since logits are never materialized.
			cacheConfigApplied := cache != nil
			return sdpa.ScaledDotProductAttention(ctx, key, value, mask, sinks, vmla, scale, cacheConfigApplied)
		}
	}

	// Manual attention path: we have access to pre-softmax logits.
	// This is where KAN shadow training and hot-swap happens.
	query = query.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := key.MulmatFullPrec(ctx, query)
	kq = kq.Scale(ctx, scale)
	if mask != nil {
		kq = kq.Add(ctx, mask)
	}

	// === KAN Shadow Training / Hot-Swap Point ===
	kq = applyAttentionWeights(ctx, kq)

	kqv := value.Mulmat(ctx, kq)

	if vmla != nil {
		kqv = vmla.Mulmat(ctx, kqv)
	}

	return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
}

// applyAttentionWeights applies either softmax or KAN-enhanced attention weights.
//
// All operations stay within the GGML computation graph (no tensor data reads).
// Training work is deferred to FlushKANTraining() which runs after Compute().
//
// When no KAN trainer is active: standard softmax.
//
// When KAN is active and layer has converged:
//
//	Temperature-scaled softmax using the KAN's learned effective scale.
//	This captures the KAN's sharpening effect without needing custom GPU kernels.
//	softmax(effectiveScale * logits) ≈ KAN(logits)
//
// When KAN is active but not yet converged:
//
//	Standard softmax (ground truth), with tensors registered for
//	deferred shadow training after Compute().
func applyAttentionWeights(ctx ml.Context, kq ml.Tensor) ml.Tensor {
	kanMu.RLock()
	trainer := kanTrainer
	kanMu.RUnlock()

	if trainer == nil {
		return kq.Softmax(ctx)
	}

	layerIdx := ctx.LayerIndex()
	if layerIdx < 0 {
		// No layer context (e.g., vision encoder) -- fall back to softmax
		return kq.Softmax(ctx)
	}

	key := kan.LayerKey(layerIdx)

	// Ensure the layer exists in the trainer
	trainer.GetOrCreateLayer(key)

	if trainer.IsConverged(key) {
		// Hot-swap: express KAN's effect as temperature-scaled softmax.
		// The effective scale captures the KAN's learned transform slope.
		// scale > 1.0 = sharper attention (Phase 2 effect).
		effectiveScale := trainer.GetEffectiveScale(key)

		// Register for deferred Phase 2 evolution (processed after Compute)
		if trainer.IsPhase2Active(key) {
			kanPendingMu.Lock()
			kanPending = append(kanPending, kanPendingItem{
				key:       key,
				kq:        kq,
				shape:     kq.Shape(),
				converged: true,
			})
			kanPendingMu.Unlock()
		}

		if effectiveScale != 1.0 {
			kq = kq.Scale(ctx, effectiveScale)
		}
		return kq.Softmax(ctx)
	}

	// Not converged: use standard softmax as output
	softmaxOut := kq.Softmax(ctx)

	// Register for deferred shadow training (processed after Compute)
	if trainer.ShouldTrain(key) {
		kanPendingMu.Lock()
		kanPending = append(kanPending, kanPendingItem{
			key:     key,
			kq:      kq,
			softmax: softmaxOut,
			shape:   kq.Shape(),
		})
		kanPendingMu.Unlock()
	}

	return softmaxOut
}
