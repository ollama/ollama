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

	// KAN attention: only bypass SDPA for layers still training.
	// Converged layers pass through SDPA with effectiveScale baked into the
	// scale factor — full flash attention performance is restored after convergence.
	kanMu.RLock()
	trainer := kanTrainer
	kanMu.RUnlock()

	needManualPath := false
	kanScale := 1.0
	if trainer != nil {
		layerIdx := ctx.LayerIndex()
		if layerIdx >= 0 {
			key := kan.LayerKey(layerIdx)
			trainer.GetOrCreateLayer(key)

			if trainer.IsConverged(key) {
				// Converged: can use SDPA with the KAN's effective scale.
				// This preserves flash attention performance while applying
				// the Phase 2 sharpening effect.
				kanScale = trainer.GetEffectiveScale(key)

				// Still register for Phase 2 evolution if active.
				// We can't capture logits through SDPA, but Phase 2 can
				// use its existing state to continue evolving.
			} else {
				// Not converged: need manual path to capture logits for training
				needManualPath = true
			}
		}
	}

	if !needManualPath {
		if sdpa, ok := query.(ml.ScaledDotProductAttention); ok {
			cacheConfigApplied := cache != nil
			return sdpa.ScaledDotProductAttention(ctx, key, value, mask, sinks, vmla, scale*kanScale, cacheConfigApplied)
		}
	}

	// Manual attention path: we have access to pre-softmax logits.
	// Used for non-converged KAN layers (shadow training) or when
	// the backend doesn't support SDPA.
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

// applyAttentionWeights applies softmax with optional KAN shadow training.
//
// This is only reached via the manual attention path, which means either:
//   - No KAN trainer is active (standard softmax)
//   - A non-converged KAN layer needs logit capture for training
//   - The backend doesn't support SDPA
//
// Converged KAN layers go through SDPA with effectiveScale baked into the
// scale factor (handled in AttentionWithVMLA above), so they never reach here.
// This preserves flash attention performance after convergence.
func applyAttentionWeights(ctx ml.Context, kq ml.Tensor) ml.Tensor {
	kanMu.RLock()
	trainer := kanTrainer
	kanMu.RUnlock()

	if trainer == nil {
		return kq.Softmax(ctx)
	}

	layerIdx := ctx.LayerIndex()
	if layerIdx < 0 {
		return kq.Softmax(ctx)
	}

	key := kan.LayerKey(layerIdx)

	// Use standard softmax as output (ground truth for training)
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
