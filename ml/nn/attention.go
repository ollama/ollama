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

// Attention implements scaled dot-product attention for transformer models:
// Attention(Q, K, V) = softmax(QK^T/√d_k)V
//
// When KAN attention is enabled, this also shadow-trains a Geometric KAN
// to replace softmax. After convergence, the KAN is hot-swapped in.
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

	if sdpa, ok := query.(ml.ScaledDotProductAttention); ok {
		// Flash attention path: softmax is fused in the kernel.
		// KAN cannot operate here since logits are never materialized.
		cacheConfigApplied := cache != nil
		return sdpa.ScaledDotProductAttention(ctx, key, value, mask, sinks, vmla, scale, cacheConfigApplied)
	}

	// Non-flash attention path: we have access to pre-softmax logits.
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

// applyAttentionWeights applies either softmax or KAN attention weights.
//
// When no KAN trainer is active, this is just kq.Softmax(ctx).
//
// When KAN is active:
//   - If the layer's KAN has converged and hot-swap is enabled, use KAN output
//   - Otherwise, compute softmax (for ground truth) and shadow-train the KAN
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

	// If converged and hot-swap enabled, use KAN directly on GPU tensors
	// by pulling logits to CPU, running KAN, pushing back.
	// If Phase 2 is active, the KAN also evolves its weights on this forward pass.
	if trainer.IsConverged(key) {
		return applyKANAttention(ctx, kq, trainer, key)
	}

	// Compute softmax as ground truth
	softmaxOut := kq.Softmax(ctx)

	// Shadow-train KAN in the background (CPU-side)
	if trainer.ShouldTrain(key) {
		go shadowTrainKAN(kq, softmaxOut, trainer, key)
	}

	return softmaxOut
}

// applyKANAttention runs the converged KAN on attention logits.
// Pulls logits to CPU, runs KAN forward pass, pushes result back to GPU.
//
// If Phase 2 self-evolution is active, this also triggers a background
// adaptation step that shifts the KAN weights toward sharper attention
// patterns, bounded by a drift safety rail from the graduation checkpoint.
func applyKANAttention(ctx ml.Context, kq ml.Tensor, trainer *kan.ShadowTrainer, key string) ml.Tensor {
	shape := kq.Shape()
	logits := kq.Floats()

	kanLayer := trainer.GetOrCreateLayer(key)

	// Determine seqK and seqQ from tensor shape
	// kq shape after permute: [seqK, heads, seqQ] or similar
	seqK := 1
	seqQ := 1
	if len(shape) >= 1 {
		seqK = shape[0]
	}
	if len(shape) >= 3 {
		seqQ = shape[2]
	}

	// Phase 2: self-evolution -- adapt weights in the background
	if trainer.IsPhase2Active(key) {
		go trainer.Phase2Step(key, logits, seqK, seqQ)
	}

	// Run KAN forward pass on CPU
	result := kanLayer.Forward(logits, seqK, seqQ)

	// Push back to GPU
	return ctx.FromFloats(result, shape...)
}

// shadowTrainKAN runs one KAN training step on CPU.
// Called as a goroutine to avoid blocking the GPU pipeline.
func shadowTrainKAN(kq, softmaxOut ml.Tensor, trainer *kan.ShadowTrainer, key string) {
	defer func() {
		if r := recover(); r != nil {
			slog.Debug("KAN shadow training recovered from panic", "layer", key, "error", r)
		}
	}()

	shape := kq.Shape()
	logits := kq.Floats()
	expected := softmaxOut.Floats()

	seqK := 1
	seqQ := 1
	if len(shape) >= 1 {
		seqK = shape[0]
	}
	if len(shape) >= 3 {
		seqQ = shape[2]
	}

	trainer.TrainStep(key, logits, expected, seqK, seqQ)
}
