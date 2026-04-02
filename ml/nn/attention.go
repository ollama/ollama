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

// kanLayerCounter tracks attention call count to derive layer index.
// Model architectures don't call ctx.Layer(i) before Attention(), so
// ctx.LayerIndex() is always -1. We use a simple counter that resets
// each generation step (via FlushKANTraining) to assign layer indices.
var (
	kanLayerCounter int
	kanLayerMu      sync.Mutex
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

// GetKANPendingTensors returns all KAN training tensors that need to be
// included in Compute() for data materialization. Call this after the
// forward pass but before Compute(), and pass the returned tensors
// alongside the model output.
func GetKANPendingTensors() []ml.Tensor {
	kanPendingMu.Lock()
	defer kanPendingMu.Unlock()

	var tensors []ml.Tensor
	for _, item := range kanPending {
		if item.kq != nil {
			tensors = append(tensors, item.kq)
		}
		if item.softmax != nil {
			tensors = append(tensors, item.softmax)
		}
	}
	return tensors
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
	// Reset layer counter for the next generation step
	kanLayerMu.Lock()
	kanLayerCounter = 0
	kanLayerMu.Unlock()

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
			logits := item.kq.ReadFloats()

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
				expected := item.softmax.ReadFloats()
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

	// KAN attention: always use SDPA for the actual computation.
	// For training, compute pre-softmax logits separately (just key*query matmul).
	// The manual attention path is broken for cached tensors (double permutation
	// of value), so we never use it.
	kanMu.RLock()
	trainer := kanTrainer
	kanMu.RUnlock()

	kanScale := 1.0
	kanLayerKey := ""
	needTrainingData := false
	if trainer != nil {
		// Derive layer index from call counter since model architectures
		// don't set ctx.Layer(i) before calling Attention.
		kanLayerMu.Lock()
		layerIdx := kanLayerCounter
		kanLayerCounter++
		kanLayerMu.Unlock()

		layerKey := kan.LayerKey(layerIdx)
		kanLayerKey = layerKey
		trainer.GetOrCreateLayer(layerKey)

		if trainer.IsConverged(layerKey) {
			// Converged: use SDPA with the KAN's effective scale.
			kanScale = trainer.GetEffectiveScale(layerKey)

			// Phase 2 self-evolution needs logit data to sharpen attention.
			if trainer.IsPhase2Active(layerKey) && trainer.NeedsPhase2Data(layerKey) {
				needTrainingData = true
			}
		} else {
			// Not converged: collect training data (respects TrainEveryN)
			if trainer.ShouldTrain(layerKey) {
				needTrainingData = true
			}
		}
	}

	// Collect training data by computing kq separately (just key*query matmul).
	// This is lightweight — no value multiplication needed for training.
	if needTrainingData && kanLayerKey != "" {
		qPerm := query.Permute(ctx, 0, 2, 1, 3)
		kPerm := key.Permute(ctx, 0, 2, 1, 3)
		kq := kPerm.MulmatFullPrec(ctx, qPerm)
		kq = kq.Scale(ctx, scale)
		if mask != nil {
			kq = kq.Add(ctx, mask)
		}
		softmaxOut := kq.Softmax(ctx)
		ctx.Forward(kq, softmaxOut)

		converged := trainer.IsConverged(kanLayerKey)
		kanPendingMu.Lock()
		item := kanPendingItem{
			key:       kanLayerKey,
			kq:        kq,
			shape:     kq.Shape(),
			converged: converged,
		}
		if !converged {
			item.softmax = softmaxOut
		}
		kanPending = append(kanPending, item)
		kanPendingMu.Unlock()
	}

	// Always use SDPA for the actual attention computation.
	if sdpa, ok := query.(ml.ScaledDotProductAttention); ok {
		cacheConfigApplied := cache != nil
		return sdpa.ScaledDotProductAttention(ctx, key, value, mask, sinks, vmla, scale*kanScale, cacheConfigApplied)
	}

	// Fallback manual path (should never be reached with GGML backend,
	// but kept for completeness). Note: this does NOT support KAN training
	// because the value permutation is incompatible with cached tensors.
	query = query.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := key.MulmatFullPrec(ctx, query)
	kq = kq.Scale(ctx, scale)
	if mask != nil {
		kq = kq.Add(ctx, mask)
	}
	kq = kq.Softmax(ctx)

	kqv := value.Mulmat(ctx, kq)

	if vmla != nil {
		kqv = vmla.Mulmat(ctx, kqv)
	}

	return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
}

