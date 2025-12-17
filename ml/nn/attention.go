package nn

import (
	"log"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/attention"
)

type fastAttention interface {
	SDPA(ctx ml.Context, key, value ml.Tensor, opts ...func(*attention.Options)) ml.Tensor
}

// Attention implements scaled dot-product attention for transformer models:
// Attention(Q, K, V) = softmax(QK^T/√d_k)V
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

func Attention(ctx ml.Context, query, key, value ml.Tensor, cache kvcache.Cache, fns ...func(*attention.Options)) ml.Tensor {
	if key != nil && value != nil {
		if query.Dim(0) != key.Dim(0) {
			log.Fatalf("d_k in attention operation does not match between query(%v) and key(%v)", query.Dim(0), key.Dim(0))
		}

		if key.Dim(1) != value.Dim(1) {
			log.Fatalf("kv_heads in attention operation does not match between key(%v) and value(%v)", key.Dim(1), value.Dim(1))
		}

		if key.Dim(2) != value.Dim(2) {
			log.Fatalf("seq_len_k in attention operation does not match between key(%v) and value(%v)", key.Dim(2), value.Dim(2))
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

	if t, ok := query.(fastAttention); ok {
		return t.SDPA(ctx, key, value, append([]func(*attention.Options){
			attention.WithMask(mask),
			func(opts *attention.Options) { opts.Cached = cache != nil },
		}, fns...)...)
	}

	panic("Attention not implemented for this tensor type")
}
