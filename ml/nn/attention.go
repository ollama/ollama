package nn

import (
	"fmt"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

// Attention implements scaled dot-product attention for transformer models:
// Attention(Q, K, V) = softmax(QK^T/√d_k)V
//
// Parameters:
//   - ctx: Context for tensor operations
//   - query: Query tensor (Q) with shape [seq_len_q, heads, d_k]
//   - key: Key tensor (K) with shape [seq_len_k, kv_heads, d_k], can be nil to read from cache only
//   - value: Value tensor (V) with shape [seq_len_k, kv_heads, d_v], can be nil to read from cache only
//   - scale: Scaling factor, typically 1/√d_k where d_k is the key dimension
//   - cache: KV cache to store key/value and get past history, can be nil to only use provided key/value
//
// Returns:
//
//	Attention output with shape [d_v, heads, seq_len_q]
func Attention(ctx ml.Context, query, key, value ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	if key != nil && value != nil {
		if query.Dim(2) != key.Dim(2) {
			panic(fmt.Errorf("d_k in attention operation does not match between query(%v) and key(%v)", query.Dim(2), key.Dim(2)))
		}

		if key.Dim(1) != value.Dim(1) {
			panic(fmt.Errorf("kv_heads in attention operation does not match between key(%v) and value(%v)", key.Dim(1), value.Dim(1)))
		}

		if key.Dim(0) != value.Dim(0) {
			panic(fmt.Errorf("seq_len_k in attention operation does not match between key(%v) and value(%v)", key.Dim(0), value.Dim(0)))
		}

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

	// Only use the fast SDPA implementation if we have a cache, since that's what
	// will do any expected backend-specific transformations for us
	if sdpa, ok := query.(ml.ScaledDotProductAttention); ok && cache != nil {
		return sdpa.ScaledDotProductAttention(ctx, key, value, mask, scale)
	} else {
		query = query.Permute(ctx, 1, 0, 2, 3)
		key = key.Permute(ctx, 1, 0, 2, 3)
		value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

		kq := key.MulmatFullPrec(ctx, query)

		kq = kq.Scale(ctx, scale)
		if mask != nil {
			kq = kq.Add(ctx, mask)
		}
		kq = kq.Softmax(ctx)

		kqv := value.Mulmat(ctx, kq)
		return kqv.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	}
}
