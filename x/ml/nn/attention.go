package nn

import (
	"fmt"

	"github.com/ollama/ollama/x/kvcache"
	"github.com/ollama/ollama/x/ml"
)

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

	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"q": query, "k": key, "v": value}, true)
	// panic("after cache get") //
	// 2025/12/10 16:02:33 INFO XXX tensors are similar q=0.9999869465827942 shape="[1 8 13 256]" min_difference=[-0.07926178] max_difference=[0.07012844]
	// 2025/12/10 16:02:33 INFO XXX tensors are similar k=0.9999891519546509 shape="[1 4 13 256]" min_difference=[-0.21365738] max_difference=[0.19916534]
	// 2025/12/10 16:02:33 INFO XXX tensors are similar v=0.9999960660934448 shape="[1 4 13 256]" min_difference=[-0.32923126] max_difference=[0.32646942]

	// var mask ml.Tensor
	if cache != nil {
		key, value, _ = cache.Get(ctx)
	}
	// ctx.CompareWith("/tmp/test", map[string]ml.Tensor{"q": query.Contiguous(ctx, false), "k": key.Contiguous(ctx, false), "v": value.Contiguous(ctx, false)}, true)
	// panic("after cache get") //
	// 2025/12/10 15:34:03 INFO XXX tensors are similar q=0.9999869465827942 shape="[1 8 13 256]" min_difference=[-0.07926178] max_difference=[0.07012844]
	// 2025/12/10 15:34:03 INFO XXX tensors are similar k=0.9999881982803345 shape="[1 4 13 256]" min_difference=[-0.25] max_difference=[0.25]
	// 2025/12/10 15:34:03 INFO XXX tensors are similar v=0.9999913573265076 shape="[1 4 13 256]" min_difference=[-0.5] max_difference=[0.5]

	// Only use the fast SDPA implementation if we have a cache, since that's what
	// will do any expected backend-specific transformations for us

	if cache != nil {
		// TODO what to do with vmla?
		// return query.Transpose(ctx, 0, 2, 1, 3).ScaledDotProductAttention(ctx, key.Transpose(ctx, 0, 2, 1, 3), value.Transpose(ctx, 0, 2, 1, 3), scale, "array", mask, sinks)
		return query.ScaledDotProductAttention(ctx, key, value, scale, "causal", nil, sinks)

		// TODO these two produce identical output, but not similar enough - 92.9% - should be 99.999%
	} else {
		panic("else case not supported")
		// TODO transpose shapes are wrong
		// key = key.Transpose(ctx, 0, 2, 1, 3)
		// value = value.Transpose(ctx, 1, 2, 0, 3).Contiguous(ctx, false)

		// kq := query.Matmul(ctx, key)

		// kq = kq.Scale(ctx, scale)
		// if mask != nil {
		// 	kq = kq.Add(ctx, mask)
		// }
		// kq = kq.Softmax(ctx)

		// kqv := kq.Matmul(ctx, value)

		// if vmla != nil {
		// 	kqv = kqv.Matmul(ctx, vmla)
		// }

		// return kqv.Transpose(ctx, 0, 2, 1, 3).Contiguous(ctx, false)
	}
}
