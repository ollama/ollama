package nn

import (
	"github.com/ollama/ollama/ml"
)

func Attention(ctx ml.Context, query, key, value, mask ml.Tensor, scale float64) ml.Tensor {
	if sdpa, ok := query.(ml.ScaledDotProductAttention); ok {
		return sdpa.ScaledDotProductAttention(ctx, key, value, mask, scale)
	} else {
		kq := key.MulmatFullPrec(ctx, query)

		kq = kq.Scale(ctx, scale)
		if mask != nil {
			kq = kq.Add(ctx, mask)
		}
		kq = kq.Softmax(ctx)

		kqv := value.Mulmat(ctx, kq)
		return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	}
}
