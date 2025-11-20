package nn

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/rope"
)

// fastRoPE is an interface for tensors that support fast rotary positional embedding.
type fastRoPE interface {
	RoPE(ctx ml.Context, positions ml.Tensor, dim int, base, scale float32, options ...func(*rope.Options)) ml.Tensor
}

// RoPE applies rotary positional embedding to tensor `t`.
func RoPE(ctx ml.Context, t, positions ml.Tensor, dim int, base, scale float32, options ...func(*rope.Options)) ml.Tensor {
	if t, ok := t.(fastRoPE); ok {
		return t.RoPE(ctx, positions, dim, base, scale, options...)
	}

	panic("RoPE not implemented for this tensor type")
}
