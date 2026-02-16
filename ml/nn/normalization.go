package nn

import (
	"github.com/ollama/ollama/ml"
)

type LayerNorm struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *LayerNorm) Forward(ctx ml.Context, t ml.Tensor, eps float32) ml.Tensor {
	return t.LayerNorm(ctx, m.Weight, m.Bias, eps)
}

type RMSNorm struct {
	Weight ml.Tensor `gguf:"weight"`
}

func (m *RMSNorm) Forward(ctx ml.Context, t ml.Tensor, eps float32) ml.Tensor {
	return t.RMSNorm(ctx, m.Weight, eps)
}
