package nn

import (
	"github.com/ollama/ollama/ml"
)

type LayerNorm struct {
	Weight ml.Tensor `ggml:"weight"`
	Bias   ml.Tensor `ggml:"bias"`
}

func (m *LayerNorm) Forward(ctx ml.Context, t ml.Tensor, eps float32) ml.Tensor {
	return t.LayerNorm(ctx, m.Weight, m.Bias, eps)
}

type RMSNorm struct {
	Weight ml.Tensor `ggml:"weight"`
	Bias   ml.Tensor `ggml:"bias"`
}

func (m *RMSNorm) Forward(ctx ml.Context, t ml.Tensor, eps float32) ml.Tensor {
	return t.RMSNorm(ctx, m.Weight, m.Bias, eps)
}
