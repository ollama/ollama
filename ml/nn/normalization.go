package nn

import (
	"github.com/ollama/ollama/ml"
)

type LayerNorm struct {
	Weight ml.Tensor `ggml:"weight"`
	Bias   ml.Tensor `ggml:"bias"`
}

func (m *LayerNorm) Forward(ctx ml.Context, t ml.Tensor, eps float32) ml.Tensor {
	t = t.Norm(ctx, eps).Mul(ctx, m.Weight)
	if m.Bias != nil {
		t = t.Add(ctx, m.Bias)
	}

	return t
}

type RMSNorm struct {
	Weight ml.Tensor `ggml:"weight"`
	Bias   ml.Tensor `ggml:"bias"`
}

func (m *RMSNorm) Forward(ctx ml.Context, t ml.Tensor, eps float32) ml.Tensor {
	t = t.RMSNorm(ctx, eps).Mul(ctx, m.Weight)
	if m.Bias != nil {
		t = t.Add(ctx, m.Bias)
	}

	return t
}
