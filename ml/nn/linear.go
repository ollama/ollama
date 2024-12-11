package nn

import "github.com/ollama/ollama/ml"

type Linear struct {
	Weight ml.Tensor `ggml:"weight"`
	Bias   ml.Tensor `ggml:"bias"`
}

func (m *Linear) Forward(ctx ml.Context, t ml.Tensor) ml.Tensor {
	t = m.Weight.Mulmat(ctx, t)
	if m.Bias != nil {
		t = t.Add(ctx, m.Bias)
	}

	return t
}
