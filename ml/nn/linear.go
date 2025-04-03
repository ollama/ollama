package nn

import "github.com/ollama/ollama/ml"

type Linear struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *Linear) Forward(ctx ml.Context, t ml.Tensor) ml.Tensor {
	t = t.Matmul(ctx, m.Weight.Permute(ctx, 1, 0, 2, 3))
	if m.Bias != nil {
		t = t.Add(ctx, m.Bias)
	}

	return t
}
