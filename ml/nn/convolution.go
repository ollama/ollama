package nn

import "github.com/ollama/ollama/ml"

type Conv2D struct {
	Weight ml.Tensor `gguf:"weight"`
}

func (m *Conv2D) Forward(ctx ml.Context, t ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return m.Weight.Conv2D(ctx, t, s0, s1, p0, p1, d0, d1)
}
