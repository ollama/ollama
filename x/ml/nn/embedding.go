package nn

import "github.com/ollama/ollama/x/ml"

type Embedding struct {
	Weight ml.Tensor `gguf:"weight"`
}

func (m *Embedding) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	return m.Weight.TakeAxes(ctx, hiddenState, 0)
}
