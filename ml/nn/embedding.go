package nn

import "github.com/ollama/ollama/ml"

type Embedding struct {
	Weight ml.Tensor `gguf:"weight"`
}

func (m *Embedding) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	return m.Weight.Rows(ctx, hiddenState)
}
