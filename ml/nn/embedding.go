package nn

import "github.com/ollama/ollama/ml"

type Embedding struct {
	Weight ml.Tensor `ggml:"weight"`
}

func (m *Embedding) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	return m.Weight.Rows(ctx, hiddenState)
}
