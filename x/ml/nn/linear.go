package nn

import "github.com/ollama/ollama/x/ml"

type Linear struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *Linear) Forward(ctx ml.Context, t ml.Tensor) ml.Tensor {
	t = t.Matmul(ctx, m.Weight.Transpose(ctx))
	if m.Bias != nil {
		t = t.Add(ctx, m.Bias)
	}

	return t
}

type LinearBatch struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *LinearBatch) Forward(ctx ml.Context, t, indices ml.Tensor) ml.Tensor {
	panic("not yet ported")
	// t = m.Weight.MulmatID(ctx, t, indices)
	// if m.Bias != nil {
	// 	t = t.AddID(ctx, m.Bias, indices)
	// }

	// return t
}
