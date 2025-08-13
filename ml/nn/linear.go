package nn

import "github.com/ollama/ollama/ml"

type Linear struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *Linear) Forward(ctx ml.Context, t ml.Tensor) ml.Tensor {
	t = m.Weight.Mulmat(ctx, t)
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
	t = m.Weight.MulmatID(ctx, t, indices)
	if m.Bias != nil {
		var bias ml.Tensor
		if len(indices.Shape()) > 1 {
			// FIXME: Rows does not support 2D indices for a 2D input tensor so reshape indices to 1D.
			bias = m.Bias.Rows(ctx, indices.Contiguous(ctx, indices.Dim(0)*indices.Dim(1))).
				Duplicate(ctx).
				Reshape(ctx, m.Bias.Dim(0), indices.Dim(0), indices.Dim(1))
		} else {
			bias = m.Bias.Rows(ctx, indices)
		}
		t = t.Add(ctx, bias)
	}

	return t
}
