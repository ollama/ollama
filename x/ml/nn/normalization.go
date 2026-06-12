package nn

import (
	"github.com/ollama/ollama/x/ml"
)

type LayerNorm struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *LayerNorm) Forward(ctx ml.Context, t ml.Tensor, eps float32) ml.Tensor {
	return t.LayerNorm(ctx, m.Weight, m.Bias, eps)
}

type RMSNorm struct {
	Weight ml.Tensor `gguf:"weight"`
}

func (m *RMSNorm) Forward(ctx ml.Context, t ml.Tensor, eps float32) ml.Tensor {
	// slog.Info("RMSNorm", "eps", eps)
	// fmt.Fprintln(os.Stderr, t.ToString())
	// fmt.Fprintln(os.Stderr, m.Weight.ToString())

	// TODO this is probably model specific, not generalized...
	w := m.Weight.Add(ctx, ctx.FromFloats([]float32{1.0}, 1))

	return t.RMSNorm(ctx, w, eps)
}
