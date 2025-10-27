package nn

import "github.com/ollama/ollama/ml"

type Conv2D struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *Conv2D) Forward(ctx ml.Context, t ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	t = m.Weight.Conv2D(ctx, t, s0, s1, p0, p1, d0, d1)
	if m.Bias != nil {
		bias := m.Bias
		// Broadcast bias along spatial dimensions to match convolution output layout.
		bias = bias.Reshape(ctx, 1, 1, bias.Dim(0), 1)
		t = t.Add(ctx, bias)
	}
	return t
}

type Conv3D struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *Conv3D) Forward(ctx ml.Context, t ml.Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) ml.Tensor {
	t = m.Weight.Conv3D(ctx, t, c, s0, s1, s2, p0, p1, p2, d0, d1, d2)
	if m.Bias != nil {
		t = t.Add(ctx, m.Bias)
	}
	return t
}
