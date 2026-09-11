package nn

import "github.com/ollama/ollama/x/ml"

type Conv2D struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *Conv2D) Forward(ctx ml.Context, t ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	t = m.Weight.Conv2D(ctx, t, s0, s1, p0, p1, d0, d1, 1)
	if m.Bias != nil {
		// Bias shape is (out_channels,) while t shape is (width, height, out_channels, batch)
		t = t.Add(ctx, m.Bias.Reshape(ctx, 1, 1, -1))
	}
	return t
}

type Conv3D struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (m *Conv3D) Forward(ctx ml.Context, t ml.Tensor, s0, s1, s2, p0, p1, p2, d0, d1, d2, g int) ml.Tensor {
	t = m.Weight.Conv3D(ctx, t, s0, s1, s2, p0, p1, p2, d0, d1, d2, g)
	if m.Bias != nil {
		t = t.Add(ctx, m.Bias)
	}
	return t
}
