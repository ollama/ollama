package mlx

type Linear struct {
	Weight Tensor `weight:"weight"`
	Bias   Tensor `weight:"bias"`
}

// Forward computes the linear transformation: x @ Weight.T + Bias
func (m Linear) Forward(x *Tensor) *Tensor {
	w := m.Weight.Transpose(1, 0)
	if m.Bias.Valid() {
		return m.Bias.Addmm(x, w, 1.0, 1.0)
	}

	return x.Matmul(w)
}

type Embedding struct {
	Weight Tensor `weight:"weight"`
}

func (e *Embedding) Forward(indices *Tensor) *Tensor {
	return e.Weight.TakeAxis(indices, 0)
}

func (e *Embedding) AsLinear() Linear {
	return Linear{
		Weight: e.Weight,
	}
}
