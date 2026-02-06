package mlx

import "cmp"

type Quantization struct {
	Scales    Array  `weight:"scales"`
	Biases    Array  `weight:"biases"`
	GroupSize int    `json:"group_size"`
	Bits      int    `json:"bits"`
	Mode      string `json:"mode"`
}

type Linear struct {
	Weight Array `weight:"weight"`
	Bias   Array `weight:"bias"`

	Quantization
}

// Forward computes the linear transformation: x @ Weight.T + Bias
func (m Linear) Forward(x *Array) *Array {
	if m.Scales.Valid() {
		x = x.QuantizedMatmul(
			&m.Weight,
			&m.Scales,
			&m.Biases,
			true,
			m.GroupSize,
			m.Bits,
			cmp.Or(m.Mode, "affine"),
		)
		if m.Bias.Valid() {
			x = m.Bias.Add(x)
		}
		return x
	}

	w := m.Weight.Transpose(1, 0)
	if m.Bias.Valid() {
		return m.Bias.Addmm(x, w, 1.0, 1.0)
	}

	return x.Matmul(w)
}

func (m Linear) Gather(x, lhs, rhs *Array, sorted bool) *Array {
	if m.Scales.Valid() {
		x = x.GatherQMM(
			&m.Weight,
			&m.Scales,
			&m.Biases,
			lhs,
			rhs,
			sorted,
			m.GroupSize,
			m.Bits,
			cmp.Or(m.Mode, "affine"),
			sorted,
		)
		if m.Bias.Valid() {
			x = m.Bias.Add(x)
		}
		return x
	} else {
		w := m.Weight.Transpose(0, 2, 1)
		x = x.GatherMM(w, lhs, rhs, sorted)
	}

	if m.Bias.Valid() {
		x = m.Bias.Add(x)
	}

	return x
}

type Embedding struct {
	Weight Array `weight:"weight"`

	Quantization
}

func (e *Embedding) Forward(indices *Array) *Array {
	if e.Scales.Valid() {
		w := e.Weight.TakeAxis(indices, 0)
		return w.Dequantize(
			e.Scales.TakeAxis(indices, 0),
			e.Biases.TakeAxis(indices, 0),
			e.GroupSize,
			e.Bits,
			cmp.Or(e.Mode, "affine"),
		)
	}

	return e.Weight.TakeAxis(indices, 0)
}

func (e *Embedding) AsLinear() Linear {
	return Linear{
		Weight:       e.Weight,
		Quantization: e.Quantization,
	}
}
