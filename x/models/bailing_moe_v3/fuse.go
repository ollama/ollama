package bailing_moe_v3

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// fuseLinearRows concatenates linear layers that share the same input into one
// layer whose output rows are the sources' rows back to back, so N thin
// matmuls per forward become one. Affine and mx quantization groups run along
// the input axis within each output row, so concatenating rows (weights,
// scales, and qbiases alike) is bit-exact. Returns nil when the layers'
// representations are not uniform enough to concatenate.
func fuseLinearRows(layers ...nn.LinearLayer) nn.LinearLayer {
	if len(layers) < 2 {
		return nil
	}
	switch layers[0].(type) {
	case *nn.Linear:
		weights := make([]*mlx.Array, len(layers))
		for i, layer := range layers {
			l, ok := layer.(*nn.Linear)
			if !ok || l.Bias != nil {
				return nil
			}
			weights[i] = l.Weight
		}
		for _, w := range weights[1:] {
			if w.DType() != weights[0].DType() {
				return nil
			}
		}
		fused := mlx.Concatenate(weights, 0)
		mlx.Eval(fused)
		return &nn.Linear{Weight: fused}

	case *nn.QuantizedLinear:
		base, ok := layers[0].(*nn.QuantizedLinear)
		if !ok {
			return nil
		}
		weights := make([]*mlx.Array, len(layers))
		scales := make([]*mlx.Array, len(layers))
		var qbiases []*mlx.Array
		if base.QBiases != nil {
			qbiases = make([]*mlx.Array, len(layers))
		}
		for i, layer := range layers {
			l, ok := layer.(*nn.QuantizedLinear)
			if !ok || l.Bias != nil || l.GlobalScale != nil ||
				l.GroupSize != base.GroupSize || l.Bits != base.Bits || l.Mode != base.Mode ||
				(l.QBiases == nil) != (base.QBiases == nil) ||
				l.Weight.DType() != base.Weight.DType() || l.Scales.DType() != base.Scales.DType() {
				return nil
			}
			weights[i] = l.Weight
			scales[i] = l.Scales
			if qbiases != nil {
				qbiases[i] = l.QBiases
			}
		}
		fused := &nn.QuantizedLinear{
			Weight:    mlx.Concatenate(weights, 0),
			Scales:    mlx.Concatenate(scales, 0),
			GroupSize: base.GroupSize,
			Bits:      base.Bits,
			Mode:      base.Mode,
		}
		toEval := []*mlx.Array{fused.Weight, fused.Scales}
		if qbiases != nil {
			fused.QBiases = mlx.Concatenate(qbiases, 0)
			toEval = append(toEval, fused.QBiases)
		}
		mlx.Eval(toEval...)
		return fused
	}
	return nil
}

// fuseGateUp row-fuses GateProj and UpProj and, on success, drops the
// originals so the loader's pin pass does not keep both copies resident.
func (m *DenseMLP) fuseGateUp() {
	m.GateUpProj = fuseLinearRows(m.GateProj, m.UpProj)
	if m.GateUpProj != nil {
		m.GateProj, m.UpProj = nil, nil
	}
}

// sliceCols returns x[..., start:stop] over the last axis of a rank-3 tensor.
func sliceCols(x *mlx.Array, B, L, start, stop int32) *mlx.Array {
	return mlx.SliceStartStop(x, []int32{0, 0, start}, []int32{B, L, stop})
}
