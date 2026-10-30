package qwen3_5

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// UnembedCandidates scores one hidden position in FP32, gathering candidate rows
// before conversion to avoid an FP32 copy of the full vocabulary head.
func (m *Model) UnembedCandidates(hidden, candidates *mlx.Array) *mlx.Array {
	var weight, bias *mlx.Array
	switch head := m.LMHead.(type) {
	case *nn.Linear:
		weight, bias = head.Weight.TakeAxis(candidates, 0), head.Bias
	case *nn.QuantizedLinear:
		var qbiases *mlx.Array
		if head.QBiases != nil {
			qbiases = head.QBiases.TakeAxis(candidates, 0)
		}
		scale := head.GlobalScale
		if scale != nil && scale.Size() > 1 {
			scale = scale.TakeAxis(candidates, 0)
		}
		weight = mlx.Dequantize(head.Weight.TakeAxis(candidates, 0), head.Scales.TakeAxis(candidates, 0), qbiases,
			head.GroupSize, head.Bits, head.Mode, scale)
		bias = head.Bias
	default:
		return m.Unembed(hidden).Reshape(-1).TakeAxis(candidates, 0)
	}
	logits := hidden.AsType(mlx.DTypeFloat32).Matmul(weight.AsType(mlx.DTypeFloat32).Transpose(1, 0)).Reshape(-1)
	if bias != nil {
		logits = logits.Add(bias.TakeAxis(candidates, 0).AsType(mlx.DTypeFloat32))
	}
	return logits
}
