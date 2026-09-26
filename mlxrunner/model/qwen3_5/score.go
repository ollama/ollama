package qwen3_5

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// UnembedCandidates scores one hidden position. For dense heads, gather before
// casting so FP32 accumulation neither rounds logits to BF16 nor allocates an
// FP32 copy of the full vocabulary head. This also covers tied dense embeddings.
func (m *Model) UnembedCandidates(hidden, candidates *mlx.Array) *mlx.Array {
	if head, ok := m.LMHead.(*nn.Linear); ok {
		weight := head.Weight.TakeAxis(candidates, 0).AsType(mlx.DTypeFloat32)
		logits := hidden.AsType(mlx.DTypeFloat32).Matmul(weight.Transpose(1, 0)).Reshape(-1)
		if head.Bias != nil {
			logits = logits.Add(head.Bias.TakeAxis(candidates, 0).AsType(mlx.DTypeFloat32))
		}
		return logits
	}
	// Retain existing quantization, bias, and global-scale handling. Selected-row
	// dequantization can be added separately when its formats are validated.
	return m.Unembed(hidden).Reshape(-1).TakeAxis(candidates, 0)
}
