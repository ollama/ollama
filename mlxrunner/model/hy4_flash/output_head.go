package hy4_flash

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type outputHead struct {
	*nn.Linear
	Batched *nn.Linear
}

func loadOutputHead(weight *mlx.Array) nn.LinearLayer {
	if weight.DType() == mlx.DTypeBFloat16 && weight.NumDims() == 2 && (mlx.MetalIsAvailable() || mlx.CUDAIsAvailable()) {
		return &outputHead{Linear: nn.NewLinear(weight.Clone(), nil)}
	}
	return nn.NewLinear(weight.AsType(mlx.DTypeFloat32), nil)
}

func (h *outputHead) Forward(x *mlx.Array) *mlx.Array {
	projection := h.Linear
	if h.Weight.DType() == mlx.DTypeBFloat16 {
		if out, ok := mlx.TryMatvecBF16Weights(x, h.Weight); ok {
			return out
		}
		if h.Batched != nil {
			projection = h.Batched
		}
		// Promote the held handle once on the MLX worker. Ordinary decode
		// holds one view, while DFly keeps BF16 for its per-token proposals.
		if projection.Weight.DType() == mlx.DTypeBFloat16 {
			projection.Weight.Set(projection.Weight.AsType(mlx.DTypeFloat32))
		}
		if h.Batched != nil && x.Size() == h.Weight.Dim(1) {
			// The single-token kernel declined a singleton. Share the FP32
			// weight so the unused BF16 matrix can be released.
			h.Weight.Set(projection.Weight)
		}
	}
	return projection.Forward(x)
}
