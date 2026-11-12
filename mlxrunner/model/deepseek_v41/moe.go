package deepseek_v41

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type expertGate struct {
	Weight    *mlx.Array
	Bias      *mlx.Array
	ImageBias *mlx.Array
	TopK      int32
	Scale     float32
}

// route applies the correction bias only while selecting experts. The
// returned mixture weights come from the original, unbiased scores.
func (g *expertGate) route(x, imageMask *mlx.Array) (weights, indices *mlx.Array) {
	logits := mlx.Matmul(x.AsType(mlx.DTypeFloat32), mlx.Transpose(g.Weight.AsType(mlx.DTypeFloat32), 1, 0))
	scores := mlx.Softplus(logits).Sqrt()
	bias := g.Bias
	if imageMask != nil && g.ImageBias != nil {
		bias = mlx.Where(mlx.ExpandDims(imageMask, -1), g.ImageBias, bias)
	}
	selected := mlx.Add(scores, bias)
	indices = mlx.Argpartition(mlx.Neg(selected), int(g.TopK)-1, -1)
	indices = indices.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(g.TopK)))
	weights = mlx.TakeAlongAxis(scores, indices, -1)
	if g.TopK > 1 {
		weights = mlx.Div(weights, mlx.AddScalar(mlx.Sum(weights, -1, true), 1e-20))
	}
	return mlx.MulScalar(weights, g.Scale), indices
}

type expertBank struct {
	Weight *mlx.Array
	Scale  *mlx.Array
}

func (b expertBank) apply(x, indices *mlx.Array) *mlx.Array {
	return mlx.GatherQMM(x, b.Weight, b.Scale, nil, nil, indices, true, 32, 4, "mxfp4", nil, false)
}

type mixtureOfExperts struct {
	Gate   *expertGate
	Up     expertBank
	GateUp expertBank
	Down   expertBank
	Shared []nn.LinearLayer
	Limit  float32
}

func (m *mixtureOfExperts) forward(x, imageMask *mlx.Array) *mlx.Array {
	dims := x.Dims()
	batch, length, width := int32(dims[0]), int32(dims[1]), int32(dims[2])
	flat := mlx.Reshape(x, batch*length, width)
	var flatMask *mlx.Array
	if imageMask != nil {
		flatMask = mlx.Reshape(imageMask, batch*length)
	}
	weights, indices := m.Gate.route(mlx.Reshape(flat, 1, batch*length, width), flatMask)
	indices = mlx.Reshape(indices, batch*length, m.Gate.TopK)
	weights = mlx.Reshape(weights, batch*length, m.Gate.TopK)
	selected := mlx.Reshape(flat, batch*length, 1, 1, width)
	gate := m.GateUp.apply(selected, indices).AsType(mlx.DTypeFloat32)
	up := m.Up.apply(selected, indices).AsType(mlx.DTypeFloat32)
	gate = mlx.Minimum(gate, mlx.FromValue(m.Limit))
	up = mlx.Clip(up, mlx.FromValue(-m.Limit), mlx.FromValue(m.Limit))
	hidden := mlx.Mul(mlx.Mul(gate, mlx.Sigmoid(gate)), up).AsType(x.DType())
	out := m.Down.apply(hidden, indices)
	out = mlx.Squeeze(out, 2)
	out = mlx.Sum(mlx.Mul(out.AsType(mlx.DTypeFloat32), mlx.ExpandDims(weights, -1)), 1, false)

	sharedGate := m.Shared[0].Forward(x).AsType(mlx.DTypeFloat32)
	sharedUp := m.Shared[2].Forward(x).AsType(mlx.DTypeFloat32)
	sharedGate = mlx.Minimum(sharedGate, mlx.FromValue(m.Limit))
	sharedUp = mlx.Clip(sharedUp, mlx.FromValue(-m.Limit), mlx.FromValue(m.Limit))
	sharedHidden := mlx.Mul(mlx.Mul(sharedGate, mlx.Sigmoid(sharedGate)), sharedUp).AsType(x.DType())
	shared := m.Shared[1].Forward(sharedHidden)
	return mlx.Add(mlx.Reshape(out.AsType(x.DType()), batch, length, width), shared)
}

func loadExpertBank(tensors map[string]*mlx.Array, name string, experts, output, input int) (expertBank, error) {
	weight, scale := tensors[name+".weight"], tensors[name+".weight_scale"]
	if weight == nil || scale == nil {
		return expertBank{}, fmt.Errorf("missing packed expert bank %s", name)
	}
	if weight.DType() != mlx.DTypeUint32 || scale.DType() != mlx.DTypeUint8 || input%32 != 0 {
		return expertBank{}, fmt.Errorf("expert bank %s requires U32 MXFP4 weights and U8 scales", name)
	}
	if got, want := weight.Dims(), []int{experts, output, input / 8}; len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		return expertBank{}, fmt.Errorf("expert bank %s has shape %v, want %v", name, got, want)
	}
	if got, want := scale.Dims(), []int{experts, output, input / 32}; len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		return expertBank{}, fmt.Errorf("expert scale %s has shape %v, want %v", name, got, want)
	}
	return expertBank{Weight: weight, Scale: scale}, nil
}
