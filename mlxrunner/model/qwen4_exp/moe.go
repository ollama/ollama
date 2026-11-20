package qwen4_exp

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type sparseMoE struct {
	Gate       nn.LinearLayer
	GateUp     *model.ExpertProjection
	Down       *model.ExpertProjection
	SharedGate nn.LinearLayer
	Shared     *denseMLP
}

type denseMLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

func (m *denseMLP) Forward(x *mlx.Array) *mlx.Array {
	return m.DownProj.Forward(mlx.SwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}

func (m *Model) loadSparseMoE(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string) (*sparseMoE, error) {
	moe := &sparseMoE{}
	var err error
	if moe.Gate, err = requiredLinear(linears, prefix+".gate"); err != nil {
		return nil, err
	}
	if moe.GateUp, err = model.LoadExpertProjection(tensors, prefix+".experts.gate_up_proj", m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant); err != nil {
		return nil, err
	}
	if moe.Down, err = model.LoadExpertProjection(tensors, prefix+".experts.down_proj", m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant); err != nil {
		return nil, err
	}
	if moe.SharedGate, err = requiredLinear(linears, prefix+".shared_expert_gate"); err != nil {
		return nil, err
	}
	moe.Shared = &denseMLP{
		GateProj: linears.Make(prefix + ".shared_expert.gate_proj"),
		UpProj:   linears.Make(prefix + ".shared_expert.up_proj"),
		DownProj: linears.Make(prefix + ".shared_expert.down_proj"),
	}
	if moe.Shared.GateProj == nil || moe.Shared.UpProj == nil || moe.Shared.DownProj == nil {
		return nil, fmt.Errorf("missing shared expert projection at %s", prefix)
	}
	return moe, nil
}

func (m *sparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.NumExpertsPerTok

	probs := mlx.SoftmaxAxis(m.Gate.Forward(x), -1, true)
	indices := mlx.Argpartition(mlx.Neg(probs), int(topK)-1, -1)
	indices = indices.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(topK)))
	scores := mlx.TakeAlongAxis(probs, indices, -1)
	scores = mlx.Div(scores, mlx.Sum(scores, -1, true))

	xFlat := mlx.Reshape(mlx.ExpandDims(mlx.ExpandDims(x, -2), -2), B*L, 1, 1, cfg.HiddenSize)
	indexFlat := mlx.Reshape(indices, B*L, topK)
	doSort := B*L >= 64
	var inverse *mlx.Array
	if doSort {
		all := mlx.Flatten(indexFlat)
		order := mlx.Argsort(all, 0)
		inverse = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, topK), 0), 1)
		indexFlat = mlx.Reshape(mlx.Take(all, order, 0), B*L*topK, 1)
	}

	gateUp := m.GateUp.Forward(xFlat, indexFlat, doSort)
	mid := gateUp.Dim(-1) / 2
	gate := gateUp.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, mid))
	up := gateUp.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(mid, mlx.End))
	hidden := mlx.SwiGLU(gate, up)
	down := m.Down.Forward(hidden, indexFlat, doSort)
	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), inverse, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}
	differentiated := mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
	out := mlx.Sum(mlx.Mul(differentiated, mlx.ExpandDims(scores, -1)), 2, false)

	shared := m.Shared.Forward(x)
	shared = mlx.Mul(shared, mlx.Sigmoid(m.SharedGate.Forward(x)))
	return mlx.Add(out, shared)
}
