package qwen4_exp

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type sparseMoE struct {
	Gate       nn.LinearLayer
	GateUp     *expertProjection
	Down       *expertProjection
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

// expertProjection keeps a packed expert bank and its quantization companions
// together. Unquantized weights are transposed once at load for GatherMM.
type expertProjection struct {
	Weight      *mlx.Array
	Scales      *mlx.Array
	Biases      *mlx.Array
	GlobalScale *mlx.Array
	GroupSize   int
	Bits        int
	Mode        string
}

func (p *expertProjection) Forward(x, indices *mlx.Array, sorted bool) *mlx.Array {
	if p.Scales != nil {
		return mlx.GatherQMM(x, p.Weight, p.Scales, p.Biases,
			nil, indices, true, p.GroupSize, p.Bits, p.Mode, p.GlobalScale, sorted)
	}
	return mlx.GatherMM(x, p.Weight, nil, indices, sorted)
}

// readExpertBank reads one routed-expert projection, stacked or per expert,
// and enforces the model's policy: quantized banks must use a gather kernel
// mode, and only NVFP4 and MXFP8 banks carry a global scale.
func (m *Model) readExpertBank(tensors map[string]*mlx.Array, prefix, proj string) (*model.ExpertBank, error) {
	quant := model.QuantDefaults{GroupSize: m.quantGroup, Bits: m.quantBits, Mode: m.quantMode, TensorQuant: m.tensorQuant}
	bank, err := model.ReadExpertBank(tensors, quant, int(m.NumExperts), proj, prefix+".experts")
	if err != nil || bank == nil {
		return nil, err
	}
	if bank.Quantized() && !model.SupportsGatherQMM(bank.Mode, bank.Bits) {
		return nil, fmt.Errorf("packed expert tensor %q uses unsupported quantization mode=%q bits=%d", bank.Key, bank.Mode, bank.Bits)
	}
	if bank.GlobalScale != nil && bank.Mode != "nvfp4" && bank.Mode != "mxfp8" {
		return nil, fmt.Errorf("expert tensor %q has a global scale but uses mode=%q", bank.Key, bank.Mode)
	}
	return bank, nil
}

// projection returns a bank as the gather kernels consume it; unquantized
// banks are transposed for GatherMM.
func projection(bank *model.ExpertBank) *expertProjection {
	if !bank.Quantized() {
		return &expertProjection{Weight: mlx.Transpose(bank.Weight, 0, 2, 1)}
	}
	return &expertProjection{Weight: bank.Weight, Scales: bank.Scales, Biases: bank.Biases, GlobalScale: bank.GlobalScale, GroupSize: bank.GroupSize, Bits: bank.Bits, Mode: bank.Mode}
}

// fuseGateUp joins separate gate and up banks into the fused gate_up bank the
// forward pass runs, along the output dimension. Quantized banks fuse only
// when their quantization and global scales match.
func fuseGateUp(gate, up *model.ExpertBank) (*model.ExpertBank, error) {
	if gate.Quantized() != up.Quantized() || gate.Mode != up.Mode || gate.Bits != up.Bits || gate.GroupSize != up.GroupSize ||
		(gate.Biases == nil) != (up.Biases == nil) || !model.SameGlobalScales(gate.GlobalScale, up.GlobalScale) {
		return nil, fmt.Errorf("expert gate %q and up %q differ in quantization and cannot be fused", gate.Key, up.Key)
	}
	concat := func(a, b *mlx.Array) *mlx.Array {
		if a == nil {
			return nil
		}
		return mlx.Concatenate([]*mlx.Array{a, b}, 1)
	}
	return &model.ExpertBank{
		Weight: concat(gate.Weight, up.Weight), Scales: concat(gate.Scales, up.Scales), Biases: concat(gate.Biases, up.Biases),
		GlobalScale: gate.GlobalScale, GroupSize: gate.GroupSize, Bits: gate.Bits, Mode: gate.Mode, Key: gate.Key,
	}, nil
}

func (m *Model) loadSparseMoE(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string) (*sparseMoE, error) {
	moe := &sparseMoE{}
	var err error
	if moe.Gate, err = requiredLinear(linears, prefix+".gate"); err != nil {
		return nil, err
	}
	gateUp, err := m.readExpertBank(tensors, prefix, "gate_up_proj")
	if err != nil {
		return nil, err
	}
	if gateUp == nil {
		gate, err := m.readExpertBank(tensors, prefix, "gate_proj")
		if err != nil {
			return nil, err
		}
		up, err := m.readExpertBank(tensors, prefix, "up_proj")
		if err != nil {
			return nil, err
		}
		if gate == nil || up == nil {
			return nil, fmt.Errorf("missing packed expert tensor %q", prefix+".experts.gate_up_proj")
		}
		if gateUp, err = fuseGateUp(gate, up); err != nil {
			return nil, err
		}
	}
	down, err := m.readExpertBank(tensors, prefix, "down_proj")
	if err != nil {
		return nil, err
	}
	if down == nil {
		return nil, fmt.Errorf("missing packed expert tensor %q", prefix+".experts.down_proj")
	}
	moe.GateUp, moe.Down = projection(gateUp), projection(down)
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
