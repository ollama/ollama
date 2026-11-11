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

func supportsGatherQMM(mode string, bits int) bool {
	switch mode {
	case "affine":
		return bits == 4 || bits == 8
	case "mxfp8":
		return bits == 8
	case "nvfp4", "mxfp4":
		return bits == 4
	default:
		return false
	}
}

func (m *Model) loadExpertProjection(tensors map[string]*mlx.Array, base string) (*expertProjection, error) {
	p := &expertProjection{}
	var key string
	for _, candidate := range []string{base, base + ".weight"} {
		if p.Weight = tensors[candidate]; p.Weight != nil {
			key = candidate
			break
		}
	}
	if p.Weight == nil {
		return nil, fmt.Errorf("missing packed expert tensor %q", base)
	}
	p.Scales = tensors[key+"_scale"]
	if p.Scales == nil {
		p.Weight = mlx.Transpose(p.Weight, 0, 2, 1)
	} else {
		p.Biases = tensors[key+"_qbias"]
		p.GroupSize, p.Bits, p.Mode = model.ResolveLinearQuantParams(m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant, key, p.Weight, p.Scales)
		if !supportsGatherQMM(p.Mode, p.Bits) {
			return nil, fmt.Errorf("packed expert tensor %q uses unsupported quantization mode=%q bits=%d", key, p.Mode, p.Bits)
		}
	}
	var err error
	p.GlobalScale, err = loadExpertGlobalScale(tensors, base, p.Weight.Dim(0), p.Mode)
	if err != nil {
		return nil, err
	}
	return p, nil
}

func (m *Model) loadSparseMoE(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string) (*sparseMoE, error) {
	moe := &sparseMoE{}
	var err error
	if moe.Gate, err = requiredLinear(linears, prefix+".gate"); err != nil {
		return nil, err
	}
	if moe.GateUp, err = m.loadExpertProjection(tensors, prefix+".experts.gate_up_proj"); err != nil {
		return nil, err
	}
	if moe.Down, err = m.loadExpertProjection(tensors, prefix+".experts.down_proj"); err != nil {
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

func loadExpertGlobalScale(tensors map[string]*mlx.Array, base string, experts int, mode string) (*mlx.Array, error) {
	scale, _ := model.ReadGlobalScale(tensors, base, base+".weight")
	if scale == nil {
		return nil, nil
	}
	if mode != "nvfp4" {
		return nil, fmt.Errorf("expert tensor %q has a global scale but uses mode=%q", base, mode)
	}
	if scale.Size() != 1 && scale.Size() != experts {
		return nil, fmt.Errorf("expert tensor %q has %d global scales for %d experts", base, scale.Size(), experts)
	}
	return model.PrepareGatherQMMGlobalScale(scale, experts), nil
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
