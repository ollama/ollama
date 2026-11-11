package qwen4_exp

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type hyperConnection struct {
	Norm         *streamRMSNorm
	InputMixDown nn.LinearLayer
	InputMixUp   nn.LinearLayer
	BlockInject  nn.LinearLayer
	PackedInput  nn.LinearLayer
	MixDownDim   int32
}

type hyperConnectionState struct {
	residual *mlx.Array
	normed   *mlx.Array
	inject   *mlx.Array
}

func loadHyperConnection(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, withInjection bool, cfg *Config) (*hyperConnection, error) {
	h := &hyperConnection{}
	var err error
	if h.Norm, err = loadStreamNorm(tensors, prefix+".hc_norm", cfg); err != nil {
		return nil, err
	}
	if h.InputMixDown, err = requiredLinear(linears, prefix+".input_mix_weight_down"); err != nil {
		return nil, err
	}
	h.MixDownDim = h.InputMixDown.OutputDim()
	if h.InputMixUp, err = requiredLinear(linears, prefix+".input_mix_weight_up"); err != nil {
		return nil, err
	}
	if withInjection {
		if h.BlockInject, err = requiredLinear(linears, prefix+".block_inject_weight"); err != nil {
			return nil, err
		}
		if packed := packHyperConnectionInput(h.InputMixDown, h.BlockInject); packed != nil {
			h.PackedInput = packed
			h.InputMixDown = nil
			h.BlockInject = nil
		}
	}
	return h, nil
}

func packHyperConnectionInput(down, inject nn.LinearLayer) nn.LinearLayer {
	switch down := down.(type) {
	case *nn.Linear:
		inject, ok := inject.(*nn.Linear)
		if !ok || down.Bias != nil || inject.Bias != nil || down.Weight.Dim(1) != inject.Weight.Dim(1) {
			return nil
		}
		return nn.NewLinear(mlx.Concatenate([]*mlx.Array{down.Weight, inject.Weight}, 0), nil)
	case *nn.QuantizedLinear:
		inject, ok := inject.(*nn.QuantizedLinear)
		if !ok || down.Bias != nil || inject.Bias != nil ||
			down.Scales == nil || inject.Scales == nil ||
			down.QBiases != nil || inject.QBiases != nil ||
			down.GlobalScale != nil || inject.GlobalScale != nil ||
			down.GroupSize != inject.GroupSize || down.Bits != inject.Bits || down.Mode != inject.Mode ||
			down.Weight.NumDims() != 2 || inject.Weight.NumDims() != 2 ||
			down.Scales.NumDims() != 2 || inject.Scales.NumDims() != 2 ||
			down.Weight.Dim(1) != inject.Weight.Dim(1) || down.Scales.Dim(1) != inject.Scales.Dim(1) {
			return nil
		}
		return &nn.QuantizedLinear{
			Weight:    mlx.Concatenate([]*mlx.Array{down.Weight, inject.Weight}, 0),
			Scales:    mlx.Concatenate([]*mlx.Array{down.Scales, inject.Scales}, 0),
			GroupSize: down.GroupSize,
			Bits:      down.Bits,
			Mode:      down.Mode,
		}
	}
	return nil
}

var (
	hyperConnectionSiLU = mlx.Compile2(
		"Qwen4HyperConnectionSiLU",
		func(x, count *mlx.Array) *mlx.Array {
			return mlx.SiLU(mlx.Div(x, count))
		},
		mlx.Shapeless(),
	)
	hyperConnectionMix = mlx.Compile3(
		"Qwen4HyperConnectionMix",
		func(streams, mix, count *mlx.Array) *mlx.Array {
			return mlx.Div(mlx.Sum(mlx.Mul(streams, mlx.Sigmoid(mix)), 2, false), count)
		},
		mlx.Shapeless(),
	)
	hyperConnectionInject = mlx.Compile(
		"Qwen4HyperConnectionInject",
		func(in ...*mlx.Array) []*mlx.Array {
			residual, branch, logits, count := in[0], in[1], in[2], in[3]
			weight := mlx.Sigmoid(mlx.Div(logits, count))
			two := mlx.FromValue(float32(2)).AsType(weight.DType())
			weight = mlx.Mul(weight, two)
			update := mlx.Mul(mlx.ExpandDims(weight, -1), mlx.ExpandDims(branch, -2))
			return []*mlx.Array{mlx.Add(residual, update)}
		},
		mlx.Shapeless(),
	)
)

func expandStreams(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	expanded := mlx.Tile(mlx.ExpandDims(x, -2), []int32{1, 1, cfg.HCCount, 1})
	return mlx.Reshape(expanded, B, L, cfg.HCCount*cfg.HiddenSize)
}

// Prepare reduces the residual streams into one branch input and preserves
// the state needed to inject the branch output.
func (h *hyperConnection) Prepare(residual *mlx.Array, cfg *Config) (*mlx.Array, hyperConnectionState) {
	dims := residual.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	normed := h.Norm.Forward(residual, cfg.RMSNormEps)
	var down, inject *mlx.Array
	if h.PackedInput != nil {
		packed := h.PackedInput.Forward(normed)
		down = packed.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(h.MixDownDim)))
		inject = packed.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(int(h.MixDownDim), mlx.End))
	} else {
		down = h.InputMixDown.Forward(normed)
	}
	count := mlx.NewScalarArray(float32(cfg.HCCount)).AsType(down.DType())
	mixHidden := hyperConnectionSiLU(down, count)
	mix := h.InputMixUp.Forward(mixHidden)
	mix = mlx.Reshape(mix, B, L, cfg.HCCount, cfg.HiddenSize)
	streams := mlx.Reshape(normed, B, L, cfg.HCCount, cfg.HiddenSize)
	count = mlx.NewScalarArray(float32(cfg.HCCount)).AsType(streams.DType())
	branch := hyperConnectionMix(streams, mix, count)
	return branch, hyperConnectionState{residual: residual, normed: normed, inject: inject}
}

// Inject broadcasts one branch result back into the residual streams. The
// public HC post-connection uses 2*sigmoid so an untrained coefficient is one;
// block_inject_weight has exactly one dynamic coefficient per stream.
func (h *hyperConnection) Inject(state hyperConnectionState, branch *mlx.Array, cfg *Config) *mlx.Array {
	dims := branch.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	logits := state.inject
	if logits == nil {
		logits = h.BlockInject.Forward(state.normed)
	}
	count := mlx.NewScalarArray(float32(cfg.HCCount)).AsType(logits.DType())
	residual := mlx.Reshape(state.residual, B, L, cfg.HCCount, cfg.HiddenSize)
	out := hyperConnectionInject(residual, branch, logits, count)[0]
	return mlx.Reshape(out, B, L, cfg.HCCount*cfg.HiddenSize)
}

// Reduce applies the trained input mixer as the final stream reduction. The
// final checkpoint mixer has no block injection tensor, matching this role.
func (h *hyperConnection) Reduce(residual *mlx.Array, cfg *Config) *mlx.Array {
	branch, _ := h.Prepare(residual, cfg)
	return branch
}

type streamRMSNorm struct {
	Weight *mlx.Array
}

func loadStreamNorm(tensors map[string]*mlx.Array, name string, cfg *Config) (*streamRMSNorm, error) {
	weight, err := requiredArray(tensors, name+".weight", name)
	if err != nil {
		return nil, err
	}
	if weight.Size() != int(cfg.HCCount*cfg.HiddenSize) {
		return nil, fmt.Errorf("%s has %d elements, want %d", name, weight.Size(), cfg.HCCount*cfg.HiddenSize)
	}
	weight = mlx.Reshape(mlx.AddScalar(weight, 1), cfg.HCCount, cfg.HiddenSize)
	return &streamRMSNorm{Weight: weight}, nil
}

var streamRMSNormFn = mlx.Compile(
	"Qwen4StreamRMSNorm",
	func(in ...*mlx.Array) []*mlx.Array {
		x, weight, eps := in[0], in[1], in[2]
		dtype := x.DType()
		values := x.AsType(mlx.DTypeFloat32)
		meanSquare := mlx.Mean(mlx.Mul(values, values), -1, true)
		invRMS := mlx.Div(mlx.NewScalarArray(1), mlx.Add(meanSquare, eps).Sqrt())
		return []*mlx.Array{mlx.Mul(mlx.Mul(values, invRMS), weight.AsType(mlx.DTypeFloat32)).AsType(dtype)}
	},
	mlx.Shapeless(),
)

func (n *streamRMSNorm) Forward(x *mlx.Array, eps float32) *mlx.Array {
	originalShape := x.Dims()
	if originalShape[len(originalShape)-1] == n.Weight.Size() {
		shape := append([]int(nil), originalShape[:len(originalShape)-1]...)
		shape = append(shape, n.Weight.Dim(0), n.Weight.Dim(1))
		x = x.Reshape(shape...)
	}
	epsArray := mlx.NewScalarArray(eps)
	normalized := streamRMSNormFn(x, n.Weight, epsArray)[0]
	return normalized.Reshape(originalShape...)
}
