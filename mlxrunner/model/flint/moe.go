package flint

import (
	"fmt"
	"slices"
	"sync"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type MLPBlock interface {
	Forward(*mlx.Array, *Config) *mlx.Array
}

// The fused activation retains float32 precision until its BF16 output.
var flintSwiGLU = mlx.Compile2("FlintSwiGLU", func(gate, up *mlx.Array) *mlx.Array {
	dtype := gate.DType()
	return mlx.Mul(mlx.SiLU(gate.AsType(mlx.DTypeFloat32)), up.AsType(mlx.DTypeFloat32)).AsType(dtype)
}, mlx.Shapeless())

var flintExpertWeightedSum = mlx.Compile2("FlintExpertWeightedSum", func(selected, weights *mlx.Array) *mlx.Array {
	dtype := selected.DType()
	weighted := mlx.Mul(selected.AsType(mlx.DTypeFloat32), mlx.ExpandDims(weights, -1))
	return mlx.Sum(weighted, 1, false).AsType(dtype)
}, mlx.Shapeless())

// Keep the routed, shared, and residual additions in one graph while
// preserving Flint's BF16 rounding after the routed sum.
var flintExpertWeightedSumAdd = mlx.Compile("FlintExpertWeightedSumAdd", func(in ...*mlx.Array) []*mlx.Array {
	selected, weights, shared, residual := in[0], in[1], in[2], in[3]
	weighted := mlx.Mul(selected.AsType(mlx.DTypeFloat32), mlx.ExpandDims(weights, -1))
	routed := mlx.Sum(weighted, 1, false).AsType(selected.DType())
	combined := mlx.Add(routed, shared).AsType(shared.DType())
	return []*mlx.Array{mlx.Add(residual, combined)}
}, mlx.Shapeless())

// Cache one compiled router per configured top-k across Flint model instances.
// The shape-specific graph keeps the slice bound static at trace time.
var flintRouters sync.Map // map[int32]mlx.CompileFunc

func flintRouter(topK int32) mlx.CompileFunc {
	if cached, ok := flintRouters.Load(topK); ok {
		return cached.(mlx.CompileFunc)
	}
	candidate := mlx.Compile(fmt.Sprintf("FlintSigmoidTopK%d", topK), func(in ...*mlx.Array) []*mlx.Array {
		logits := in[0].AsType(mlx.DTypeFloat32)
		probabilities := mlx.Sigmoid(logits)
		indices := mlx.Argpartition(mlx.Neg(probabilities), int(topK)-1, -1)
		indices = mlx.SliceStartStop(indices, []int32{0, 0}, []int32{int32(logits.Dim(0)), topK})
		weights := mlx.TakeAlongAxis(probabilities, indices, -1)
		weights = mlx.Div(weights, mlx.Sum(weights, -1, true))
		return []*mlx.Array{weights, indices}
	})
	cached, _ := flintRouters.LoadOrStore(topK, candidate)
	return cached.(mlx.CompileFunc)
}

type DenseMLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

func loadDenseMLP(linears model.LinearFactory, prefix string) *DenseMLP {
	m := &DenseMLP{
		GateProj: linears.Make(prefix + ".gate_proj"),
		UpProj:   linears.Make(prefix + ".up_proj"),
		DownProj: linears.Make(prefix + ".down_proj"),
	}
	if m.GateProj == nil || m.UpProj == nil || m.DownProj == nil {
		return nil
	}
	return m
}

func (m *DenseMLP) Forward(x *mlx.Array, _ *Config) *mlx.Array {
	return m.DownProj.Forward(flintSwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}

type SparseMoE struct {
	Router  nn.LinearLayer
	Shared  *DenseMLP
	Experts *Experts
	route   mlx.CompileFunc
}

type Experts struct {
	Gate *ExpertProjection
	Up   *ExpertProjection
	Down *ExpertProjection
}

type ExpertProjection struct {
	Weight      *mlx.Array
	Scales      *mlx.Array
	Biases      *mlx.Array
	GlobalScale *mlx.Array
	GroupSize   int
	Bits        int
	Mode        string
}

// loadExpertProjection reads one routed-expert projection, stacked or per
// expert, and checks it against Flint's shapes: an unquantized bank must be
// BF16 [experts, out, in]; a quantized one must use a gather kernel mode and,
// for NVFP4, carry a global scale.
func loadExpertProjection(tensors map[string]*mlx.Array, mlpPrefix, proj string, out, in int32, cfg *Config) (*ExpertProjection, error) {
	quant := model.QuantDefaults{GroupSize: cfg.QuantGroupSize, Bits: cfg.QuantBits, Mode: cfg.QuantMode, TensorQuant: cfg.TensorQuant}
	bank, err := model.ReadExpertBank(tensors, quant, int(cfg.NumExperts), proj, mlpPrefix+".experts")
	if err != nil {
		return nil, err
	}
	if bank == nil {
		return nil, fmt.Errorf("missing %s.experts.%s", mlpPrefix, proj)
	}
	w, path := bank.Weight, bank.Key
	if w.Dim(1) != int(out) {
		return nil, fmt.Errorf("%s has shape %v, want [%d,%d,*]", path, w.Dims(), cfg.NumExperts, out)
	}
	p := &ExpertProjection{Weight: w, Scales: bank.Scales, Biases: bank.Biases, GlobalScale: bank.GlobalScale, GroupSize: bank.GroupSize, Bits: bank.Bits, Mode: bank.Mode}
	if !bank.Quantized() {
		if w.DType() != mlx.DTypeBFloat16 || w.Dim(2) != int(in) {
			return nil, fmt.Errorf("%s has invalid unquantized expert shape or dtype", path)
		}
		return p, nil
	}
	if !model.SupportsGatherQMM(p.Mode, p.Bits) || p.GroupSize <= 0 ||
		int(in)%p.GroupSize != 0 || int(in)%(32/p.Bits) != 0 ||
		w.DType() != mlx.DTypeUint32 || w.Dim(2) != int(in)/(32/p.Bits) ||
		p.Scales.NumDims() != 3 || p.Scales.Dim(0) != int(cfg.NumExperts) ||
		p.Scales.Dim(1) != int(out) || p.Scales.Dim(2) != int(in)/p.GroupSize {
		return nil, fmt.Errorf("%s has unsupported expert quantization: weight=%v scale=%v mode=%q bits=%d group=%d", path, w.Dims(), p.Scales.Dims(), p.Mode, p.Bits, p.GroupSize)
	}
	if p.Mode != "affine" && p.Biases != nil {
		return nil, fmt.Errorf("%s has bias with floating point quantization %q", path, p.Mode)
	}
	if p.Biases != nil && !slices.Equal(p.Biases.Dims(), p.Scales.Dims()) {
		return nil, fmt.Errorf("%s has invalid expert quantization bias shape %v", path, p.Biases.Dims())
	}
	if p.Mode == "nvfp4" && p.GlobalScale == nil {
		return nil, fmt.Errorf("%s is missing NVFP4 global scale", path)
	}
	return p, nil
}

func (p *ExpertProjection) Forward(x, indices *mlx.Array) *mlx.Array {
	return p.forward(x, indices, false)
}

func (p *ExpertProjection) forward(x, indices *mlx.Array, sorted bool) *mlx.Array {
	if p.Scales != nil {
		return mlx.GatherQMM(x, p.Weight, p.Scales, p.Biases, nil, indices,
			true, p.GroupSize, p.Bits, p.Mode, p.GlobalScale, sorted)
	}
	return mlx.GatherMM(x, mlx.Transpose(p.Weight, 0, 2, 1), nil, indices, sorted)
}

func loadSparseMoE(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, cfg *Config) (*SparseMoE, error) {
	mlpPrefix := prefix + ".mlp"
	m := &SparseMoE{
		Router: linears.Make(mlpPrefix + ".gate"),
		Shared: loadDenseMLP(linears, mlpPrefix+".shared_expert"),
		route:  flintRouter(cfg.NumExpertsPerTok),
	}
	if m.Router == nil || m.Shared == nil {
		return nil, fmt.Errorf("missing MoE router or shared expert at %s", mlpPrefix)
	}
	var projections [3]*ExpertProjection
	for k, proj := range []struct {
		name    string
		out, in int32
	}{
		{"gate_proj", cfg.MoeIntermediateSize, cfg.HiddenSize},
		{"up_proj", cfg.MoeIntermediateSize, cfg.HiddenSize},
		{"down_proj", cfg.HiddenSize, cfg.MoeIntermediateSize},
	} {
		p, err := loadExpertProjection(tensors, mlpPrefix, proj.name, proj.out, proj.in, cfg)
		if err != nil {
			return nil, err
		}
		projections[k] = p
	}
	// Unquantized banks stay in publisher layout; transposition is a view at use.
	m.Experts = &Experts{Gate: projections[0], Up: projections[1], Down: projections[2]}
	return m, nil
}

func (m *SparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	return m.forward(x, nil, cfg)
}

func (m *SparseMoE) ForwardAdd(x, residual *mlx.Array, cfg *Config) *mlx.Array {
	return m.forward(x, residual, cfg)
}

func (m *SparseMoE) forward(x, residual *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	batch, length := int32(dims[0]), int32(dims[1])
	flat := mlx.Reshape(x, batch*length, cfg.HiddenSize)
	route := m.route
	if route == nil { // Small hand-built models in tests do not use the loader.
		route = flintRouter(cfg.NumExpertsPerTok)
	}
	routing := route(m.Router.Forward(flat))
	weights, indices := routing[0], routing[1]

	selected := m.Experts.Forward(flat, indices, cfg)
	shared := m.Shared.Forward(x, cfg)
	if residual != nil {
		flatShared := mlx.Reshape(shared, batch*length, cfg.HiddenSize)
		flatResidual := mlx.Reshape(residual, batch*length, cfg.HiddenSize)
		out := flintExpertWeightedSumAdd(selected, weights, flatShared, flatResidual)[0]
		return mlx.Reshape(out, batch, length, cfg.HiddenSize)
	}
	routed := flintExpertWeightedSum(selected, weights)
	routed = mlx.Reshape(routed, batch, length, cfg.HiddenSize)
	return mlx.Add(routed, shared)
}

func (e *Experts) Forward(x, indices *mlx.Array, cfg *Config) *mlx.Array {
	tokens := int32(x.Dim(0))
	topK := cfg.NumExpertsPerTok
	expanded := mlx.Reshape(x, tokens, 1, 1, cfg.HiddenSize)
	var inverse *mlx.Array
	sorted := tokens >= 256
	if sorted {
		// At 256 tokens Flint has six routes per token and 256 experts. The
		// sorted gather can reuse each expert bank across several inputs.
		routes := tokens * topK
		flatIndices := mlx.Flatten(indices)
		order := mlx.Argsort(flatIndices, 0)
		inverse = mlx.Argsort(order, 0)
		indices = mlx.Reshape(mlx.Take(flatIndices, order, 0), routes, 1)
		expanded = mlx.Reshape(
			mlx.Take(x, mlx.FloorDivideScalar(order, topK), 0),
			routes, 1, 1, cfg.HiddenSize)
	}
	gate := e.Gate.forward(expanded, indices, sorted)
	up := e.Up.forward(expanded, indices, sorted)
	activated := flintSwiGLU(gate, up)
	down := e.Down.forward(activated, indices, sorted)
	if sorted {
		routes := tokens * topK
		down = mlx.Take(mlx.Reshape(down, routes, cfg.HiddenSize), inverse, 0)
	}
	return mlx.Reshape(down, tokens, topK, cfg.HiddenSize)
}
