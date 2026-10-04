package kolibri1

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
)

// MoE routes on float32 logits plus the learned balancing bias. The bias
// affects selection only; mixture weights are sigmoid of the original logits.
type MoE struct {
	Router                 *mlx.Array
	ExpertBias             *mlx.Array
	GateUp, Gate, Up, Down *ExpertLinear
	GateScale, UpScale     *mlx.Array
	Shared                 *MLP
}

// ExpertLinear holds a stacked bank without expanding it into dense weights.
// Exported array fields are retained by the runner's model scope collector.
type ExpertLinear struct {
	Weight, Scales, Biases, GlobalScale *mlx.Array
	GroupSize, Bits                     int
	Mode                                string
}

func loadExpert(tensors map[string]*mlx.Array, path string, cfg *Config) (*ExpertLinear, error) {
	w := tensors[path+".weight"]
	if w == nil || w.NumDims() != 3 || w.Dim(0) != int(cfg.NumExperts) {
		return nil, fmt.Errorf("missing or invalid expert bank %s", path)
	}
	e := &ExpertLinear{Weight: w, Scales: tensors[path+".weight_scale"], Biases: tensors[path+".weight_qbias"]}
	if e.Scales != nil {
		e.GroupSize, e.Bits, e.Mode = model.ResolveLinearQuantParams(cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant, path+".weight", w, e.Scales)
		e.GlobalScale, _ = model.ReadGlobalScale(tensors, path+".weight")
		e.GlobalScale = model.PrepareGatherQMMGlobalScale(e.GlobalScale, int(cfg.NumExperts))
	}
	return e, nil
}

func loadMoE(tensors map[string]*mlx.Array, linears model.LinearFactory, prefix string, cfg *Config) (*MoE, error) {
	m := &MoE{Router: tensors[prefix+".mlp.gate.weight"], ExpertBias: tensors[prefix+".moe.router.expert_bias"]}
	if m.Router == nil || m.ExpertBias == nil {
		return nil, fmt.Errorf("missing router weight or expert bias")
	}
	if m.Router.NumDims() != 2 || m.Router.Dim(0) != int(cfg.NumExperts) || m.Router.Dim(1) != int(cfg.HiddenSize) || m.ExpertBias.Size() != int(cfg.NumExperts) {
		return nil, fmt.Errorf("invalid router shape")
	}
	m.Router = m.Router.AsType(mlx.DTypeFloat32)
	m.ExpertBias = m.ExpertBias.AsType(mlx.DTypeFloat32)
	var err error
	for _, p := range []struct {
		name string
		dst  **ExpertLinear
	}{{"gate_proj", &m.Gate}, {"up_proj", &m.Up}, {"down_proj", &m.Down}} {
		*p.dst, err = loadExpert(tensors, prefix+".mlp.experts."+p.name, cfg)
		if err != nil {
			return nil, err
		}
	}
	// Joining output rows preserves every quantization group and saves one
	// gather matmul per layer. Apply the two global scales after splitting.
	m.GateUp = fuseExperts(m.Gate, m.Up)
	if m.GateUp != nil {
		// Keep scalar scales scalar: broadcasting them for GatherQMM and
		// then gathering identical values adds two kernels to every layer.
		m.GateScale, _ = model.ReadGlobalScale(tensors, prefix+".mlp.experts.gate_proj.weight")
		m.UpScale, _ = model.ReadGlobalScale(tensors, prefix+".mlp.experts.up_proj.weight")
		m.Gate, m.Up = nil, nil
	}
	m.Shared = &MLP{
		GateProj: linears.Make(prefix + ".mlp.shared_experts.gate_proj"),
		UpProj:   linears.Make(prefix + ".mlp.shared_experts.up_proj"),
		DownProj: linears.Make(prefix + ".mlp.shared_experts.down_proj"),
	}
	if m.Shared.GateProj == nil || m.Shared.UpProj == nil || m.Shared.DownProj == nil {
		return nil, fmt.Errorf("missing shared expert projections")
	}
	return m, nil
}

func fuseExperts(a, b *ExpertLinear) *ExpertLinear {
	if (a.Scales == nil) != (b.Scales == nil) || (a.Biases == nil) != (b.Biases == nil) || a.GroupSize != b.GroupSize || a.Bits != b.Bits || a.Mode != b.Mode {
		return nil
	}
	if a.Weight.Dim(0) != b.Weight.Dim(0) || a.Weight.Dim(2) != b.Weight.Dim(2) {
		return nil
	}
	e := *a
	e.GlobalScale = nil
	e.Weight = mlx.Concatenate([]*mlx.Array{a.Weight, b.Weight}, 1)
	if a.Scales != nil {
		e.Scales = mlx.Concatenate([]*mlx.Array{a.Scales, b.Scales}, 1)
	}
	if a.Biases != nil {
		e.Biases = mlx.Concatenate([]*mlx.Array{a.Biases, b.Biases}, 1)
	}
	return &e
}

func (e *ExpertLinear) Forward(x, indices *mlx.Array, sorted bool) *mlx.Array {
	if e.Scales != nil {
		return mlx.GatherQMM(x, e.Weight, e.Scales, e.Biases, nil, indices, true, e.GroupSize, e.Bits, e.Mode, e.GlobalScale, sorted)
	}
	return mlx.GatherMM(x, mlx.Transpose(e.Weight, 0, 2, 1), nil, indices, sorted)
}

func (m *MoE) route(x *mlx.Array, cfg *Config) (indices, scores *mlx.Array) {
	logits := mlx.Matmul(x.AsType(mlx.DTypeFloat32), mlx.Transpose(m.Router, 1, 0))
	indices = mlx.Argpartition(mlx.Neg(mlx.Add(logits, m.ExpertBias)), int(cfg.NumExpertsPerTok)-1, -1)
	indices = indices.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(cfg.NumExpertsPerTok)))
	scores = mlx.Sigmoid(mlx.TakeAlongAxis(logits, indices, -1))
	if cfg.NormTopKProb {
		scores = mlx.Div(scores, mlx.AddScalar(mlx.Sum(scores, -1, true), 1e-20))
	}
	return indices, scores
}

func (m *MoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	B, L := int32(x.Dim(0)), int32(x.Dim(1))
	indices, scores := m.route(x, cfg)
	xFlat := mlx.Reshape(x, B*L, 1, 1, cfg.HiddenSize)
	idxFlat := mlx.Reshape(indices, B*L, cfg.NumExpertsPerTok)
	// Group large prefill batches by expert; decode avoids the sorting cost.
	sorted := B*L >= 64
	var inverse *mlx.Array
	if sorted {
		flat := mlx.Flatten(idxFlat)
		order := mlx.Argsort(flat, 0)
		inverse = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, cfg.NumExpertsPerTok), 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(flat, order, 0), B*L*cfg.NumExpertsPerTok, 1)
	}
	var hidden *mlx.Array
	if m.GateUp != nil {
		gu := m.GateUp.Forward(xFlat, idxFlat, sorted)
		gate := gu.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(cfg.MoEIntermediateSize)))
		up := gu.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(int(cfg.MoEIntermediateSize), int(2*cfg.MoEIntermediateSize)))
		scaleRows := func(scale *mlx.Array) *mlx.Array {
			if scale == nil || scale.Size() == 1 {
				return scale
			}
			return mlx.ExpandDims(mlx.ExpandDims(mlx.Take(scale, idxFlat, 0), -1), -1)
		}
		hidden = mlx.SwiGLUScaled(gate, scaleRows(m.GateScale), up, scaleRows(m.UpScale))
	} else {
		gate := m.Gate.Forward(xFlat, idxFlat, sorted)
		up := m.Up.Forward(xFlat, idxFlat, sorted)
		hidden = mlx.SwiGLU(gate, up)
	}
	y := m.Down.Forward(hidden, idxFlat, sorted)
	if sorted {
		y = mlx.Take(mlx.Reshape(y, B*L*cfg.NumExpertsPerTok, cfg.HiddenSize), inverse, 0)
	}
	y = mlx.Reshape(y, B, L, cfg.NumExpertsPerTok, cfg.HiddenSize)
	y = mlx.Sum(mlx.Mul(y.AsType(mlx.DTypeFloat32), mlx.ExpandDims(scores, -1)), 2, false).AsType(x.DType())
	return mlx.Add(y, m.Shared.Forward(x))
}
