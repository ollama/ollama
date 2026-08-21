package bailing_moe_v3

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

type Router struct {
	Gate       nn.LinearLayer
	ExpertBias *mlx.Array
	// fusedDecode is the compiled single-token routing graph (B = L = 1),
	// built once per router with the config baked in. Decode-step routing is
	// ~18 tiny data-movement kernels; compiling them collapses the host graph
	// construction and fuses the elementwise segments.
	fusedDecode mlx.CompileFunc
}

type SwitchMLP struct {
	// GatherMM consumes [experts, input, output]. Quantized GatherQMM keeps
	// imported weights in their native packed [experts, output, input] layout.
	GateWeight *mlx.Array
	UpWeight   *mlx.Array
	DownWeight *mlx.Array

	GateWeightQ, GateScales, GateBiases *mlx.Array
	UpWeightQ, UpScales, UpBiases       *mlx.Array
	DownWeightQ, DownScales, DownBiases *mlx.Array

	// GateUpWeightQ, when non-nil, is the per-expert row-concatenation of
	// GateWeightQ and UpWeightQ (and companions): one GatherQMM computes
	// both projections, halving the gather count on the hot decode path.
	// Rows within an expert are independent quant groups, so the fusion is
	// bit-exact.
	GateUpWeightQ, GateUpScales, GateUpBiases *mlx.Array

	GateBits, UpBits, DownBits                int
	GateGroupSize, UpGroupSize, DownGroupSize int
	GateMode, UpMode, DownMode                string

	UseQuantized bool
}

type stackedExpertWeights struct {
	Weight    *mlx.Array
	Scales    *mlx.Array
	Biases    *mlx.Array
	Bits      int
	GroupSize int
	Mode      string
}

type SparseMoE struct {
	Router       *Router
	Switch       *SwitchMLP
	SharedExpert *DenseMLP
}

func firstK(a *mlx.Array, k int32) *mlx.Array {
	dims := a.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, d := range dims {
		stop[i] = int32(d)
	}
	stop[len(stop)-1] = k
	return mlx.SliceStartStop(a, start, stop)
}

// routeTopK selects the top-k experts from grouped sigmoid scores: top-2
// experts score each group, top-k groups survive, and the top-k experts are
// chosen from the surviving groups. Expert bias affects selection but not the
// returned weights. scores and routingScores are [B, L, E].
func routeTopK(scores, routingScores *mlx.Array, B, L int32, cfg *Config) (*mlx.Array, *mlx.Array) {
	expertsPerGroup := cfg.NumExperts / cfg.NGroup
	grouped := mlx.Reshape(routingScores, B, L, cfg.NGroup, expertsPerGroup)

	groupTop2 := firstK(mlx.Argpartition(mlx.Neg(grouped), 1, -1), 2)
	groupScores := mlx.Sum(mlx.TakeAlongAxis(grouped, groupTop2, -1), -1, false)
	groupIndices := firstK(mlx.Argpartition(mlx.Neg(groupScores), int(cfg.TopKGroup)-1, -1), cfg.TopKGroup)
	groupIndices = groupIndices.AsType(mlx.DTypeInt32)

	groupGather := mlx.Tile(mlx.ExpandDims(groupIndices, -1), []int32{1, 1, 1, expertsPerGroup})
	candidateScores := mlx.TakeAlongAxis(grouped, groupGather, 2)
	candidateScores = mlx.Reshape(candidateScores, B, L, cfg.TopKGroup*expertsPerGroup)

	locals := make([]int32, expertsPerGroup)
	for i := range locals {
		locals[i] = int32(i)
	}
	localIDs := mlx.NewArrayInt32(locals, []int32{1, 1, 1, expertsPerGroup})
	localIDs = mlx.Tile(localIDs, []int32{B, L, cfg.TopKGroup, 1})
	globalIDs := mlx.Add(mlx.MulScalar(groupGather, float32(expertsPerGroup)), localIDs)
	globalIDs = mlx.Reshape(globalIDs, B, L, cfg.TopKGroup*expertsPerGroup)

	selectedInCandidates := firstK(mlx.Argpartition(mlx.Neg(candidateScores), int(cfg.NumExpertsPerTok)-1, -1), cfg.NumExpertsPerTok)
	indices := mlx.TakeAlongAxis(globalIDs, selectedInCandidates, -1).AsType(mlx.DTypeInt32)
	weights := mlx.TakeAlongAxis(scores, indices, -1)
	if cfg.NormTopKProb && cfg.NumExpertsPerTok > 1 {
		weights = mlx.Div(weights, mlx.AddScalar(mlx.Sum(weights, -1, true), 1e-20))
	}
	weights = mlx.MulScalar(weights, cfg.RoutedScalingFactor)
	return indices, weights
}

// buildFusedDecodeRouter compiles the B = L = 1 routing graph. The gate
// matmul stays outside; inputs are the f32 logits [1,1,E] plus the expert
// bias when the router has one. Config values are baked into the closure, so
// each router owns its compiled program.
func buildFusedDecodeRouter(cfg *Config, hasBias bool) mlx.CompileFunc {
	cfgCopy := *cfg
	return mlx.Compile("BailingDecodeRouter", func(in ...*mlx.Array) []*mlx.Array {
		logits := in[0]
		scores := mlx.Sigmoid(logits)
		routingScores := scores
		if hasBias {
			routingScores = mlx.Add(scores, mlx.Reshape(in[1].AsType(mlx.DTypeFloat32), 1, 1, cfgCopy.NumExperts))
		}
		indices, weights := routeTopK(scores, routingScores, 1, 1, &cfgCopy)
		return []*mlx.Array{indices, weights}
	})
}

// Forward implements Bailing's no-aux grouped sigmoid router:
// top-2 experts score each group, top-k groups survive, and top-k experts are
// selected from those groups. Expert bias affects selection but not weights.
func (r *Router) Forward(x *mlx.Array, cfg *Config) (*mlx.Array, *mlx.Array) {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	logits := r.Gate.Forward(x.AsType(mlx.DTypeFloat32))
	if B == 1 && L == 1 && r.fusedDecode != nil {
		inputs := []*mlx.Array{logits}
		if r.ExpertBias != nil {
			inputs = append(inputs, r.ExpertBias)
		}
		outs := r.fusedDecode(inputs...)
		return outs[0], outs[1]
	}

	scores := mlx.Sigmoid(logits)
	routingScores := scores
	if r.ExpertBias != nil {
		routingScores = mlx.Add(scores, mlx.Reshape(r.ExpertBias.AsType(mlx.DTypeFloat32), 1, 1, cfg.NumExperts))
	}
	return routeTopK(scores, routingScores, B, L, cfg)
}

func (s *SwitchMLP) Forward(x, indices *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.NumExpertsPerTok
	xFlat := mlx.Reshape(mlx.ExpandDims(mlx.ExpandDims(x, -2), -2), B*L, 1, 1, cfg.HiddenSize)
	idxFlat := mlx.Reshape(indices, B*L, topK)

	doSort := B*L >= 64
	var inverse *mlx.Array
	n := B * L * topK
	if doSort {
		all := mlx.Flatten(idxFlat)
		order := mlx.Argsort(all, 0)
		inverse = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, topK), 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(all, order, 0), n, 1)
	}

	var gate, up, hidden, down *mlx.Array
	if s.UseQuantized {
		if s.GateUpWeightQ != nil {
			gateUp := mlx.GatherQMM(xFlat, s.GateUpWeightQ, s.GateUpScales, s.GateUpBiases,
				nil, idxFlat, true, s.GateGroupSize, s.GateBits, s.GateMode, doSort)
			dims := gateUp.Dims()
			half := int32(dims[len(dims)-1]) / 2
			starts := make([]int32, len(dims))
			stops := make([]int32, len(dims))
			for i, d := range dims {
				stops[i] = int32(d)
			}
			stops[len(stops)-1] = half
			gate = mlx.SliceStartStop(gateUp, starts, stops)
			starts2 := append([]int32(nil), starts...)
			stops2 := append([]int32(nil), stops...)
			starts2[len(starts2)-1] = half
			stops2[len(stops2)-1] = 2 * half
			up = mlx.SliceStartStop(gateUp, starts2, stops2)
		} else {
			gate = mlx.GatherQMM(xFlat, s.GateWeightQ, s.GateScales, s.GateBiases,
				nil, idxFlat, true, s.GateGroupSize, s.GateBits, s.GateMode, doSort)
			up = mlx.GatherQMM(xFlat, s.UpWeightQ, s.UpScales, s.UpBiases,
				nil, idxFlat, true, s.UpGroupSize, s.UpBits, s.UpMode, doSort)
		}
		hidden = mlx.SwiGLU(gate, up)
		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases,
			nil, idxFlat, true, s.DownGroupSize, s.DownBits, s.DownMode, doSort)
	} else {
		gate = mlx.GatherMM(xFlat, s.GateWeight, nil, idxFlat, doSort)
		up = mlx.GatherMM(xFlat, s.UpWeight, nil, idxFlat, doSort)
		hidden = mlx.SwiGLU(gate, up)
		down = mlx.GatherMM(hidden, s.DownWeight, nil, idxFlat, doSort)
	}
	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), inverse, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}
	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
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

func loadStackedProjection(tensors map[string]*mlx.Array, cfg *Config, base string) (*stackedExpertWeights, error) {
	key := base + ".weight"
	weight := tensors[key]
	if weight == nil {
		return nil, nil
	}

	scales := tensors[key+"_scale"]
	if scales == nil {
		if cfg.TensorQuant != nil && cfg.TensorQuant[key] != nil {
			return nil, fmt.Errorf("quantized stacked expert projection %s is missing its scale tensor", base)
		}
		return &stackedExpertWeights{Weight: weight}, nil
	}

	biases := tensors[key+"_qbias"]
	groupSize, bits, mode := model.ResolveLinearQuantParams(
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant,
		key, weight, scales,
	)
	if !supportsGatherQMM(mode, bits) {
		return nil, fmt.Errorf("stacked expert projection %s uses unsupported quantization mode=%q bits=%d", base, mode, bits)
	}
	if mode == "affine" && biases == nil {
		return nil, fmt.Errorf("stacked expert projection %s uses affine quantization but is missing its qbias tensor", base)
	}

	return &stackedExpertWeights{
		Weight:    weight,
		Scales:    scales,
		Biases:    biases,
		Bits:      bits,
		GroupSize: groupSize,
		Mode:      mode,
	}, nil
}

func (m *SparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	indices, weights := m.Router.Forward(x, cfg)
	expertOut := m.Switch.Forward(x, indices, cfg)
	y := mlx.Sum(mlx.Mul(expertOut.AsType(mlx.DTypeFloat32), mlx.ExpandDims(weights, -1)), 2, false).AsType(x.DType())
	if m.SharedExpert != nil {
		y = mlx.Add(y, m.SharedExpert.Forward(x, cfg))
	}
	return y
}

func loadSparseMoE(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, cfg *Config) (*SparseMoE, error) {
	p := prefix + ".mlp"
	routerWeight := tensors[p+".gate.weight"]
	if routerWeight == nil {
		return nil, fmt.Errorf("missing MoE gate.weight")
	}
	routerWeight = routerWeight.AsType(mlx.DTypeFloat32).Clone()
	mlx.Eval(routerWeight)
	router := &Router{
		Gate:       nn.NewLinear(routerWeight, nil),
		ExpertBias: tensors[p+".gate.expert_bias"],
	}
	router.fusedDecode = buildFusedDecodeRouter(cfg, router.ExpertBias != nil)
	if cfg.RouterUsesExpertBias && router.ExpertBias == nil {
		return nil, fmt.Errorf("missing MoE gate.expert_bias")
	}

	// The safetensors import plan normally fuses the 128 per-expert tensors into
	// three stacked tensors. Use transpose views of those tensors directly: a
	// second stacked copy costs several GB for this model and can make macOS kill
	// the runner during load.
	stackedGate, err := loadStackedProjection(tensors, cfg, p+".experts.gate_proj")
	if err != nil {
		return nil, err
	}
	stackedUp, err := loadStackedProjection(tensors, cfg, p+".experts.up_proj")
	if err != nil {
		return nil, err
	}
	stackedDown, err := loadStackedProjection(tensors, cfg, p+".experts.down_proj")
	if err != nil {
		return nil, err
	}
	stackedCount := 0
	for _, projection := range []*stackedExpertWeights{stackedGate, stackedUp, stackedDown} {
		if projection != nil {
			stackedCount++
		}
	}
	if stackedCount != 0 && stackedCount != 3 {
		return nil, fmt.Errorf("incomplete stacked expert projections: found %d of 3", stackedCount)
	}
	if stackedCount == 3 {
		shared := &DenseMLP{
			GateProj: linears.Make(p + ".shared_experts.gate_proj"),
			UpProj:   linears.Make(p + ".shared_experts.up_proj"),
			DownProj: linears.Make(p + ".shared_experts.down_proj"),
		}
		if shared.GateProj == nil || shared.UpProj == nil || shared.DownProj == nil {
			return nil, fmt.Errorf("missing shared expert projection")
		}
		shared.fuseGateUp()

		switchMLP := &SwitchMLP{}
		quantizedCount := 0
		for _, projection := range []*stackedExpertWeights{stackedGate, stackedUp, stackedDown} {
			if projection.Scales != nil {
				quantizedCount++
			}
		}
		if quantizedCount != 0 && quantizedCount != 3 {
			return nil, fmt.Errorf("incomplete quantized stacked expert projections: found scales for %d of 3", quantizedCount)
		}
		if quantizedCount == 3 {
			switchMLP.UseQuantized = true
			switchMLP.GateWeightQ, switchMLP.GateScales, switchMLP.GateBiases = stackedGate.Weight, stackedGate.Scales, stackedGate.Biases
			switchMLP.GateBits, switchMLP.GateGroupSize, switchMLP.GateMode = stackedGate.Bits, stackedGate.GroupSize, stackedGate.Mode
			switchMLP.UpWeightQ, switchMLP.UpScales, switchMLP.UpBiases = stackedUp.Weight, stackedUp.Scales, stackedUp.Biases
			switchMLP.UpBits, switchMLP.UpGroupSize, switchMLP.UpMode = stackedUp.Bits, stackedUp.GroupSize, stackedUp.Mode
			switchMLP.DownWeightQ, switchMLP.DownScales, switchMLP.DownBiases = stackedDown.Weight, stackedDown.Scales, stackedDown.Biases
			switchMLP.DownBits, switchMLP.DownGroupSize, switchMLP.DownMode = stackedDown.Bits, stackedDown.GroupSize, stackedDown.Mode

			// Concatenate gate and up along the per-expert output rows so one
			// GatherQMM serves both projections. Quant groups run along the
			// input axis inside each row, so this is bit-exact; it requires
			// matching quantization and layouts on both tensors.
			if stackedGate.Bits == stackedUp.Bits && stackedGate.GroupSize == stackedUp.GroupSize &&
				stackedGate.Mode == stackedUp.Mode &&
				(stackedGate.Biases == nil) == (stackedUp.Biases == nil) {
				switchMLP.GateUpWeightQ = mlx.Concatenate([]*mlx.Array{stackedGate.Weight, stackedUp.Weight}, 1)
				switchMLP.GateUpScales = mlx.Concatenate([]*mlx.Array{stackedGate.Scales, stackedUp.Scales}, 1)
				toEval := []*mlx.Array{switchMLP.GateUpWeightQ, switchMLP.GateUpScales}
				if stackedGate.Biases != nil {
					switchMLP.GateUpBiases = mlx.Concatenate([]*mlx.Array{stackedGate.Biases, stackedUp.Biases}, 1)
					toEval = append(toEval, switchMLP.GateUpBiases)
				}
				mlx.Eval(toEval...)
				// Forward uses only the fused tensors; drop the originals so
				// the loader's pin pass does not keep both copies resident.
				switchMLP.GateWeightQ, switchMLP.GateScales, switchMLP.GateBiases = nil, nil, nil
				switchMLP.UpWeightQ, switchMLP.UpScales, switchMLP.UpBiases = nil, nil, nil
			}
		} else {
			switchMLP.GateWeight = mlx.Transpose(stackedGate.Weight, 0, 2, 1)
			switchMLP.UpWeight = mlx.Transpose(stackedUp.Weight, 0, 2, 1)
			switchMLP.DownWeight = mlx.Transpose(stackedDown.Weight, 0, 2, 1)
		}
		return &SparseMoE{
			Router:       router,
			Switch:       switchMLP,
			SharedExpert: shared,
		}, nil
	}

	gateParts := make([]*mlx.Array, 0, cfg.NumExperts)
	upParts := make([]*mlx.Array, 0, cfg.NumExperts)
	downParts := make([]*mlx.Array, 0, cfg.NumExperts)
	consumed := make([]string, 0, cfg.NumExperts*3)
	for expert := range cfg.NumExperts {
		base := fmt.Sprintf("%s.experts.%d", p, expert)
		gateKey := base + ".gate_proj.weight"
		upKey := base + ".up_proj.weight"
		downKey := base + ".down_proj.weight"
		for _, key := range []string{gateKey, upKey, downKey} {
			_, hasQuantMetadata := cfg.TensorQuant[key]
			if tensors[key+"_scale"] != nil || hasQuantMetadata {
				return nil, fmt.Errorf("expert %d: unstacked quantized projection %s is not supported", expert, key)
			}
		}
		gate, up, down := tensors[gateKey], tensors[upKey], tensors[downKey]
		if gate == nil || up == nil || down == nil {
			return nil, fmt.Errorf("expert %d: missing projection", expert)
		}
		gateParts = append(gateParts, gate)
		upParts = append(upParts, up)
		downParts = append(downParts, down)
		consumed = append(consumed, gateKey, upKey, downKey)
	}
	gateWeight := mlx.Transpose(mlx.Stack(gateParts, 0), 0, 2, 1).Clone()
	upWeight := mlx.Transpose(mlx.Stack(upParts, 0), 0, 2, 1).Clone()
	downWeight := mlx.Transpose(mlx.Stack(downParts, 0), 0, 2, 1).Clone()
	mlx.Eval(gateWeight, upWeight, downWeight)
	for _, key := range consumed {
		delete(tensors, key)
	}

	shared := &DenseMLP{
		GateProj: linears.Make(p + ".shared_experts.gate_proj"),
		UpProj:   linears.Make(p + ".shared_experts.up_proj"),
		DownProj: linears.Make(p + ".shared_experts.down_proj"),
	}
	if shared.GateProj == nil || shared.UpProj == nil || shared.DownProj == nil {
		return nil, fmt.Errorf("missing shared expert projection")
	}
	shared.fuseGateUp()

	return &SparseMoE{
		Router: router,
		Switch: &SwitchMLP{
			GateWeight: gateWeight,
			UpWeight:   upWeight,
			DownWeight: downWeight,
		},
		SharedExpert: shared,
	}, nil
}
