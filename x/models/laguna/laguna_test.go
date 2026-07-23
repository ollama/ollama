package laguna

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestParseConfigLagunaXS(t *testing.T) {
	skipIfNoMLX(t)
	cfg, err := parseConfig([]byte(`{
		"model_type": "laguna",
		"hidden_size": 2048,
		"intermediate_size": 8192,
		"moe_intermediate_size": 512,
		"shared_expert_intermediate_size": 512,
		"num_hidden_layers": 4,
		"num_attention_heads": 48,
		"num_attention_heads_per_layer": [48, 64, 64, 64],
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vocab_size": 100352,
		"max_position_embeddings": 131072,
		"layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
		"sliding_window": 512,
		"mlp_only_layers": [0],
		"decoder_sparse_step": 1,
		"num_experts": 256,
		"num_experts_per_tok": 8,
		"norm_topk_prob": true,
		"moe_routed_scaling_factor": 2.5,
		"gating": "per-head",
		"rms_norm_eps": 1e-6,
		"partial_rotary_factor": 0.5,
		"rope_parameters": {
			"rope_theta": 500000,
			"rope_type": "yarn",
			"factor": 32,
			"original_max_position_embeddings": 4096,
			"beta_fast": 64,
			"beta_slow": 1,
			"attention_factor": 1
		},
		"swa_rope_parameters": {
			"partial_rotary_factor": 1.0,
			"rope_theta": 10000,
			"rope_type": "linear"
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}

	if cfg.FullRopeDim != 64 {
		t.Fatalf("FullRopeDim = %d, want 64", cfg.FullRopeDim)
	}
	if cfg.FullRopeBase != 500000 {
		t.Fatalf("FullRopeBase = %v, want 500000", cfg.FullRopeBase)
	}
	if cfg.FullRopeScale != 1 {
		t.Fatalf("FullRopeScale = %v, want explicit YaRN attention_factor", cfg.FullRopeScale)
	}
	if cfg.FullRopeFreqs == nil {
		t.Fatal("FullRopeFreqs should be precomputed for YaRN")
	}
	if cfg.SlidingRopeDim != 128 {
		t.Fatalf("SlidingRopeDim = %d, want 128", cfg.SlidingRopeDim)
	}
	if cfg.SlidingRopeBase != 10000 {
		t.Fatalf("SlidingRopeBase = %v, want 10000", cfg.SlidingRopeBase)
	}
	if !layerIsSliding(&cfg, 1) {
		t.Fatal("layer 1 should use sliding attention")
	}
	if layerUsesMoE(&cfg, 0) {
		t.Fatal("layer 0 should be dense due to mlp_only_layers")
	}
	if !layerUsesMoE(&cfg, 1) {
		t.Fatal("layer 1 should use MoE")
	}
	if got := numHeadsForLayer(&cfg, 1); got != 64 {
		t.Fatalf("numHeadsForLayer(1) = %d, want 64", got)
	}
}

func TestParseConfigLagunaFP8RopeScaling(t *testing.T) {
	skipIfNoMLX(t)
	cfg, err := parseConfig([]byte(`{
		"hidden_size": 2048,
		"intermediate_size": 8192,
		"num_hidden_layers": 1,
		"num_attention_heads": 48,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vocab_size": 100352,
		"max_position_embeddings": 131072,
		"rope_theta": 500000,
		"partial_rotary_factor": 0.5,
		"rope_scaling": {
			"rope_type": "yarn",
			"factor": 32
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.FullRopeBase != 500000 {
		t.Fatalf("FullRopeBase = %v, want 500000", cfg.FullRopeBase)
	}
	if cfg.FullRopeDim != 64 {
		t.Fatalf("FullRopeDim = %d, want 64", cfg.FullRopeDim)
	}
}

func TestParseConfigLagunaGASchema(t *testing.T) {
	skipIfNoMLX(t)
	cfg, err := parseConfig([]byte(`{
		"model_type": "laguna",
		"hidden_size": 2048,
		"intermediate_size": 8192,
		"moe_intermediate_size": 512,
		"shared_expert_intermediate_size": 512,
		"num_hidden_layers": 4,
		"num_attention_heads": 48,
		"num_attention_heads_per_layer": [48, 64, 64, 64],
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vocab_size": 100352,
		"max_position_embeddings": 131072,
		"layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
		"sliding_window": 512,
		"mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
		"num_experts": 256,
		"num_experts_per_tok": 8,
		"moe_routed_scaling_factor": 2.5,
		"gating": true,
		"rms_norm_eps": 1e-6,
		"partial_rotary_factor": 0.5,
		"rope_parameters": {
			"full_attention": {
				"rope_theta": 500000,
				"rope_type": "yarn",
				"factor": 32,
				"original_max_position_embeddings": 4096,
				"beta_fast": 64,
				"beta_slow": 1,
				"attention_factor": 1,
				"partial_rotary_factor": 0.5
			},
			"sliding_attention": {
				"rope_theta": 10000,
				"rope_type": "default",
				"partial_rotary_factor": 1.0
			}
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Gating != "per-head" {
		t.Fatalf("Gating = %q, want per-head", cfg.Gating)
	}
	if !cfg.NormTopKProb {
		t.Fatal("NormTopKProb should default true")
	}
	if cfg.FullRopeBase != 500000 {
		t.Fatalf("FullRopeBase = %v, want 500000", cfg.FullRopeBase)
	}
	if cfg.SlidingRopeBase != 10000 {
		t.Fatalf("SlidingRopeBase = %v, want 10000", cfg.SlidingRopeBase)
	}
	if cfg.FullRopeDim != 64 {
		t.Fatalf("FullRopeDim = %d, want 64", cfg.FullRopeDim)
	}
	if cfg.SlidingRopeDim != 128 {
		t.Fatalf("SlidingRopeDim = %d, want 128", cfg.SlidingRopeDim)
	}
	if layerUsesMoE(&cfg, 0) {
		t.Fatal("layer 0 should be dense due to mlp_layer_types")
	}
	if !layerUsesMoE(&cfg, 1) {
		t.Fatal("layer 1 should use MoE")
	}
}

func TestTinyLagunaLoadAndForward(t *testing.T) {
	skipIfNoMLX(t)
	cfg, err := parseConfig([]byte(`{
		"model_type": "laguna",
		"hidden_size": 8,
		"intermediate_size": 12,
		"moe_intermediate_size": 4,
		"shared_expert_intermediate_size": 4,
		"num_hidden_layers": 2,
		"num_attention_heads": 2,
		"num_attention_heads_per_layer": [2, 2],
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 16,
		"max_position_embeddings": 64,
		"layer_types": ["full_attention", "sliding_attention"],
		"sliding_window": 2,
		"mlp_only_layers": [0],
		"decoder_sparse_step": 1,
		"num_experts": 2,
		"num_experts_per_tok": 1,
		"norm_topk_prob": false,
		"moe_routed_scaling_factor": 2.5,
		"gating": "per-head",
		"rms_norm_eps": 1e-5,
		"partial_rotary_factor": 0.5,
		"rope_parameters": {
			"rope_theta": 10000,
			"rope_type": "yarn",
			"factor": 2,
			"original_max_position_embeddings": 16,
			"beta_fast": 32,
			"beta_slow": 1
		},
		"swa_rope_parameters": {
			"partial_rotary_factor": 1.0,
			"rope_theta": 10000,
			"rope_type": "linear"
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}

	m := &Model{
		Config: &cfg,
		Layers: []*Layer{
			{LayerIdx: 0, IsSliding: false},
			{LayerIdx: 1, IsSliding: true},
		},
	}
	tensors := tinyLagunaTensors()
	if err := m.LoadWeights(tensors); err != nil {
		t.Fatalf("LoadWeights failed: %v", err)
	}

	tokens := mlx.FromValues([]int32{1, 2, 3}, 1, 3)
	caches := m.NewCaches()
	defer func() {
		for _, c := range caches {
			if c != nil {
				c.Free()
			}
		}
	}()
	hidden := m.Forward(&batch.Batch{
		InputIDs:     tokens,
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{int32(tokens.Dim(1))},
	}, caches)
	mlx.Eval(hidden)
	if got := hidden.Dims(); len(got) != 3 || got[0] != 1 || got[1] != 3 || got[2] != 8 {
		t.Fatalf("hidden shape = %v, want [1 3 8]", got)
	}

	logits := m.Unembed(hidden)
	mlx.Eval(logits)
	if got := logits.Dims(); len(got) != 3 || got[0] != 1 || got[1] != 3 || got[2] != 16 {
		t.Fatalf("logits shape = %v, want [1 3 16]", got)
	}
	for i, v := range logits.Floats() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logits[%d] is not finite: %v", i, v)
		}
	}
}

func TestTinyLagunaLoadWeightsFusesDenseGateUp(t *testing.T) {
	skipIfNoMLX(t)
	cfg, err := parseConfig([]byte(`{
		"model_type": "laguna",
		"hidden_size": 8,
		"intermediate_size": 12,
		"moe_intermediate_size": 4,
		"shared_expert_intermediate_size": 4,
		"num_hidden_layers": 2,
		"num_attention_heads": 2,
		"num_attention_heads_per_layer": [2, 2],
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 16,
		"max_position_embeddings": 64,
		"layer_types": ["full_attention", "sliding_attention"],
		"sliding_window": 2,
		"mlp_only_layers": [0],
		"decoder_sparse_step": 1,
		"num_experts": 2,
		"num_experts_per_tok": 1,
		"norm_topk_prob": false,
		"moe_routed_scaling_factor": 2.5,
		"gating": "per-head",
		"rms_norm_eps": 1e-5
	}`))
	if err != nil {
		t.Fatal(err)
	}

	m := &Model{
		Config: &cfg,
		Layers: []*Layer{
			{LayerIdx: 0, IsSliding: false},
			{LayerIdx: 1, IsSliding: true},
		},
	}
	if err := m.LoadWeights(tinyLagunaTensors()); err != nil {
		t.Fatalf("LoadWeights failed: %v", err)
	}

	moe, ok := m.Layers[1].MLP.(*SparseMoE)
	if !ok {
		t.Fatalf("layer 1 MLP type = %T, want *SparseMoE", m.Layers[1].MLP)
	}
	if moe.SwitchMLP.GateUpWeight == nil {
		t.Fatal("expected fused GateUpWeight to be populated")
	}
	if got, want := moe.SwitchMLP.GateUpWeight.Dims(), []int{2, 8, 8}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("GateUpWeight dims = %v, want %v", got, want)
	}
}

func TestTinyLagunaLoadWeightsKeepsBF16SourceLayout(t *testing.T) {
	skipIfNoMLX(t)
	cfg, err := parseConfig([]byte(`{
		"model_type": "laguna",
		"hidden_size": 8,
		"intermediate_size": 12,
		"moe_intermediate_size": 4,
		"shared_expert_intermediate_size": 4,
		"num_hidden_layers": 2,
		"num_attention_heads": 2,
		"num_attention_heads_per_layer": [2, 2],
		"num_key_value_heads": 1,
		"head_dim": 4,
		"vocab_size": 16,
		"max_position_embeddings": 64,
		"layer_types": ["full_attention", "sliding_attention"],
		"sliding_window": 2,
		"mlp_only_layers": [0],
		"decoder_sparse_step": 1,
		"num_experts": 2,
		"num_experts_per_tok": 1,
		"norm_topk_prob": false,
		"moe_routed_scaling_factor": 2.5,
		"gating": "per-head",
		"rms_norm_eps": 1e-5
	}`))
	if err != nil {
		t.Fatal(err)
	}

	tensors := tinyLagunaTensors()
	for key, tensor := range tensors {
		if strings.Contains(key, ".mlp.experts.") && strings.HasSuffix(key, ".weight") {
			tensors[key] = tensor.AsType(mlx.DTypeBFloat16)
		}
	}
	m := &Model{
		Config: &cfg,
		Layers: []*Layer{
			{LayerIdx: 0, IsSliding: false},
			{LayerIdx: 1, IsSliding: true},
		},
	}
	if err := m.LoadWeights(tensors); err != nil {
		t.Fatalf("LoadWeights failed: %v", err)
	}

	moe, ok := m.Layers[1].MLP.(*SparseMoE)
	if !ok {
		t.Fatalf("layer 1 MLP type = %T, want *SparseMoE", m.Layers[1].MLP)
	}
	if !moe.SwitchMLP.GateUpWeightsSourceLayout || !moe.SwitchMLP.DownWeightSourceLayout {
		t.Fatal("expected BF16 dense SwitchMLP to keep source-layout expert weights")
	}
	if moe.SwitchMLP.GateUpWeight != nil {
		t.Fatal("expected BF16 source-layout SwitchMLP to avoid pre-fused gate/up weights")
	}
	if got, want := moe.SwitchMLP.GateWeight.Dims(), []int{2, 4, 8}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("GateWeight dims = %v, want %v", got, want)
	}
}

func TestTinyLagunaLoadWeightsKeepsMixedExpertPrecision(t *testing.T) {
	skipIfNoMLX(t)
	cfg := &Config{
		HiddenSize:                8,
		IntermediateSize:          12,
		MoeIntermediateSize:       4,
		SharedExpertIntermediate:  4,
		NumHiddenLayers:           2,
		NumAttentionHeads:         2,
		NumAttentionHeadsPerLayer: []int32{2, 2},
		NumKeyValueHeads:          1,
		HeadDim:                   4,
		VocabSize:                 16,
		LayerTypes:                []string{"full_attention", "sliding_attention"},
		MLPOnlyLayers:             []int32{0},
		DecoderSparseStep:         1,
		NumExperts:                2,
		NumExpertsPerTok:          1,
		MoeRoutedScalingFactor:    2.5,
		RMSNormEps:                1e-5,
		QuantGroupSize:            4,
		QuantBits:                 4,
		QuantMode:                 "affine",
	}

	tensors := tinyLagunaTensors()
	for expert := range 2 {
		prefix := "model.layers.1.mlp.experts." + string(rune('0'+expert))
		for _, proj := range []string{"gate_proj", "up_proj"} {
			key := prefix + "." + proj + ".weight"
			weight, scales, biases := mlx.Quantize(tensors[key], cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode)
			tensors[key] = weight
			tensors[key+"_scale"] = scales
			tensors[key+"_qbias"] = biases
		}
		downKey := prefix + ".down_proj.weight"
		tensors[downKey] = tensors[downKey].AsType(mlx.DTypeBFloat16)
	}

	m := &Model{
		Config: cfg,
		Layers: []*Layer{
			{LayerIdx: 0, IsSliding: false},
			{LayerIdx: 1, IsSliding: true},
		},
	}
	if err := m.LoadWeights(tensors); err != nil {
		t.Fatalf("LoadWeights failed: %v", err)
	}

	moe, ok := m.Layers[1].MLP.(*SparseMoE)
	if !ok {
		t.Fatalf("layer 1 MLP type = %T, want *SparseMoE", m.Layers[1].MLP)
	}
	hasFusedGateUp := moe.SwitchMLP.GateUpWeightQ != nil && moe.SwitchMLP.GateUpScales != nil
	hasSeparateGateUp := moe.SwitchMLP.GateWeightQ != nil && moe.SwitchMLP.GateScales != nil &&
		moe.SwitchMLP.UpWeightQ != nil && moe.SwitchMLP.UpScales != nil
	if !hasFusedGateUp && !hasSeparateGateUp {
		t.Fatal("expected quantized gate/up expert weights")
	}
	if moe.SwitchMLP.GateUpWeight != nil || moe.SwitchMLP.GateWeight != nil || moe.SwitchMLP.UpWeight != nil {
		t.Fatal("quantized gate/up expert weights fell back to dense")
	}
	if moe.SwitchMLP.DownWeight == nil || moe.SwitchMLP.DownWeightQ != nil || !moe.SwitchMLP.DownWeightSourceLayout {
		t.Fatal("expected BF16 down expert weights to retain source layout")
	}
}

func TestSparseMoERouteBiasAffectsSelectionNotRoutingWeights(t *testing.T) {
	skipIfNoMLX(t)
	cfg := &Config{
		HiddenSize:       1,
		NumExperts:       2,
		NumExpertsPerTok: 1,
		NormTopKProb:     false,
	}

	moe := &SparseMoE{
		Gate:                 nn.NewLinear(mlx.FromValues([]float32{-4, -3}, 2, 1).AsType(mlx.DTypeBFloat16), nil),
		EScoreCorrectionBias: mlx.FromValues([]float32{0.5, 0}, 2),
	}

	xFlat := mlx.FromValues([]float32{1}, 1, int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16)
	scores, inds, scalesFolded := moe.route(xFlat, cfg)
	if scalesFolded {
		t.Fatal("route folded scales without expert projection scales")
	}
	scores = scores.AsType(mlx.DTypeFloat32)
	inds = inds.AsType(mlx.DTypeInt32)
	mlx.Eval(scores, inds)

	gates := moe.Gate.Forward(xFlat).AsType(mlx.DTypeFloat32)
	probs := mlx.Sigmoid(gates)
	mlx.Eval(probs)
	probVals := probs.Floats()
	if probVals[0] >= probVals[1] {
		t.Fatalf("expected unbiased sigmoid scores to prefer expert 1, got %v", probVals)
	}
	if probVals[0]+0.5 <= probVals[1] {
		t.Fatalf("expected bias to flip selection to expert 0, got probs=%v", probVals)
	}
	if got := inds.Ints(); len(got) != 1 || got[0] != 0 {
		t.Fatalf("selected experts = %v, want [0]", got)
	}
	if got := scores.Floats(); len(got) != 1 || math.Abs(float64(got[0]-probVals[0])) > 1e-6 {
		t.Fatalf("routing weights = %v, want [%v] using unbiased sigmoid scores", got, probVals[0])
	}
}

func TestLagunaSigmoidTopK8CompiledMatchesEager(t *testing.T) {
	skipIfNoMLX(t)
	gates := make([]float32, 2*16)
	for i := range gates {
		gates[i] = float32((i%13)-6) * 0.2
	}
	bias := make([]float32, 16)
	scaleA := make([]float32, 16)
	scaleB := make([]float32, 16)
	for i := range bias {
		bias[i] = float32((i%5)-2) * 0.03
		scaleA[i] = 0.5 + float32(i)*0.01
		scaleB[i] = 0.75 + float32(i)*0.005
	}

	gatesArray := mlx.FromValues(gates, 2, 16).AsType(mlx.DTypeBFloat16)
	biasArray := mlx.FromValues(bias, 16)
	scaleAArray := mlx.FromValues(scaleA, 16)
	scaleBArray := mlx.FromValues(scaleB, 16)
	got := lagunaSigmoidTopK8ScaledNormalized(gatesArray, biasArray, scaleAArray, scaleBArray)

	probs, neg := mlx.SigmoidRouter(gatesArray.AsType(mlx.DTypeFloat32), biasArray)
	wantIndices := mlx.Argpartition(neg, 7, -1)
	wantIndices = mlx.SliceStartStop(wantIndices, []int32{0, 0}, []int32{2, 8})
	wantScores := mlx.TakeAlongAxis(probs, wantIndices, -1)
	wantScores = mlx.Div(wantScores, mlx.Sum(wantScores, -1, true))
	wantScores = scaleScoresByExpert(wantScores, wantIndices, scaleAArray)
	wantScores = scaleScoresByExpert(wantScores, wantIndices, scaleBArray)

	gotScores := got[0].AsType(mlx.DTypeFloat32)
	gotIndices := got[1].AsType(mlx.DTypeInt32)
	wantScores = wantScores.AsType(mlx.DTypeFloat32)
	wantIndices = wantIndices.AsType(mlx.DTypeInt32)
	mlx.Eval(gotScores, gotIndices, wantScores, wantIndices)
	assertFloatSlicesClose(t, gotScores.Floats(), wantScores.Floats(), 1e-6)
	if got, want := gotIndices.Ints(), wantIndices.Ints(); !slices.Equal(got, want) {
		t.Fatalf("indices = %v, want %v", got, want)
	}
}

func TestLagunaSwiGLUGatheredGateScaleCompiledMatchesEager(t *testing.T) {
	skipIfNoMLX(t)
	gateValues := make([]float32, 2*8*4)
	upValues := make([]float32, len(gateValues))
	for i := range gateValues {
		gateValues[i] = float32((i%17)-8) * 0.04
		upValues[i] = float32((i%11)-5) * 0.03
	}
	scaleValues := make([]float32, 16)
	for i := range scaleValues {
		scaleValues[i] = 0.5 + float32(i)*0.025
	}
	indices := mlx.FromValues([]int32{
		0, 3, 6, 9, 12, 15, 2, 5,
		1, 4, 7, 10, 13, 14, 8, 11,
	}, 2, 8)
	gate := mlx.FromValues(gateValues, 2, 8, 1, 4).AsType(mlx.DTypeBFloat16)
	up := mlx.FromValues(upValues, 2, 8, 1, 4).AsType(mlx.DTypeBFloat16)
	scales := mlx.FromValues(scaleValues, 16)

	got := lagunaSwiGLUGatheredGateScale(gate, up, scales, indices)[0]
	want := mlx.SwiGLU(applyExpertGlobalScale(gate, scales, indices), up)
	got = got.AsType(mlx.DTypeFloat32)
	want = want.AsType(mlx.DTypeFloat32)
	mlx.Eval(got, want)
	assertFloatSlicesClose(t, got.Floats(), want.Floats(), 1e-6)
}

func TestLagunaMoEWeightedSumCompiledMatchesEager(t *testing.T) {
	skipIfNoMLX(t)
	expertValues := make([]float32, 1*2*8*4)
	scoreValues := make([]float32, 1*2*8)
	for i := range expertValues {
		expertValues[i] = float32((i%19)-9) * 0.02
	}
	for i := range scoreValues {
		scoreValues[i] = float32(i+1) / 20
	}
	addAValues := []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8}
	addBValues := []float32{-0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1}
	expert := mlx.FromValues(expertValues, 1, 2, 8, 4).AsType(mlx.DTypeBFloat16)
	scores := mlx.FromValues(scoreValues, 1, 2, 8)
	scale := mlx.FromValue(float32(2.5))
	addA := mlx.FromValues(addAValues, 1, 2, 4).AsType(mlx.DTypeBFloat16)
	addB := mlx.FromValues(addBValues, 1, 2, 4).AsType(mlx.DTypeBFloat16)

	weighted := mlx.Mul(expert, mlx.ExpandDims(scores.AsType(expert.DType()), -1))
	weighted = mlx.Mul(mlx.Sum(weighted, 2, false), scale.AsType(expert.DType()))
	wantAdd := mlx.Add(weighted.AsType(addA.DType()), addA)
	wantAdd2 := mlx.Add(wantAdd, addB)
	gotAdd := lagunaMoEWeightedSumAdd(expert, scores, scale, addA)[0]
	gotAdd2 := lagunaMoEWeightedSumAdd2(expert, scores, scale, addA, addB)[0]

	gotAdd = gotAdd.AsType(mlx.DTypeFloat32)
	gotAdd2 = gotAdd2.AsType(mlx.DTypeFloat32)
	wantAdd = wantAdd.AsType(mlx.DTypeFloat32)
	wantAdd2 = wantAdd2.AsType(mlx.DTypeFloat32)
	mlx.Eval(gotAdd, gotAdd2, wantAdd, wantAdd2)
	assertFloatSlicesClose(t, gotAdd.Floats(), wantAdd.Floats(), 1e-6)
	assertFloatSlicesClose(t, gotAdd2.Floats(), wantAdd2.Floats(), 1e-6)
}

func TestSwitchMLPFusedGateUpMatchesSeparate(t *testing.T) {
	skipIfNoMLX(t)
	cfg := &Config{HiddenSize: 4, NumExpertsPerTok: 2}
	B, L := int32(2), int32(3)
	xVals := make([]float32, int(B*L*cfg.HiddenSize))
	for i := range xVals {
		xVals[i] = float32((i%17)-8) * 0.01
	}
	x := mlx.FromValues(xVals, int(B), int(L), int(cfg.HiddenSize)).AsType(mlx.DTypeBFloat16)

	indicesVals := make([]int32, B*L*cfg.NumExpertsPerTok)
	for i := 0; i < len(indicesVals); i += int(cfg.NumExpertsPerTok) {
		indicesVals[i] = int32((i / int(cfg.NumExpertsPerTok)) % 2)
		indicesVals[i+1] = int32(((i / int(cfg.NumExpertsPerTok)) + 1) % 2)
	}
	indices := mlx.FromValues(indicesVals, int(B*L), int(cfg.NumExpertsPerTok))

	separate := &SwitchMLP{
		GateWeight: makePatternExpertWeight(2, 4, 3, 0.011),
		UpWeight:   makePatternExpertWeight(2, 4, 3, 0.017),
		DownWeight: makePatternExpertWeight(2, 3, 4, 0.013),
	}
	fused := &SwitchMLP{
		GateUpWeight: fuseExpertStacks(separate.GateWeight, separate.UpWeight, 2),
		DownWeight:   separate.DownWeight,
	}

	gotSeparate := separate.Forward(x, indices, cfg)
	gotFused := fused.Forward(x, indices, cfg)
	mlx.Eval(gotSeparate, gotFused)

	gotFusedF32 := gotFused.AsType(mlx.DTypeFloat32)
	gotSeparateF32 := gotSeparate.AsType(mlx.DTypeFloat32)
	mlx.Eval(gotFusedF32, gotSeparateF32)
	assertFloatSlicesClose(t, gotFusedF32.Floats(), gotSeparateF32.Floats(), 1e-5)
}

func TestSwitchMLPMixedQuantizedGateUpDenseDownMatchesDense(t *testing.T) {
	skipIfNoMLX(t)
	cfg := &Config{HiddenSize: 32, NumExpertsPerTok: 2}
	x := makePatternExpertWeight(1, 2, int(cfg.HiddenSize), 0.013)
	indices := mlx.FromValues([]int32{0, 1, 1, 0}, 2, int(cfg.NumExpertsPerTok))

	gateWeight := makePatternExpertWeight(2, 32, 32, 0.011)
	upWeight := makePatternExpertWeight(2, 32, 32, 0.017)
	downWeight := makePatternExpertWeight(2, 32, 32, 0.013)
	gateQ, gateScales, gateBiases := mlx.Quantize(gateWeight, 32, 8, "mxfp8")
	upQ, upScales, upBiases := mlx.Quantize(upWeight, 32, 8, "mxfp8")
	mlx.Eval(gateQ, gateScales, upQ, upScales)

	mixed := &SwitchMLP{
		GateUpWeightQ:          fuseExpertStacks(gateQ, upQ, 1),
		GateUpScales:           fuseExpertStacks(gateScales, upScales, 1),
		GateUpBiases:           fuseExpertStacks(gateBiases, upBiases, 1),
		GateUpBits:             8,
		GateUpGroupSize:        32,
		GateUpMode:             "mxfp8",
		DownWeight:             downWeight,
		DownWeightSourceLayout: true,
	}
	dense := &SwitchMLP{
		GateWeight:                gateWeight,
		UpWeight:                  upWeight,
		DownWeight:                downWeight,
		GateUpWeightsSourceLayout: true,
		DownWeightSourceLayout:    true,
	}

	got := mixed.Forward(x, indices, cfg).AsType(mlx.DTypeFloat32)
	want := dense.Forward(x, indices, cfg).AsType(mlx.DTypeFloat32)
	mlx.Eval(got, want)
	assertFloatSlicesClose(t, got.Floats(), want.Floats(), 0.02)
}

func TestDenseExpertWeightForGatherMMDequantizesQuantizedWeight(t *testing.T) {
	skipIfNoMLX(t)
	weight := makePatternExpertWeight(2, 4, 32, 0.011)
	qweight, scales, qbiases := mlx.Quantize(weight, 32, 8, "mxfp8")
	mlx.Eval(qweight, scales)

	got := denseExpertWeightForGatherMM(&stackedExpertWeights{
		Weight:    qweight,
		Scales:    scales,
		Biases:    qbiases,
		GroupSize: 32,
		Bits:      8,
		Mode:      "mxfp8",
	})
	mlx.Eval(got)

	if got == nil {
		t.Fatal("denseExpertWeightForGatherMM returned nil")
	}
	if dims := got.Dims(); len(dims) != 3 || dims[0] != 2 || dims[1] != 32 || dims[2] != 4 {
		t.Fatalf("dense expert dims = %v, want [2 32 4]", dims)
	}
	if got.DType() == mlx.DTypeUint32 {
		t.Fatal("dense expert fallback kept packed U32 weight")
	}
}

func TestCombinedTensorGlobalScaleIgnoresInputGlobalScale(t *testing.T) {
	skipIfNoMLX(t)
	tensors := map[string]*mlx.Array{
		"proj.weight.global_scale":       mlx.FromValues([]float32{0.25}, 1),
		"proj.weight.input_global_scale": mlx.FromValues([]float32{8}, 1),
	}

	got, _ := combinedTensorGlobalScale(tensors, "proj.weight")
	if got == nil {
		t.Fatal("combinedTensorGlobalScale returned nil")
	}
	mlx.Eval(got)
	vals := got.Floats()
	if len(vals) != 1 || vals[0] != 0.25 {
		t.Fatalf("combinedTensorGlobalScale = %v, want [0.25]", vals)
	}
}

func tinyLagunaTensors() map[string]*mlx.Array {
	tensors := map[string]*mlx.Array{
		"model.embed_tokens.weight": weights(16, 8),
		"model.norm.weight":         ones(8),
		"lm_head.weight":            weights(16, 8),
	}
	for layer := range 2 {
		prefix := "model.layers." + string(rune('0'+layer))
		tensors[prefix+".input_layernorm.weight"] = ones(8)
		tensors[prefix+".post_attention_layernorm.weight"] = ones(8)
		tensors[prefix+".self_attn.q_proj.weight"] = weights(8, 8)
		tensors[prefix+".self_attn.k_proj.weight"] = weights(4, 8)
		tensors[prefix+".self_attn.v_proj.weight"] = weights(4, 8)
		tensors[prefix+".self_attn.o_proj.weight"] = weights(8, 8)
		tensors[prefix+".self_attn.g_proj.weight"] = weights(2, 8)
		tensors[prefix+".self_attn.q_norm.weight"] = ones(4)
		tensors[prefix+".self_attn.k_norm.weight"] = ones(4)
	}

	tensors["model.layers.0.mlp.gate_proj.weight"] = weights(12, 8)
	tensors["model.layers.0.mlp.up_proj.weight"] = weights(12, 8)
	tensors["model.layers.0.mlp.down_proj.weight"] = weights(8, 12)

	tensors["model.layers.1.mlp.gate.weight"] = weights(2, 8)
	tensors["model.layers.1.mlp.experts.e_score_correction_bias"] = mlx.FromValues([]float32{0.1, -0.1}, 2)
	for expert := range 2 {
		prefix := "model.layers.1.mlp.experts." + string(rune('0'+expert))
		tensors[prefix+".gate_proj.weight"] = weights(4, 8)
		tensors[prefix+".up_proj.weight"] = weights(4, 8)
		tensors[prefix+".down_proj.weight"] = weights(8, 4)
	}
	tensors["model.layers.1.mlp.shared_expert.gate_proj.weight"] = weights(4, 8)
	tensors["model.layers.1.mlp.shared_expert.up_proj.weight"] = weights(4, 8)
	tensors["model.layers.1.mlp.shared_expert.down_proj.weight"] = weights(8, 4)
	return tensors
}

func makeExpertWeight(vals []float32, dims ...int) *mlx.Array {
	return mlx.FromValues(vals, dims...).AsType(mlx.DTypeBFloat16)
}

func makePatternExpertWeight(numExperts, rows, cols int, scale float32) *mlx.Array {
	vals := make([]float32, numExperts*rows*cols)
	for i := range vals {
		vals[i] = float32((i%23)-11) * scale
	}
	return makeExpertWeight(vals, numExperts, rows, cols)
}

func assertFloatSlicesClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("value[%d] = %v, want %v (tol=%g)", i, got[i], want[i], tol)
		}
	}
}

func weights(rows, cols int) *mlx.Array {
	vals := make([]float32, rows*cols)
	for i := range vals {
		vals[i] = float32((i%7)-3) * 0.01
	}
	return mlx.FromValues(vals, rows, cols)
}

func ones(n int) *mlx.Array {
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = 1
	}
	return mlx.FromValues(vals, n)
}

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}
