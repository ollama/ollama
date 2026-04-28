package laguna

import (
	"math"
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
	if !moe.SwitchMLP.UseFusedGateUp {
		t.Fatal("expected dense SwitchMLP to fuse gate/up expert weights")
	}
	if moe.SwitchMLP.GateUpWeight == nil {
		t.Fatal("expected fused GateUpWeight to be populated")
	}
	if got, want := moe.SwitchMLP.GateUpWeight.Dims(), []int{2, 8, 8}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("GateUpWeight dims = %v, want %v", got, want)
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
	scores, inds := moe.route(xFlat, cfg)
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
		GateUpWeight:   fuseExpertStacks(separate.GateWeight, separate.UpWeight, 2),
		DownWeight:     separate.DownWeight,
		UseFusedGateUp: true,
	}

	gotSeparate := separate.Forward(x, indices, cfg)
	gotFused := fused.Forward(x, indices, cfg)
	mlx.Eval(gotSeparate, gotFused)

	gotFusedF32 := gotFused.AsType(mlx.DTypeFloat32)
	gotSeparateF32 := gotSeparate.AsType(mlx.DTypeFloat32)
	mlx.Eval(gotFusedF32, gotSeparateF32)
	assertFloatSlicesClose(t, gotFusedF32.Floats(), gotSeparateF32.Floats(), 1e-5)
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
