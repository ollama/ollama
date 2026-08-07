package gemma4

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

func TestParseVisionConfigUnified(t *testing.T) {
	configJSON := `{
		"architectures": ["Gemma4UnifiedForConditionalGeneration"],
		"model_type": "gemma4_unified",
		"boi_token_id": 255999,
		"eoi_token_id": 258882,
		"image_token_id": 258880,
		"audio_token_id": 258881,
		"text_config": {"hidden_size": 3840},
		"vision_config": {
			"model_type": "gemma4_unified_vision",
			"mm_embed_dim": 3840,
			"mm_posemb_size": 1120,
			"model_patch_size": 48,
			"num_soft_tokens": 280,
			"output_proj_dims": 3840,
			"patch_size": 16,
			"pooling_kernel_size": 3,
			"rms_norm_eps": 1e-06
		}
	}`

	cfg, toks, err := parseVisionConfig([]byte(configJSON))
	if err != nil {
		t.Fatal(err)
	}
	if cfg == nil {
		t.Fatal("expected vision config, got nil")
	}
	if !cfg.IsUnified() {
		t.Errorf("IsUnified() = false for %q", cfg.ModelType)
	}
	if cfg.MMEmbedDim != 3840 || cfg.MMPosembSize != 1120 || cfg.ModelPatchSize != 48 {
		t.Errorf("unified dims wrong: %+v", cfg)
	}
	if cfg.PatchSize != 16 || cfg.PoolingKernelSize != 3 {
		t.Errorf("shared dims wrong: %+v", cfg)
	}
	if toks.BOI != 255999 || toks.Image != 258880 || toks.EOI != 258882 {
		t.Errorf("token ids wrong: %+v", toks)
	}
}

func TestParseVisionConfigTower(t *testing.T) {
	configJSON := `{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"boi_token_id": 255999,
		"eoi_token_id": 258882,
		"image_token_id": 258880,
		"vision_config": {
			"model_type": "gemma4_vision",
			"hidden_size": 1152,
			"intermediate_size": 4304,
			"num_attention_heads": 16,
			"head_dim": 72,
			"num_hidden_layers": 27,
			"position_embedding_size": 10240,
			"standardize": true,
			"rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
			"patch_size": 16,
			"pooling_kernel_size": 3,
			"rms_norm_eps": 1e-06
		}
	}`

	cfg, _, err := parseVisionConfig([]byte(configJSON))
	if err != nil {
		t.Fatal(err)
	}
	if cfg == nil {
		t.Fatal("expected vision config, got nil")
	}
	if cfg.IsUnified() {
		t.Error("IsUnified() = true for tower config")
	}
	if cfg.HiddenSize != 1152 || cfg.HeadDim != 72 || cfg.NumHiddenLayers != 27 ||
		cfg.PositionEmbeddingSize != 10240 || !cfg.Standardize {
		t.Errorf("tower dims wrong: %+v", cfg)
	}
	if cfg.RopeParameters.RopeTheta != 100 {
		t.Errorf("rope theta = %v, want 100", cfg.RopeParameters.RopeTheta)
	}
}

func TestParseVisionConfigTextOnly(t *testing.T) {
	cfg, _, err := parseVisionConfig([]byte(`{"architectures": ["Gemma4ForCausalLM"], "hidden_size": 640}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg != nil {
		t.Fatalf("expected nil vision config for text-only checkpoint, got %+v", cfg)
	}
}

// zerosF32 builds a dense placeholder tensor; binding only keys on names.
func zerosF32(shape ...int) *mlx.Array {
	return mlx.Zeros(mlx.DTypeFloat32, shape...)
}

func unifiedVisionTensors() map[string]*mlx.Array {
	const patchDim, embed = 48 * 48 * 3, 32
	return map[string]*mlx.Array{
		"model.vision_embedder.patch_ln1.weight":         zerosF32(patchDim),
		"model.vision_embedder.patch_ln1.bias":           zerosF32(patchDim),
		"model.vision_embedder.patch_dense.weight":       zerosF32(embed, patchDim),
		"model.vision_embedder.patch_dense.bias":         zerosF32(embed),
		"model.vision_embedder.patch_ln2.weight":         zerosF32(embed),
		"model.vision_embedder.patch_ln2.bias":           zerosF32(embed),
		"model.vision_embedder.pos_embedding":            zerosF32(1120, 2, embed),
		"model.vision_embedder.pos_norm.weight":          zerosF32(embed),
		"model.vision_embedder.pos_norm.bias":            zerosF32(embed),
		"model.embed_vision.embedding_projection.weight": zerosF32(64, embed),
	}
}

func TestLoadVisionWeightsUnified(t *testing.T) {
	useMLXTestThread(t)

	tensors := unifiedVisionTensors()
	m := &Model{
		TextConfig: &TextConfig{},
		VisionCfg: &VisionConfig{
			ModelType: "gemma4_unified_vision", MMEmbedDim: 32, MMPosembSize: 1120,
			ModelPatchSize: 48, PatchSize: 16, PoolingKernelSize: 3, RMSNormEps: 1e-6,
		},
	}
	linears := model.NewLinearFactory(tensors, 64, 4, "affine", nil)
	if err := m.loadVisionWeights(tensors, linears); err != nil {
		t.Fatal(err)
	}
	e := m.VisionEmbedder
	if e == nil {
		t.Fatal("VisionEmbedder not bound")
	}
	if m.VisionTower != nil {
		t.Error("VisionTower bound for unified checkpoint")
	}
	if e.PatchLN1 == nil || e.PatchLN2 == nil || e.PosNorm == nil || e.PatchDense == nil {
		t.Fatalf("embedder incompletely bound: %+v", e)
	}
	if _, ok := e.PatchDense.(*nn.Linear); !ok {
		t.Errorf("dense patch_dense should bind as *nn.Linear, got %T", e.PatchDense)
	}
	if e.PosEmbX == nil || e.PosEmbY == nil {
		t.Fatal("position embedding slices not bound")
	}
	if dims := e.PosEmbX.Dims(); len(dims) != 2 || dims[0] != 1120 || dims[1] != 32 {
		t.Errorf("PosEmbX dims = %v, want [1120 32]", dims)
	}
	if m.EmbedVisionProj == nil {
		t.Fatal("EmbedVisionProj not bound")
	}
}

func TestLoadVisionWeightsUnifiedQuantizedKeepsBias(t *testing.T) {
	useMLXTestThread(t)

	tensors := unifiedVisionTensors()
	// nvfp4 double-scale patch_dense as shipped in the 12b manifest: the real
	// layer bias must survive alongside the quant sidecars.
	tensors["model.vision_embedder.patch_dense.weight"] = mlx.Zeros(mlx.DTypeUint32, 32, 864)
	tensors["model.vision_embedder.patch_dense.weight_scale"] = mlx.Zeros(mlx.DTypeUint8, 32, 432)
	tensors["model.vision_embedder.patch_dense.weight.global_scale"] = zerosF32(1)

	tq := map[string]*model.TensorQuantInfo{
		"model.vision_embedder.patch_dense.weight": {QuantType: "nvfp4", GroupSize: 16},
	}
	m := &Model{
		TextConfig: &TextConfig{},
		VisionCfg: &VisionConfig{
			ModelType: "gemma4_unified_vision", MMEmbedDim: 32, MMPosembSize: 1120,
			ModelPatchSize: 48, PatchSize: 16, PoolingKernelSize: 3, RMSNormEps: 1e-6,
		},
	}
	linears := model.NewLinearFactory(tensors, 16, 4, "nvfp4", tq)
	if err := m.loadVisionWeights(tensors, linears); err != nil {
		t.Fatal(err)
	}
	ql, ok := m.VisionEmbedder.PatchDense.(*nn.QuantizedLinear)
	if !ok {
		t.Fatalf("patch_dense should bind as *nn.QuantizedLinear, got %T", m.VisionEmbedder.PatchDense)
	}
	if ql.Bias == nil {
		t.Error("quantized patch_dense lost its real bias")
	}
	if ql.GlobalScale == nil {
		t.Error("double-scale nvfp4 global scale not bound")
	}
}

func towerVisionTensors(layers int) map[string]*mlx.Array {
	const hidden, inter, patchDim = 8, 16, 16 * 16 * 3
	tensors := map[string]*mlx.Array{
		"model.vision_tower.patch_embedder.input_proj.weight":        zerosF32(hidden, patchDim),
		"model.vision_tower.patch_embedder.position_embedding_table": zerosF32(2, 64, hidden),
		"model.vision_tower.std_bias":                                zerosF32(hidden),
		"model.vision_tower.std_scale":                               zerosF32(hidden),
		"model.embed_vision.embedding_projection.weight":             zerosF32(24, hidden),
	}
	for i := 0; i < layers; i++ {
		lp := "model.vision_tower.encoder.layers." + string(rune('0'+i)) + "."
		for _, n := range []string{"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"} {
			tensors[lp+n+".weight"] = zerosF32(hidden)
		}
		tensors[lp+"self_attn.q_norm.weight"] = zerosF32(4)
		tensors[lp+"self_attn.k_norm.weight"] = zerosF32(4)
		for _, n := range []string{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"} {
			tensors[lp+n+".linear.weight"] = zerosF32(hidden, hidden)
		}
		tensors[lp+"mlp.gate_proj.linear.weight"] = zerosF32(inter, hidden)
		tensors[lp+"mlp.up_proj.linear.weight"] = zerosF32(inter, hidden)
		tensors[lp+"mlp.down_proj.linear.weight"] = zerosF32(hidden, inter)
	}
	return tensors
}

func towerTestConfig(layers int32) *VisionConfig {
	cfg := &VisionConfig{
		ModelType: "gemma4_vision", HiddenSize: 8, IntermediateSize: 16,
		NumAttentionHeads: 2, HeadDim: 4, NumHiddenLayers: layers,
		PositionEmbeddingSize: 64, Standardize: true,
		PatchSize: 16, PoolingKernelSize: 3, RMSNormEps: 1e-6,
	}
	cfg.RopeParameters.RopeTheta = 100
	return cfg
}

func TestLoadVisionWeightsTower(t *testing.T) {
	useMLXTestThread(t)

	tensors := towerVisionTensors(2)
	m := &Model{TextConfig: &TextConfig{}, VisionCfg: towerTestConfig(2)}
	linears := model.NewLinearFactory(tensors, 64, 4, "affine", nil)
	if err := m.loadVisionWeights(tensors, linears); err != nil {
		t.Fatal(err)
	}
	tw := m.VisionTower
	if tw == nil {
		t.Fatal("VisionTower not bound")
	}
	if m.VisionEmbedder != nil {
		t.Error("VisionEmbedder bound for tower checkpoint")
	}
	if len(tw.Layers) != 2 {
		t.Fatalf("layers = %d, want 2", len(tw.Layers))
	}
	l := tw.Layers[1]
	if l.InputNorm == nil || l.PostAttnNorm == nil || l.PreFFNorm == nil || l.PostFFNorm == nil {
		t.Error("block norms incompletely bound")
	}
	if l.Attn.QProj == nil || l.Attn.KProj == nil || l.Attn.VProj == nil || l.Attn.OProj == nil {
		t.Error("attention projections incompletely bound")
	}
	if l.Attn.QNormWeight == nil || l.Attn.KNormWeight == nil {
		t.Error("q/k norm weights not bound")
	}
	if l.GateProj == nil || l.UpProj == nil || l.DownProj == nil {
		t.Error("mlp projections incompletely bound")
	}
	if tw.NegStdBias == nil || tw.StdScale == nil {
		t.Error("standardization tensors not bound")
	}
	if dims := tw.PosTableX.Dims(); len(dims) != 2 || dims[0] != 64 || dims[1] != 8 {
		t.Errorf("PosTableX dims = %v, want [64 8]", dims)
	}
}

func TestLoadVisionWeightsMissingTensor(t *testing.T) {
	useMLXTestThread(t)

	tensors := towerVisionTensors(1)
	delete(tensors, "model.vision_tower.encoder.layers.0.self_attn.k_norm.weight")
	m := &Model{TextConfig: &TextConfig{}, VisionCfg: towerTestConfig(1)}
	linears := model.NewLinearFactory(tensors, 64, 4, "affine", nil)
	err := m.loadVisionWeights(tensors, linears)
	if err == nil {
		t.Fatal("expected explicit error for missing tensor")
	}
	if !strings.Contains(err.Error(), "k_norm") {
		t.Errorf("error %q does not name the missing tensor", err)
	}
}
