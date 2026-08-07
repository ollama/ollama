package gemma4

import (
	"math"
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

// naiveRope2D mirrors mlx_vlm's apply_multidimensional_rope in float64:
// the head dim splits per spatial axis, each part rotated with its own
// coordinate; rotate_half stays within the part.
func naiveRope2D(x []float64, L, H, D int, xs, ys []int32, theta float64) []float64 {
	out := make([]float64, len(x))
	half := D / 2
	quarter := half / 2
	for l := 0; l < L; l++ {
		for h := 0; h < H; h++ {
			base := (l*H + h) * D
			for d := 0; d < 2; d++ {
				pos := float64(xs[l])
				if d == 1 {
					pos = float64(ys[l])
				}
				o := base + d*half
				for j := 0; j < quarter; j++ {
					ts := math.Pow(theta, 2*float64(j)/float64(half))
					angle := pos / ts
					c, s := math.Cos(angle), math.Sin(angle)
					x1, x2 := x[o+j], x[o+quarter+j]
					out[o+j] = x1*c - x2*s
					out[o+quarter+j] = x2*c + x1*s
				}
			}
		}
	}
	return out
}

func TestRope2DMatchesReference(t *testing.T) {
	useMLXTestThread(t)

	const L, H, D = 6, 4, 8
	xs, ys := make([]int32, L), make([]int32, L)
	for i := range xs { // 3×2 grid, row-major
		xs[i], ys[i] = int32(i%3), int32(i/3)
	}
	data := make([]float32, L*H*D)
	f64 := make([]float64, len(data))
	for i := range data {
		v := math.Sin(float64(i)*0.7) * 0.5
		data[i], f64[i] = float32(v), v
	}

	x := mlx.FromValues(data, 1, L, H, D)
	cosX, sinX := visionRopeTables(xs, D/2, 100, mlx.DTypeFloat32)
	cosY, sinY := visionRopeTables(ys, D/2, 100, mlx.DTypeFloat32)
	got := visionRope2D(x, cosX, sinX, cosY, sinY)
	mlx.Eval(got)

	want := naiveRope2D(f64, L, H, D, xs, ys, 100)
	gotF := got.AsType(mlx.DTypeFloat32).Floats()
	for i := range want {
		if diff := float64(gotF[i]) - want[i]; diff > 1e-4 || diff < -1e-4 {
			t.Fatalf("rope2d[%d] = %v, want %v", i, gotF[i], want[i])
		}
	}
}

func TestVisionAvgPoolMatchesNaive(t *testing.T) {
	useMLXTestThread(t)

	const gridW, gridH, D, k = 6, 6, 8, 3
	data := make([]float32, gridW*gridH*D)
	for i := range data {
		data[i] = float32(i%37) * 0.25
	}
	h := mlx.FromValues(data, 1, gridW*gridH, D)
	got := visionAvgPool(h, gridW, gridH, k)
	mlx.Eval(got)
	gotF := got.Floats()

	rB, cB := gridH/k, gridW/k
	if dims := got.Dims(); dims[1] != rB*cB {
		t.Fatalf("pooled length = %d, want %d", dims[1], rB*cB)
	}
	for rb := 0; rb < rB; rb++ {
		for cb := 0; cb < cB; cb++ {
			for d := 0; d < D; d++ {
				var sum float64
				for dy := 0; dy < k; dy++ {
					for dx := 0; dx < k; dx++ {
						patch := (rb*k+dy)*gridW + (cb*k + dx)
						sum += float64(data[patch*D+d])
					}
				}
				want := sum / (k * k)
				gi := (rb*cB+cb)*D + d
				if diff := float64(gotF[gi]) - want; diff > 1e-4 || diff < -1e-4 {
					t.Fatalf("pool[%d,%d,%d] = %v, want %v", rb, cb, d, gotF[gi], want)
				}
			}
		}
	}
}

// smallLinear fills a deterministic non-zero dense linear.
func smallLinear(out, in int) nn.LinearLayer {
	data := make([]float32, out*in)
	for i := range data {
		data[i] = float32((i%13)-6) * 0.05
	}
	return nn.NewLinear(mlx.FromValues(data, out, in), nil)
}

func onesArr(shape ...int) *mlx.Array {
	n := 1
	for _, s := range shape {
		n *= s
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = 1
	}
	return mlx.FromValues(data, shape...)
}

func TestVisionEmbedderForwardShapes(t *testing.T) {
	useMLXTestThread(t)

	const patchDim, embed, L = 48, 16, 4
	e := &VisionEmbedder{
		PatchLN1:   &nn.LayerNorm{Weight: onesArr(patchDim), Bias: mlx.Zeros(mlx.DTypeFloat32, patchDim)},
		PatchDense: smallLinear(embed, patchDim),
		PatchLN2:   &nn.LayerNorm{Weight: onesArr(embed), Bias: mlx.Zeros(mlx.DTypeFloat32, embed)},
		PosEmbX:    onesArr(8, embed),
		PosEmbY:    onesArr(8, embed),
		PosNorm:    &nn.LayerNorm{Weight: onesArr(embed), Bias: mlx.Zeros(mlx.DTypeFloat32, embed)},
	}
	patches := onesArr(1, L, patchDim)
	xs, ys := []int32{0, 1, 0, 1}, []int32{0, 0, 1, 1}
	out := e.Forward(patches, xs, ys)
	mlx.Eval(out)
	if d := out.Dims(); len(d) != 3 || d[0] != 1 || d[1] != L || d[2] != embed {
		t.Fatalf("embedder out dims = %v, want [1 %d %d]", d, L, embed)
	}
}

func TestVisionTowerForwardShapes(t *testing.T) {
	useMLXTestThread(t)

	cfg := towerTestConfig(2)
	const hidden, L = 8, 36 // 6×6 patch grid → 4 soft tokens
	layer := func() *VisionLayer {
		return &VisionLayer{
			InputNorm:    onesArr(hidden),
			PostAttnNorm: onesArr(hidden),
			PreFFNorm:    onesArr(hidden),
			PostFFNorm:   onesArr(hidden),
			Attn: &VisionAttention{
				QProj: smallLinear(hidden, hidden), KProj: smallLinear(hidden, hidden),
				VProj: smallLinear(hidden, hidden), OProj: smallLinear(hidden, hidden),
				QNormWeight: onesArr(4), KNormWeight: onesArr(4),
			},
			GateProj: smallLinear(16, hidden), UpProj: smallLinear(16, hidden), DownProj: smallLinear(hidden, 16),
		}
	}
	tw := &VisionTower{
		InputProj: smallLinear(hidden, 16*16*3),
		PosTableX: onesArr(64, hidden),
		PosTableY: onesArr(64, hidden),
		Layers:    []*VisionLayer{layer(), layer()},
		NegStdBias: mlx.Zeros(mlx.DTypeFloat32, hidden),
		StdScale:   onesArr(hidden),
	}
	patches := onesArr(1, L, 16*16*3)
	xs, ys := make([]int32, L), make([]int32, L)
	for i := range xs {
		xs[i], ys[i] = int32(i%6), int32(i/6)
	}
	out := tw.Forward(patches, xs, ys, 6, 6, cfg)
	mlx.Eval(out)
	if d := out.Dims(); len(d) != 3 || d[0] != 1 || d[1] != 4 || d[2] != hidden {
		t.Fatalf("tower out dims = %v, want [1 4 %d]", d, hidden)
	}
}
