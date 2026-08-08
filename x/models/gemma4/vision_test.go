package gemma4

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
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
		InputProj:  smallLinear(hidden, 16*16*3),
		PosTableX:  onesArr(64, hidden),
		PosTableY:  onesArr(64, hidden),
		Layers:     []*VisionLayer{layer(), layer()},
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

// testImagePNG builds a deterministic 480×336 image: at budget 70 the
// budget-fill factor is exactly 1.0 (480·336 = 70·48²), so resize is the
// identity and patch values are exactly predictable.
func testImagePNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x % 256), G: uint8(y % 256), B: uint8((x + y) % 256), A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func unifiedTestModel() *Model {
	cfg := &VisionConfig{
		ModelType: "gemma4_unified_vision", MMEmbedDim: 16, MMPosembSize: 1120,
		ModelPatchSize: 48, PatchSize: 16, PoolingKernelSize: 3, RMSNormEps: 1e-6,
	}
	return &Model{TextConfig: &TextConfig{}, VisionCfg: cfg,
		MMTokens: multimodalTokens{BOI: 255999, EOI: 258882, Image: 258880}}
}

func towerTestModel() *Model {
	return &Model{TextConfig: &TextConfig{}, VisionCfg: towerTestConfig(2),
		MMTokens: multimodalTokens{BOI: 255999, EOI: 258882, Image: 258880}}
}

func TestNewVisionInputUnifiedLayout(t *testing.T) {
	png := testImagePNG(t, 480, 336)
	opts := api.Options{}
	opts.ImageMaxTokens = 70

	in, err := unifiedTestModel().NewVisionInput(png, opts)
	if err != nil {
		t.Fatal(err)
	}
	vi := in.(*visionInput)
	if in.SoftTokens() != 70 {
		t.Fatalf("SoftTokens = %d, want 70", in.SoftTokens())
	}
	if vi.n != 70 || vi.patchDim != 6912 || vi.gridW != 10 || vi.gridH != 7 {
		t.Fatalf("unified geometry wrong: n=%d patchDim=%d grid=%dx%d", vi.n, vi.patchDim, vi.gridW, vi.gridH)
	}
	// Patch (px=2, py=1), inner pixel (dx=5, dy=7) → source pixel (101, 55):
	// channel-fastest layout, values pixel/255 with no further scaling.
	i := 1*10 + 2
	if vi.xs[i] != 2 || vi.ys[i] != 1 {
		t.Fatalf("positions[%d] = (%d,%d), want (2,1)", i, vi.xs[i], vi.ys[i])
	}
	base := i*6912 + (7*48+5)*3
	for c, want := range []float32{101.0 / 255, 55.0 / 255, 156.0 / 255} {
		if got := vi.patches[base+c]; !close32(got, want, 1e-3) {
			t.Fatalf("unified patch value[ch=%d] = %v, want %v", c, got, want)
		}
	}
}

func TestNewVisionInputTowerLayout(t *testing.T) {
	png := testImagePNG(t, 480, 336)
	opts := api.Options{}
	opts.ImageMaxTokens = 70

	in, err := towerTestModel().NewVisionInput(png, opts)
	if err != nil {
		t.Fatal(err)
	}
	vi := in.(*visionInput)
	if in.SoftTokens() != 70 {
		t.Fatalf("SoftTokens = %d, want 70 (48px soft-token grid)", in.SoftTokens())
	}
	if vi.n != 630 || vi.patchDim != 768 || vi.gridW != 30 || vi.gridH != 21 {
		t.Fatalf("tower geometry wrong: n=%d patchDim=%d grid=%dx%d", vi.n, vi.patchDim, vi.gridW, vi.gridH)
	}
	// Patch (px=3, py=2), inner pixel (dx=4, dy=9) → source pixel (52, 41):
	// tower patches carry 2x−1 values.
	i := 2*30 + 3
	if vi.xs[i] != 3 || vi.ys[i] != 2 {
		t.Fatalf("positions[%d] = (%d,%d), want (3,2)", i, vi.xs[i], vi.ys[i])
	}
	base := i*768 + (9*16+4)*3
	for c, want := range []float32{2*52.0/255 - 1, 2*41.0/255 - 1, 2*93.0/255 - 1} {
		if got := vi.patches[base+c]; !close32(got, want, 1e-3) {
			t.Fatalf("tower patch value[ch=%d] = %v, want %v", c, got, want)
		}
	}
}

func TestNewVisionInputSizing(t *testing.T) {
	png := testImagePNG(t, 640, 480)
	in, err := unifiedTestModel().NewVisionInput(png, api.Options{})
	if err != nil {
		t.Fatal(err)
	}
	// Default ceiling 1120: 640×480 budget-fills to 1824×1344 = 38×28 = 1064.
	if in.SoftTokens() != 1064 {
		t.Fatalf("SoftTokens = %d, want 1064", in.SoftTokens())
	}
	tw, th := llm.BudgetFillSize(640, 480, llm.Gemma4ImageAlign, api.DefaultImageMaxTokens)
	if in.SoftTokens() != (tw/48)*(th/48) {
		t.Fatalf("SoftTokens = %d disagrees with llm.BudgetFillSize %dx%d", in.SoftTokens(), tw, th)
	}
}

func TestNewVisionInputRejectsJunk(t *testing.T) {
	if _, err := unifiedTestModel().NewVisionInput([]byte("not an image"), api.Options{}); err == nil {
		t.Fatal("expected decode error for junk bytes")
	}
}

func TestVisionTokens(t *testing.T) {
	m := unifiedTestModel()
	if !m.SupportsVision() {
		// SupportsVision requires a bound path; unbound test model reports false.
		t.Log("unbound model reports SupportsVision=false as designed")
	}
	boi, img, eoi := m.VisionTokens()
	if boi != 255999 || img != 258880 || eoi != 258882 {
		t.Fatalf("VisionTokens = (%d,%d,%d)", boi, img, eoi)
	}
}

func close32(a, b, tol float32) bool {
	d := a - b
	return d < tol && d > -tol
}

func TestVisionChunkMask(t *testing.T) {
	useMLXTestThread(t)

	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 6),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{6},
		BidiSpans:    [][2]int32{{1, 4}},
	}
	mask := visionChunkMask(b, 6, 2, mlx.DTypeFloat32)
	arr := mask.AsArray(b, 6, mlx.DTypeFloat32)
	mlx.Eval(arr)
	vals := arr.Floats() // [1,1,6,6]

	at := func(q, k int) float32 { return vals[q*6+k] }
	allowed := func(q, k int) bool { return at(q, k) == 0 }

	cases := []struct {
		q, k int
		want bool
		why  string
	}{
		{1, 3, true, "bidi future within block"},
		{3, 1, true, "bidi past beyond window 2"},
		{2, 2, true, "diagonal"},
		{5, 5, true, "diagonal outside block"},
		{5, 0, false, "window excludes old text for text query"},
		{5, 4, true, "window includes recent key"},
		{0, 5, false, "causal for text query"},
		{0, 2, false, "text query may not see future image"},
		{4, 1, false, "text query beyond window may not see image"},
		{3, 2, true, "bidi within block adjacent"},
	}
	for _, tc := range cases {
		if allowed(tc.q, tc.k) != tc.want {
			t.Errorf("(q=%d,k=%d) allowed=%v, want %v — %s", tc.q, tc.k, !tc.want, tc.want, tc.why)
		}
	}

	// Full-attention variant: window 0 disables the sliding restriction.
	full := visionChunkMask(b, 6, 0, mlx.DTypeFloat32)
	fvals := full.AsArray(b, 6, mlx.DTypeFloat32)
	mlx.Eval(fvals)
	fv := fvals.Floats()
	if fv[5*6+0] != 0 {
		t.Error("full attention must allow (q=5,k=0)")
	}
	if fv[0*6+5] == 0 {
		t.Error("full attention must stay causal outside blocks")
	}

	// Memoized: same window → same array instance.
	again := visionChunkMask(b, 6, 2, mlx.DTypeFloat32)
	if again != mask {
		t.Error("visionChunkMask not memoized per window")
	}
}
