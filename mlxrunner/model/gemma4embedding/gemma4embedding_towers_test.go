package gemma4embedding

import (
	"bytes"
	"image"
	"image/png"
	"math"
	"strconv"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/gemma4"
)

// tinyTextConfig is the smallest Config that still exercises the load path:
// one sliding layer, one full layer, PLE projections on.
func tinyTextConfig() *Config {
	cfgJSON := `{
		"text_config": {
			"hidden_size": 16,
			"num_hidden_layers": 2,
			"intermediate_size": 32,
			"num_attention_heads": 2,
			"num_key_value_heads": 1,
			"head_dim": 8,
			"vocab_size": 32,
			"rms_norm_eps": 1e-6,
			"max_position_embeddings": 64,
			"sliding_window": 4,
			"layer_types": ["sliding_attention", "full_attention"],
			"hidden_size_per_layer_input": 8,
			"vocab_size_per_layer_input": 0,
			"embedding_dim": 8
		}
	}`
	cfg, err := parseConfig([]byte(cfgJSON))
	if err != nil {
		panic(err)
	}
	return cfg
}

func towerTestMat(seed, rows, cols int) *mlx.Array {
	v := make([]float32, rows*cols)
	for i := range v {
		v[i] = float32(((seed*97+i*31)%89)-44) / 89
	}
	return mlx.FromValues(v, rows, cols)
}

func towerTestVec(seed, n int) *mlx.Array {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(((seed*53+i*17)%67)+1) / 34
	}
	return mlx.FromValues(v, n)
}

// tinyTextTensors builds the full tensor set tinyTextConfig's LoadWeights
// consumes, with the "model." root the HF class nesting adds.
func tinyTextTensors(cfg *Config) map[string]*mlx.Array {
	tc := &cfg.TextConfig
	hidden := int(tc.HiddenSize)
	headDim := int(tc.HeadDim)
	kvHeads := int(tc.NumKeyValueHeads)
	heads := int(tc.NumAttentionHeads)
	vocab := int(tc.VocabSize)
	inter := int(tc.IntermediateSize)
	pleHidden := int(tc.HiddenSizePerLayer)
	embDim := int(cfg.TextConfig.EmbeddingDim)

	prefix := "model.language_model."
	tensors := map[string]*mlx.Array{
		prefix + "embed_tokens.weight":      towerTestMat(1, vocab, hidden),
		prefix + "norm.weight":              towerTestVec(2, hidden),
		"model.embedding_projection.weight": towerTestMat(3, embDim, hidden),
	}
	if pleHidden > 0 {
		tensors[prefix+"per_layer_model_projection.weight"] = towerTestMat(4, int(tc.NumHiddenLayers)*pleHidden, hidden)
		tensors[prefix+"per_layer_projection_norm.weight"] = towerTestVec(5, pleHidden)
	}
	for i := range int(tc.NumHiddenLayers) {
		lp := prefix + "layers." + strconv.Itoa(i) + "."
		tensors[lp+"input_layernorm.weight"] = towerTestVec(10+i, hidden)
		tensors[lp+"post_attention_layernorm.weight"] = towerTestVec(20+i, hidden)
		tensors[lp+"pre_feedforward_layernorm.weight"] = towerTestVec(30+i, hidden)
		tensors[lp+"post_feedforward_layernorm.weight"] = towerTestVec(40+i, hidden)
		tensors[lp+"self_attn.q_proj.weight"] = towerTestMat(50+i, heads*headDim, hidden)
		tensors[lp+"self_attn.k_proj.weight"] = towerTestMat(60+i, kvHeads*headDim, hidden)
		tensors[lp+"self_attn.v_proj.weight"] = towerTestMat(70+i, kvHeads*headDim, hidden)
		tensors[lp+"self_attn.o_proj.weight"] = towerTestMat(80+i, hidden, heads*headDim)
		tensors[lp+"self_attn.q_norm.weight"] = towerTestVec(90+i, headDim)
		tensors[lp+"self_attn.k_norm.weight"] = towerTestVec(100+i, headDim)
		tensors[lp+"mlp.gate_proj.weight"] = towerTestMat(110+i, inter, hidden)
		tensors[lp+"mlp.up_proj.weight"] = towerTestMat(120+i, inter, hidden)
		tensors[lp+"mlp.down_proj.weight"] = towerTestMat(130+i, hidden, inter)
		tensors[lp+"layer_scalar"] = towerTestVec(140+i, 1)
		if pleHidden > 0 {
			// InputGate: hidden -> pleHidden; Projection: pleHidden -> hidden.
			// PostNorm is hidden-wide (it norms the projected, hidden-dim
			// residual), matching the real checkpoint's [hidden] shape.
			tensors[lp+"per_layer_input_gate.weight"] = towerTestMat(150+i, pleHidden, hidden)
			tensors[lp+"per_layer_projection.weight"] = towerTestMat(160+i, hidden, pleHidden)
			tensors[lp+"post_per_layer_input_norm.weight"] = towerTestVec(170+i, hidden)
		}
	}
	return tensors
}

// tinyVisionTensors adds a one-layer gemma4_vision tower (hidden 8, one
// head) plus its embed_vision projection, matching the key naming and
// shapes gemma4.LoadVisionTower consumes.
func tinyVisionTensors(tensors map[string]*mlx.Array, embDim int) *gemma4.VisionConfig {
	const hidden = 8
	cfg := &gemma4.VisionConfig{
		ModelType:         "gemma4_vision",
		HiddenSize:        hidden,
		NumHiddenLayers:   1,
		NumAttentionHeads: 1,
		HeadDim:           hidden,
		PatchSize:         2,
		PoolingKernelSize: 1,
		RMSNormEps:        1e-6,
		Standardize:       true,
	}

	posRows := 8
	tensors["vision_tower.patch_embedder.input_proj.weight"] = towerTestMat(201, hidden, 2*2*3)
	{
		v := make([]float32, 2*posRows*hidden)
		for i := range v {
			v[i] = float32(((202*97+i*31)%89)-44) / 89
		}
		tensors["vision_tower.patch_embedder.position_embedding_table"] = mlx.FromValues(v, 2, posRows, hidden)
	}
	lp := "vision_tower.encoder.layers.0."
	for _, name := range []string{
		"input_layernorm.weight", "post_attention_layernorm.weight",
		"pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
		"self_attn.q_norm.weight", "self_attn.k_norm.weight",
	} {
		tensors[lp+name] = towerTestVec(203, hidden)
	}
	for i, name := range []string{
		"self_attn.q_proj.weight", "self_attn.k_proj.weight",
		"self_attn.v_proj.weight", "self_attn.o_proj.weight",
		"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
	} {
		tensors[lp+name] = towerTestMat(210+i, hidden, hidden)
	}
	tensors["vision_tower.std_bias"] = towerTestVec(220, hidden)
	tensors["vision_tower.std_scale"] = towerTestVec(221, hidden)
	tensors["embed_vision.embedding_projection.weight"] = towerTestMat(222, embDim, hidden)
	return cfg
}

// tinyAudioTensors adds a one-layer gemma4_audio tower (hidden 128 to keep
// the mel subsample arithmetic intact) plus embed_audio, matching
// gemma4.LoadAudioTower's key naming and the wrapper test's shapes.
func tinyAudioTensors(tensors map[string]*mlx.Array, embDim int) *gemma4.AudioConfig {
	const hidden = 128
	cfg := &gemma4.AudioConfig{
		ModelType:         "gemma4_audio",
		HiddenSize:        hidden,
		NumHiddenLayers:   1,
		NumAttentionHeads: 8,
		ConvKernelSize:    5,
		ResidualWeight:    0.5,
		ChunkSize:         12,
		ContextLeft:       13,
		LogitCap:          50,
		InvalidLogit:      -1e9,
		RMSNormEps:        1e-6,
		GradientClipping:  1e10,
	}

	{
		v := make([]float32, (hidden/4)*1*3*3)
		for i := range v {
			v[i] = float32(((301*97+i*31)%89)-44) / (89 * 3)
		}
		tensors["audio_tower.subsample_conv_projection.layer0.conv.weight"] = mlx.FromValues(v, hidden/4, 1, 3, 3)
	}
	tensors["audio_tower.subsample_conv_projection.layer0.norm.weight"] = towerTestVec(302, hidden/4)
	{
		v := make([]float32, hidden*3*3*(hidden/4))
		for i := range v {
			v[i] = float32(((303*97+i*31)%89)-44) / (89 * 20)
		}
		tensors["audio_tower.subsample_conv_projection.layer1.conv.weight"] = mlx.FromValues(v, hidden, hidden/4, 3, 3)
	}
	tensors["audio_tower.subsample_conv_projection.layer1.norm.weight"] = towerTestVec(304, hidden)
	// After two stride-2 convs over the 128-bin mel axis: 128 -> 32.
	tensors["audio_tower.subsample_conv_projection.input_proj_linear.weight"] = towerTestMat(305, hidden, 32*hidden)

	lp := "audio_tower.layers.0."
	headDim := hidden / int(cfg.NumAttentionHeads)
	for _, name := range []string{
		"feed_forward1.pre_layer_norm.weight", "feed_forward1.post_layer_norm.weight",
		"feed_forward2.pre_layer_norm.weight", "feed_forward2.post_layer_norm.weight",
		"lconv1d.pre_layer_norm.weight", "lconv1d.conv_norm.weight",
		"norm_pre_attn.weight", "norm_post_attn.weight", "norm_out.weight",
	} {
		tensors[lp+name] = towerTestVec(306, hidden)
	}
	tensors[lp+"self_attn.per_dim_scale"] = towerTestVec(307, headDim)
	for i, name := range []string{
		"feed_forward1.ffw_layer_1.weight", "feed_forward1.ffw_layer_2.weight",
		"feed_forward2.ffw_layer_1.weight", "feed_forward2.ffw_layer_2.weight",
		"self_attn.q_proj.weight", "self_attn.k_proj.weight",
		"self_attn.v_proj.weight", "self_attn.post.weight",
		"self_attn.relative_k_proj.weight",
	} {
		tensors[lp+name] = towerTestMat(310+i, hidden, hidden)
	}
	tensors[lp+"lconv1d.linear_start.weight"] = towerTestMat(320, 2*hidden, hidden)
	{
		v := make([]float32, hidden*int(cfg.ConvKernelSize))
		for i := range v {
			v[i] = float32(((321*97+i*31)%89)-44) / 445
		}
		tensors[lp+"lconv1d.depthwise_conv1d.weight"] = mlx.FromValues(v, hidden, 1, int(cfg.ConvKernelSize))
	}
	tensors[lp+"lconv1d.linear_end.weight"] = towerTestMat(322, hidden, hidden)
	tensors["audio_tower.output_proj.weight"] = towerTestMat(323, hidden, hidden)
	tensors["embed_audio.embedding_projection.weight"] = towerTestMat(324, embDim, hidden)
	return cfg
}

// loadModelTensors wires a Model from cfg + tensors the way LoadWeights
// expects, and returns the load error (nil on success).
func loadModelTensors(cfg *Config, tensors map[string]*mlx.Array) (*Model, error) {
	m := &Model{Layers: make([]*DecoderLayer, cfg.NumHiddenLayers), Cfg: cfg}
	m.Cfg.QuantGroupSize, m.Cfg.QuantBits, m.Cfg.QuantMode = model.QuantizationParams("")
	err := m.LoadWeights(tensors)
	return m, err
}

// TestTowerLoadingVariants covers all four embeddinggemma-2 variant shapes
// with synthetic tensors: text-only, text-vision, text-audio, full.
func TestTowerLoadingVariants(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		withVision, withAudio bool
	}{
		{"text-only", false, false},
		{"text-vision", true, false},
		{"text-audio", false, true},
		{"full", true, true},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			cfg := tinyTextConfig()
			tensors := tinyTextTensors(cfg)
			if tc.withVision {
				cfg.VisionConfig = tinyVisionTensors(tensors, int(cfg.TextConfig.EmbeddingDim))
			}
			if tc.withAudio {
				cfg.AudioConfig = tinyAudioTensors(tensors, int(cfg.TextConfig.EmbeddingDim))
			}

			m, err := loadModelTensors(cfg, tensors)
			if err != nil {
				t.Errorf("LoadWeights: %v", err)
				return
			}

			if got := m.VisionTower != nil; got != tc.withVision {
				t.Errorf("VisionTower non-nil = %v, want %v", got, tc.withVision)
			}
			if got := m.EmbedVision != nil; got != tc.withVision {
				t.Errorf("EmbedVision non-nil = %v, want %v", got, tc.withVision)
			}
			if got := m.AudioTower != nil; got != tc.withAudio {
				t.Errorf("AudioTower non-nil = %v, want %v", got, tc.withAudio)
			}
			if got := m.EmbedAudio != nil; got != tc.withAudio {
				t.Errorf("EmbedAudio non-nil = %v, want %v", got, tc.withAudio)
			}
		})
	}
}

// TestTextForwardUnchangedWithoutTowers pins the no-tower text path: a
// forward through the synthetic text-only model must succeed and produce
// finite outputs of the expected shape, proving tower wiring didn't disturb
// the existing path.
func TestTextForwardUnchangedWithoutTowers(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := tinyTextConfig()
		m, err := loadModelTensors(cfg, tinyTextTensors(cfg))
		if err != nil {
			t.Errorf("LoadWeights: %v", err)
			return
		}
		if m.VisionTower != nil || m.AudioTower != nil || m.EmbedVision != nil || m.EmbedAudio != nil {
			t.Error("towers should be nil on a text-only tensor set")
			return
		}

		toks := []int32{1, 5, 9, 2}
		out, _ := m.Forward(&batch.Batch{
			InputIDs:     mlx.FromValues(toks, 1, len(toks)),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(toks))},
		}, nil)
		mlx.Eval(out)
		dims := out.Dims()
		if len(dims) != 3 || dims[1] != len(toks) || dims[2] != int(cfg.TextConfig.EmbeddingDim) {
			t.Errorf("output dims %v, want [1 %d %d]", dims, len(toks), cfg.TextConfig.EmbeddingDim)
			return
		}
		for i, v := range out.AsType(mlx.DTypeFloat32).Floats() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("non-finite output at %d: %v", i, v)
				return
			}
		}
	})
}

// TestTowerLoadErrors checks the hard-error path (tower weights without the
// matching config subtree) and the inverse (config subtrees with no tower
// tensors must load text-only — presence of config never triggers a load).
func TestTowerLoadErrors(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// Vision weights but no vision_config: must error.
		cfg := tinyTextConfig()
		tensors := tinyTextTensors(cfg)
		tinyVisionTensors(tensors, int(cfg.TextConfig.EmbeddingDim))
		if _, err := loadModelTensors(cfg, tensors); err == nil {
			t.Error("expected error for vision weights without vision_config")
		}

		// Audio weights but no audio_config: must error.
		cfg = tinyTextConfig()
		tensors = tinyTextTensors(cfg)
		tinyAudioTensors(tensors, int(cfg.TextConfig.EmbeddingDim))
		if _, err := loadModelTensors(cfg, tensors); err == nil {
			t.Error("expected error for audio weights without audio_config")
		}

		// Configs present but no tower tensors: loads text-only.
		cfg2 := tinyTextConfig()
		cfg2.VisionConfig = &gemma4.VisionConfig{ModelType: "gemma4_vision"}
		cfg2.AudioConfig = &gemma4.AudioConfig{ModelType: "gemma4_audio"}
		m2, err := loadModelTensors(cfg2, tinyTextTensors(cfg2))
		if err != nil {
			t.Errorf("LoadWeights with config-only subtrees: %v", err)
			return
		}
		if m2.VisionTower != nil || m2.AudioTower != nil {
			t.Error("config-only subtrees must not trigger tower loads")
		}
	})
}

// TestMediaForwardDiffsFromTextOnly is the Task 4 model-side prove-out: a
// text-vision variant must (a) encode a synthetic image through the vision
// tower and (b) splice the tower features into the token stream, changing
// the forward output relative to the same token stream run without media.
// The image is hand-built patch data (PrepareMedia's PNG decode is covered
// in gemma4's own tests); the PreparedRequest the runner would get is
// rebuilt by hand so EncodeMedia is the code path under test.
func TestMediaForwardDiffsFromTextOnly(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := tinyTextConfig()
		tensors := tinyTextTensors(cfg)
		// embed_vision must project into the token-embedding stream, i.e.
		// the text hidden size — that is what scatterMedia rewires rows
		// with. (The tiny text config's EmbeddingDim is the OUTPUT dim,
		// applied later by model.embedding_projection.)
		textHidden := int(cfg.TextConfig.HiddenSize)
		cfg.VisionConfig = tinyVisionTensors(tensors, textHidden)
		// parseConfig fills RopeTheta only for configs that come through
		// its JSON unmarshal; the helper-built subtree needs it set the
		// same way (0 would trip 1/Pow(0, i) in the RoPE tables).
		cfg.VisionConfig.RopeTheta = 100

		m, err := loadModelTensors(cfg, tensors)
		if err != nil {
			t.Errorf("LoadWeights: %v", err)
			return
		}
		if m.VisionTower == nil || m.EmbedVision == nil {
			t.Error("vision tower/embedder must be loaded for the media path")
			return
		}

		// One synthetic image: a 4x4 patch grid (16 patches over a PatchSize
		// 2 tower, PoolingKernelSize 1 keeps NumSoftTokens == 16), with
		// deterministic pixel values.
		const grid = 4
		const patchD = 2 * 2 * 3
		const patches = grid * grid
		pixels := make([]float32, patches*patchD)
		positions := make([]int32, 2*patches)
		for p := range patches {
			positions[2*p] = int32(p % grid)
			positions[2*p+1] = int32(p / grid)
			for d := range patchD {
				pixels[p*patchD+d] = float32((p*31+d*7)%97) / 97
			}
		}
		geom := gemma4.ImageGeometry{PatchesW: grid, PatchesH: grid, NumSoftTokens: patches}

		// Token stream: one text token, then the 16 image soft-token
		// placeholders at positions [1,17), then two more text tokens.
		// Input token values only matter for the embedding lookup; every id
		// stays inside the tiny vocab (32).
		const mediaPos = 1
		toks := make([]int32, 0, 3+patches)
		toks = append(toks, 3)
		for range patches {
			toks = append(toks, 7)
		}
		toks = append(toks, 11, 13)

		// EncodeMedia is the runner-facing entry: hand it the PreparedRequest
		// PrepareMedia would have produced for this image at position 1.
		prepared := &model.PreparedRequest{
			Tokens: toks,
			Items: []model.PreparedItem{{
				Range:     [2]int{mediaPos, mediaPos + patches},
				Source:    0,
				MediaData: pixels,
				Dims:      []int{patches, patchD},
				Opaque:    preparedImage{positions: positions, geom: geom},
			}},
		}
		media, err := m.EncodeMedia(prepared)
		if err != nil {
			t.Errorf("EncodeMedia: %v", err)
			return
		}
		if len(media) != 1 {
			t.Errorf("EncodeMedia returned %d items, want 1", len(media))
			return
		}
		if media[0].Pos != mediaPos || media[0].Seq != 0 {
			t.Errorf("media item pos/seq = (%d,%d), want (%d,0)", media[0].Pos, media[0].Seq, mediaPos)
		}
		// The tower code path must actually have run: the feature array is
		// [NumSoftTokens, text hidden] with finite values.
		feats := media[0].Features
		if feats == nil {
			t.Error("EncodeMedia returned nil features; tower did not run")
			return
		}
		mlx.Eval(feats)
		if got := feats.Dims(); len(got) != 2 || got[0] != patches || got[1] != textHidden {
			t.Errorf("feature dims %v, want [%d %d]", got, patches, textHidden)
			return
		}
		for i, v := range feats.AsType(mlx.DTypeFloat32).Floats() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("non-finite tower feature at %d: %v", i, v)
				return
			}
		}

		newBatch := func(withMedia bool) *batch.Batch {
			b := &batch.Batch{
				InputIDs:     mlx.FromValues(toks, 1, len(toks)),
				SeqOffsets:   []int32{0},
				SeqQueryLens: []int32{int32(len(toks))},
			}
			if withMedia {
				b.Media = media
			}
			return b
		}

		plain, _ := m.Forward(newBatch(false), nil)
		withMedia, _ := m.Forward(newBatch(true), nil)
		plain = plain.AsType(mlx.DTypeFloat32)
		withMedia = withMedia.AsType(mlx.DTypeFloat32)
		mlx.Eval(plain, withMedia)

		pDims, mDims := plain.Dims(), withMedia.Dims()
		if len(pDims) != 3 || pDims[1] != len(toks) || pDims[2] != int(cfg.TextConfig.EmbeddingDim) {
			t.Errorf("plain output dims %v, want [1 %d %d]", pDims, len(toks), cfg.TextConfig.EmbeddingDim)
			return
		}
		if len(mDims) != 3 || mDims[1] != len(toks) || mDims[2] != int(cfg.TextConfig.EmbeddingDim) {
			t.Errorf("media output dims %v, want [1 %d %d]", mDims, len(toks), cfg.TextConfig.EmbeddingDim)
			return
		}

		pv, mv := plain.Floats(), withMedia.Floats()
		if len(pv) != len(mv) {
			t.Errorf("output lengths %d vs %d", len(pv), len(mv))
			return
		}
		embDim := int(cfg.TextConfig.EmbeddingDim)
		rowChanged := make([]bool, len(toks))
		anyChange := false
		for row := range toks {
			for d := range embDim {
				i := row*embDim + d
				if math.Abs(float64(pv[i]-mv[i])) > 1e-5 {
					rowChanged[row] = true
					anyChange = true
					break
				}
			}
		}
		// (a) The media run differs: the scatter path had an effect.
		if !anyChange {
			t.Error("forward output identical with and without media; scatter is a no-op")
		}
		// (b) The replaced rows themselves changed (the tower features
		// landed at Pos..Pos+NumSoftTokens).
		for row := mediaPos; row < mediaPos+patches; row++ {
			if !rowChanged[row] {
				t.Errorf("media row %d unchanged; scatter missed it", row)
			}
		}
		// Bidirectional attention means text rows see the media too: the
		// leading text row must change, proving the spliced features
		// propagated through the encoder rather than sitting in dead rows.
		if !rowChanged[0] {
			t.Error("text row 0 unchanged; spliced media did not propagate through attention")
		}
	})
}

// TestVisionTowerLoadSeedsPositionCapacity: LoadVisionTower must seed the
// config's position-table capacity (the [2, positions, hidden] table's
// positions axis), or ProcessImage rejects every image with "patch grid
// exceeds vision position embedding size 0" — the failure that shipped the
// dict-form media path broken.
func TestVisionTowerLoadSeedsPositionCapacity(t *testing.T) {
	mlxtest.RunSubtest(t, "seed", func(t *mlxtest.T) {
		cfg := tinyTextConfig()
		tensors := tinyTextTensors(cfg)
		vcfg := tinyVisionTensors(tensors, int(cfg.TextConfig.EmbeddingDim))
		cfg.VisionConfig = vcfg

		linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
		if _, _, err := gemma4.LoadVisionTower(tensors, vcfg, linears); err != nil {
			t.Fatalf("LoadVisionTower: %v", err)
		}

		// A tiny PNG must clear the grid check. Budget 70 is the smallest
		// the reference processor supports; patch 2/pool 1 keeps the grid
		// inside the 8-position table for small inputs.
		var buf bytes.Buffer
		if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 8, 8))); err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := gemma4.ProcessImage(buf.Bytes(), vcfg, 70); err != nil {
			t.Errorf("ProcessImage after load: %v", err)
		}
	})
}
