package qwen3_5

import (
	"fmt"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/tokenizer"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

func TestParseConfigNestedDefaults(t *testing.T) {
	data := []byte(`{
		"model_type": "Qwen3_5MoeForConditionalGeneration",
		"text_config": {
			"hidden_size": 4096,
			"intermediate_size": 14336,
			"num_hidden_layers": 8,
			"num_attention_heads": 32,
			"num_key_value_heads": 8,
			"head_dim": 128,
			"linear_num_value_heads": 64,
			"linear_num_key_heads": 16,
			"linear_key_head_dim": 128,
			"linear_value_head_dim": 128,
			"linear_conv_kernel_dim": 4,
			"num_experts": 16,
			"num_experts_per_tok": 4,
			"moe_intermediate_size": 2048,
			"shared_expert_intermediate_size": 4096,
			"rope_parameters": {
				"rope_theta": 500000,
				"partial_rotary_factor": 0.5
			}
		}
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.RopeTheta != 500000 {
		t.Fatalf("rope theta mismatch: got %v", cfg.RopeTheta)
	}
	if cfg.RopeDim != 64 {
		t.Fatalf("rope dim mismatch: got %d want 64", cfg.RopeDim)
	}
	if cfg.FullAttentionInterval != 4 {
		t.Fatalf("full_attention_interval default mismatch: got %d want 4", cfg.FullAttentionInterval)
	}
	if !cfg.NormTopKProb {
		t.Fatalf("norm_topk_prob should default to true for MoE")
	}
}

func TestLayerSelectionHelpers(t *testing.T) {
	cfg := &Config{TextConfig: TextConfig{
		NumHiddenLayers:       6,
		FullAttentionInterval: 3,
		NumExperts:            8,
		DecoderSparseStep:     2,
		MLPOnlyLayers:         []int32{1},
	}}

	if !layerIsLinear(cfg, 0) {
		t.Fatalf("layer 0 should be linear")
	}
	if layerIsLinear(cfg, 2) {
		t.Fatalf("layer 2 should be full attention")
	}

	if layerUsesMoE(cfg, 1) {
		t.Fatalf("layer 1 should be forced dense by mlp_only_layers")
	}
	if !layerUsesMoE(cfg, 3) {
		t.Fatalf("layer 3 should use moe with decoder_sparse_step=2")
	}
}

func TestSupportsGatherQMM(t *testing.T) {
	tests := []struct {
		mode string
		bits int
		want bool
	}{
		{mode: "affine", bits: 4, want: true},
		{mode: "affine", bits: 8, want: true},
		{mode: "mxfp8", bits: 8, want: true},
		{mode: "nvfp4", bits: 4, want: true},
		{mode: "mxfp4", bits: 4, want: true},
		{mode: "mxfp8", bits: 4, want: false},
		{mode: "affine", bits: 3, want: false},
	}

	for _, tt := range tests {
		if got := supportsGatherQMM(tt.mode, tt.bits); got != tt.want {
			t.Fatalf("supportsGatherQMM(%q, %d) = %v, want %v", tt.mode, tt.bits, got, tt.want)
		}
	}
}

func TestResolveTensorPathLayout(t *testing.T) {
	dummy := mlx.New("dummy")

	tests := []struct {
		name          string
		key           string
		wantContainer string
		wantModel     string
	}{
		{
			name:          "standard",
			key:           "model.embed_tokens.weight",
			wantContainer: "",
			wantModel:     "model.",
		},
		{
			name:          "nested language model with inner model",
			key:           "model.language_model.model.embed_tokens.weight",
			wantContainer: "model.language_model.",
			wantModel:     "model.",
		},
		{
			name:          "nested language model without inner model",
			key:           "model.language_model.embed_tokens.weight",
			wantContainer: "model.language_model.",
			wantModel:     "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layout := resolveTensorPathLayout(map[string]*mlx.Array{
				tt.key: dummy,
			})

			if layout.containerPrefix != tt.wantContainer || layout.modelPrefix != tt.wantModel {
				t.Fatalf(
					"resolveTensorPathLayout() = {%q %q}, want {%q %q}",
					layout.containerPrefix,
					layout.modelPrefix,
					tt.wantContainer,
					tt.wantModel,
				)
			}
		})
	}
}

func TestNewCachesLayout(t *testing.T) {
	m := &Model{
		Config: &Config{TextConfig: TextConfig{
			LinearConvKernelDim: 4,
			LinearNumKeyHeads:   2,
			LinearKeyHeadDim:    8,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  16,
		}},
		Layers: []*Layer{
			{IsLinear: true},
			{IsLinear: false},
			{IsLinear: true},
		},
	}

	caches := m.NewCaches()
	if len(caches) != len(m.Layers) {
		t.Fatalf("len(caches) = %d, want %d", len(caches), len(m.Layers))
	}

	if _, ok := caches[0].(*cache.RecurrentCache); !ok {
		t.Fatalf("cache[0] = %T, want *cache.RecurrentCache", caches[0])
	}
	if _, ok := caches[1].(*cache.KVCache); !ok {
		t.Fatalf("cache[1] = %T, want *cache.KVCache", caches[1])
	}
	if _, ok := caches[2].(*cache.RecurrentCache); !ok {
		t.Fatalf("cache[2] = %T, want *cache.RecurrentCache", caches[2])
	}
}

func TestLoadWeightsPreservesLinearAttentionNormWeightDType(t *testing.T) {
	skipIfNoMLX(t)

	cfg := &Config{TextConfig: TextConfig{
		HiddenSize:            4,
		IntermediateSize:      8,
		NumHiddenLayers:       2,
		NumAttentionHeads:     1,
		NumKeyValueHeads:      1,
		HeadDim:               4,
		RMSNormEps:            1e-6,
		TieWordEmbeddings:     true,
		LayerTypes:            []string{"linear", "full"},
		LinearNumValueHeads:   1,
		LinearNumKeyHeads:     1,
		LinearKeyHeadDim:      2,
		LinearValueHeadDim:    2,
		LinearConvKernelDim:   4,
		FullAttentionInterval: 2,
	}}

	m := &Model{
		Config: cfg,
		Layers: make([]*Layer, cfg.NumHiddenLayers),
	}

	bf16 := mlx.DTypeBFloat16
	f32 := mlx.DTypeFloat32
	tensors := map[string]*mlx.Array{
		"model.embed_tokens.weight":                      mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4).AsType(bf16),
		"model.norm.weight":                              mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.0.input_layernorm.weight":          mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.0.post_attention_layernorm.weight": mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.0.linear_attn.in_proj_qkv.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
		}, 6, 4),
		"model.layers.0.linear_attn.in_proj_z.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		}, 2, 4),
		"model.layers.0.linear_attn.in_proj_b.weight": mlx.FromValues([]float32{1, 0, 0, 0}, 1, 4),
		"model.layers.0.linear_attn.in_proj_a.weight": mlx.FromValues([]float32{0, 1, 0, 0}, 1, 4),
		"model.layers.0.linear_attn.out_proj.weight": mlx.FromValues([]float32{
			1, 0,
			0, 1,
			1, 1,
			0, 0,
		}, 4, 2),
		"model.layers.0.linear_attn.conv1d.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
		}, 6, 4),
		"model.layers.0.linear_attn.norm.weight": mlx.FromValues([]float32{1, 1}, 2),
		"model.layers.0.linear_attn.dt_bias":     mlx.FromValues([]float32{0}, 1),
		"model.layers.0.linear_attn.A_log":       mlx.FromValues([]float32{0}, 1),
		"model.layers.0.mlp.gate_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.0.mlp.up_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.0.mlp.down_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0,
		}, 4, 8),
		"model.layers.1.input_layernorm.weight":          mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.1.post_attention_layernorm.weight": mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.1.self_attn.q_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.1.self_attn.k_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}, 4, 4),
		"model.layers.1.self_attn.v_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}, 4, 4),
		"model.layers.1.self_attn.o_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}, 4, 4),
		"model.layers.1.self_attn.q_norm.weight": mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.1.self_attn.k_norm.weight": mlx.FromValues([]float32{1, 1, 1, 1}, 4),
		"model.layers.1.mlp.gate_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.1.mlp.up_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 1, 0, 0,
			0, 1, 1, 0,
			0, 0, 1, 1,
			1, 0, 0, 1,
		}, 8, 4),
		"model.layers.1.mlp.down_proj.weight": mlx.FromValues([]float32{
			1, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0,
		}, 4, 8),
	}

	if err := m.LoadWeights(tensors); err != nil {
		t.Fatalf("LoadWeights failed: %v", err)
	}

	if got := m.Layers[0].InputNorm.Weight.DType(); got != f32 {
		t.Fatalf("layer 0 input norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[0].PostAttentionNorm.Weight.DType(); got != f32 {
		t.Fatalf("layer 0 post-attn norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[1].InputNorm.Weight.DType(); got != f32 {
		t.Fatalf("layer 1 input norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[1].PostAttentionNorm.Weight.DType(); got != f32 {
		t.Fatalf("layer 1 post-attn norm dtype = %v, want %v", got, f32)
	}

	if got := m.Norm.Weight.DType(); got != f32 {
		t.Fatalf("final norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[0].Linear.NormWeight.DType(); got != f32 {
		t.Fatalf("linear-attn norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[1].FullAttn.QNorm.Weight.DType(); got != f32 {
		t.Fatalf("q norm dtype = %v, want %v", got, f32)
	}
	if got := m.Layers[1].FullAttn.KNorm.Weight.DType(); got != f32 {
		t.Fatalf("k norm dtype = %v, want %v", got, f32)
	}
}

func TestParseConfigVisionFields(t *testing.T) {
	data := []byte(`{
		"text_config": {
			"hidden_size": 4096,
			"intermediate_size": 14336,
			"num_hidden_layers": 4,
			"num_attention_heads": 32,
			"num_key_value_heads": 8,
			"head_dim": 128,
			"linear_num_value_heads": 64,
			"linear_num_key_heads": 16,
			"linear_key_head_dim": 128,
			"linear_value_head_dim": 128,
			"linear_conv_kernel_dim": 4,
			"rope_parameters": {
				"rope_theta": 10000000
			}
		},
		"vision_config": {
			"depth": 2,
			"hidden_size": 256,
			"num_heads": 8,
			"in_channels": 3,
			"patch_size": 14,
			"spatial_merge_size": 2,
			"layer_norm_epsilon": 0.000001,
			"temporal_patch_size": 2,
			"num_position_embeddings": 2304
		},
		"image_token_id": 111,
		"vision_start_token_id": 112,
		"vision_end_token_id": 113
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.Vision == nil {
		t.Fatal("vision config should be parsed")
	}
	if cfg.Vision.Depth != 2 {
		t.Fatalf("vision.depth mismatch: got %d", cfg.Vision.Depth)
	}
	if cfg.Vision.GridPerSide != 48 {
		t.Fatalf("vision grid-per-side mismatch: got %d want 48", cfg.Vision.GridPerSide)
	}
	if cfg.Vision.RopeTheta != 10000 {
		t.Fatalf("vision rope_theta should default to 10000, got %v", cfg.Vision.RopeTheta)
	}
	if cfg.RopeTheta != 10000000 {
		t.Fatalf("text rope_theta mismatch: got %v", cfg.RopeTheta)
	}
	if cfg.ImageTokenID != 111 || cfg.VisionStartToken != 112 || cfg.VisionEndToken != 113 {
		t.Fatalf("vision token ids mismatch: got image=%d start=%d end=%d", cfg.ImageTokenID, cfg.VisionStartToken, cfg.VisionEndToken)
	}
}

func TestParseConfigMRoPEFromRopeParameters(t *testing.T) {
	data := []byte(`{
		"text_config": {
			"hidden_size": 2048,
			"intermediate_size": 8192,
			"num_hidden_layers": 4,
			"num_attention_heads": 16,
			"num_key_value_heads": 2,
			"head_dim": 256,
			"linear_num_value_heads": 32,
			"linear_num_key_heads": 16,
			"linear_key_head_dim": 128,
			"linear_value_head_dim": 128,
			"linear_conv_kernel_dim": 4,
			"rope_parameters": {
				"rope_theta": 10000000,
				"partial_rotary_factor": 0.25,
				"mrope_interleaved": true,
				"mrope_section": [11, 11, 10]
			}
		}
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if !cfg.MRoPEInterleaved {
		t.Fatal("mrope_interleaved should be parsed from rope_parameters")
	}
	if !slices.Equal(cfg.MRoPESections, []int32{11, 11, 10}) {
		t.Fatalf("mrope sections mismatch: got %v", cfg.MRoPESections)
	}
	if cfg.RopeDim != 64 {
		t.Fatalf("rope dim mismatch: got %d want 64", cfg.RopeDim)
	}
}

func TestParseConfigVisionTokenDefaults(t *testing.T) {
	data := []byte(`{
		"text_config": {
			"hidden_size": 4096,
			"intermediate_size": 14336,
			"num_hidden_layers": 2,
			"num_attention_heads": 32,
			"num_key_value_heads": 8,
			"head_dim": 128,
			"linear_num_value_heads": 64,
			"linear_num_key_heads": 16,
			"linear_key_head_dim": 128,
			"linear_value_head_dim": 128,
			"linear_conv_kernel_dim": 4
		}
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.ImageTokenID != 151655 {
		t.Fatalf("default image token mismatch: got %d", cfg.ImageTokenID)
	}
	if cfg.VisionStartToken != 151652 {
		t.Fatalf("default vision start token mismatch: got %d", cfg.VisionStartToken)
	}
	if cfg.VisionEndToken != 151653 {
		t.Fatalf("default vision end token mismatch: got %d", cfg.VisionEndToken)
	}
}

func TestResolveVisionPrefix(t *testing.T) {
	tests := []struct {
		name    string
		tensors map[string]*mlx.Array
		want    string
	}{
		{
			name: "legacy visual prefix",
			tensors: map[string]*mlx.Array{
				"model.visual.patch_embed.proj.weight": mlx.New("patch"),
			},
			want: "model.visual",
		},
		{
			name: "imported vision tower prefix",
			tensors: map[string]*mlx.Array{
				"vision_tower.blocks.0.attn.qkv.weight": mlx.New("qkv"),
			},
			want: "vision_tower",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveVisionPrefix(tt.tensors, "language_model."); got != tt.want {
				t.Fatalf("resolveVisionPrefix() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestVisionPreprocessorOverridesDefaults(t *testing.T) {
	v := &VisionConfig{}
	v.applyDefaults()

	v.applyPreprocessorConfig([]byte(`{
		"patch_size": 16,
		"temporal_patch_size": 3,
		"merge_size": 4,
		"size": {
			"shortest_edge": 1024,
			"longest_edge": 8192
		},
		"image_mean": [0.1, 0.2, 0.3],
		"image_std": [0.9, 0.8, 0.7]
	}`))

	if v.PatchSize != 16 {
		t.Fatalf("patch_size mismatch: got %d want 16", v.PatchSize)
	}
	if v.TemporalPatchSize != 3 {
		t.Fatalf("temporal_patch_size mismatch: got %d want 3", v.TemporalPatchSize)
	}
	if v.SpatialMergeSize != 4 {
		t.Fatalf("merge_size mismatch: got %d want 4", v.SpatialMergeSize)
	}
	if v.Size.ShortestEdge != 1024 || v.Size.LongestEdge != 8192 {
		t.Fatalf("size mismatch: got shortest=%d longest=%d", v.Size.ShortestEdge, v.Size.LongestEdge)
	}
	if v.ImageMean[0] != 0.1 || v.ImageStd[2] != 0.7 {
		t.Fatalf("image preprocessing stats mismatch: mean=%v std=%v", v.ImageMean, v.ImageStd)
	}
}

func TestVisionImageProcessorUsesPreprocessorSize(t *testing.T) {
	v := &VisionConfig{}
	v.applyDefaults()

	v.applyPreprocessorConfig([]byte(`{
		"size": {
			"shortest_edge": 65536,
			"longest_edge": 16777216
		},
		"patch_size": 16,
		"temporal_patch_size": 2,
		"merge_size": 2,
		"image_mean": [0.5, 0.5, 0.5],
		"image_std": [0.5, 0.5, 0.5]
	}`))

	p := newVisionImageProcessor(v)
	if p == nil {
		t.Fatal("newVisionImageProcessor returned nil")
	}

	if p.shortestEdge != 65536 || p.longestEdge != 16777216 {
		t.Fatalf("processor size mismatch: shortest=%d longest=%d", p.shortestEdge, p.longestEdge)
	}
}

func testTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()

	tok, err := tokenizer.LoadFromBytes([]byte(`{
		"model": {
			"type": "BPE",
			"vocab": {"a": 0},
			"merges": []
		}
	}`))
	if err != nil {
		t.Fatalf("failed to load test tokenizer: %v", err)
	}

	return tok
}

func TestTokenizePromptWithResolvedImagesStoresVisionSpans(t *testing.T) {
	skipIfNoMLX(t)

	m := &Model{
		tok: testTokenizer(t),
		Config: &Config{
			ImageTokenID:     101,
			VisionStartToken: 102,
			VisionEndToken:   103,
			Vision:           &VisionConfig{SpatialMergeSize: 2},
		},
		Vision:         &VisionModel{},
		ImageProcessor: &VisionImageProcessor{},
	}

	main := mlx.FromValues([]float32{
		10, 11,
		20, 21,
	}, 1, 2, 2)

	resolveCalls := 0
	got, state, err := m.tokenizePromptWithResolvedImages(
		"a[img-7][img-7]a",
		[]base.ImageInput{{ID: 7, Data: []byte("img7")}},
		func(data []byte) (*VisionEmbeddings, error) {
			if string(data) != "img7" {
				return nil, fmt.Errorf("unexpected data: %q", string(data))
			}
			resolveCalls++
			return &VisionEmbeddings{
				Main: main,
				Grid: &VisionGrid{Height: 2, Width: 2, Temporal: 1},
			}, nil
		},
	)
	if err != nil {
		t.Fatalf("tokenizePromptWithResolvedImages returned error: %v", err)
	}
	if resolveCalls != 1 {
		t.Fatalf("resolve calls mismatch: got %d want 1", resolveCalls)
	}

	want := []int32{
		0,
		102, 101, 101, 103,
		102, 101, 101, 103,
		0,
	}
	if !slices.Equal(got, want) {
		t.Fatalf("expanded tokens mismatch: got %v want %v", got, want)
	}

	if state == nil {
		t.Fatal("expected prompt vision state")
	}
	if len(state.Spans) != 2 {
		t.Fatalf("prompt span count mismatch: got %d want 2", len(state.Spans))
	}
	if state.Spans[0].Start != 2 || state.Spans[0].End != 4 {
		t.Fatalf("first span mismatch: got [%d,%d)", state.Spans[0].Start, state.Spans[0].End)
	}
	if state.Spans[1].Start != 6 || state.Spans[1].End != 8 {
		t.Fatalf("second span mismatch: got [%d,%d)", state.Spans[1].Start, state.Spans[1].End)
	}
	wantPos := []int32{0, 1, 2, 2, 3, 4, 5, 5, 6, 7}
	if !slices.Equal(state.PositionCache, wantPos) {
		t.Fatalf("position cache mismatch: got %v want %v", state.PositionCache, wantPos)
	}
}

func TestBuildPromptMRoPEPositions(t *testing.T) {
	m := &Model{
		Config: &Config{
			Vision: &VisionConfig{SpatialMergeSize: 2},
		},
	}
	state := &promptVisionState{
		PositionCache: []int32{0, 1, 2, 2, 2, 2, 2, 2, 5, 6},
		Spans: []promptVisionSpan{
			{
				Start: 2,
				End:   8,
				Grid:  &VisionGrid{Height: 4, Width: 6, Temporal: 1},
			},
		},
	}

	pos := m.buildPromptMRoPEPositions(state, 0, 10)
	if got, want := pos[0], []int32{0, 1, 2, 2, 2, 2, 2, 2, 5, 6}; !slices.Equal(got, want) {
		t.Fatalf("time positions mismatch: got %v want %v", got, want)
	}
	if got, want := pos[1], []int32{0, 1, 2, 2, 2, 3, 3, 3, 5, 6}; !slices.Equal(got, want) {
		t.Fatalf("height positions mismatch: got %v want %v", got, want)
	}
	if got, want := pos[2], []int32{0, 1, 2, 3, 4, 2, 3, 4, 5, 6}; !slices.Equal(got, want) {
		t.Fatalf("width positions mismatch: got %v want %v", got, want)
	}
}

func TestMapPromptPositionContinuesAfterCache(t *testing.T) {
	state := &promptVisionState{PositionCache: []int32{0, 1, 2, 2, 3}}

	if got := mapPromptPosition(state, 3); got != 2 {
		t.Fatalf("mapPromptPosition(3) = %d, want 2", got)
	}
	if got := mapPromptPosition(state, 5); got != 4 {
		t.Fatalf("mapPromptPosition(5) = %d, want 4", got)
	}
}

func TestApplyPromptVisionEmbeddings(t *testing.T) {
	skipIfNoMLX(t)

	m := &Model{}
	state := &promptVisionState{
		Spans: []promptVisionSpan{
			{
				Start: 1,
				End:   3,
				Main: mlx.FromValues([]float32{
					10, 11,
					20, 21,
				}, 1, 2, 2),
			},
		},
	}

	h := mlx.FromValues([]float32{
		0, 1,
		2, 3,
		4, 5,
		6, 7,
	}, 1, 4, 2)

	got := m.applyPromptVisionEmbeddings(h, 0, state)
	mlx.Eval(got)

	want := []float32{
		0, 1,
		10, 11,
		20, 21,
		6, 7,
	}
	if !slices.Equal(got.Floats(), want) {
		t.Fatalf("embedding replacement mismatch: got %v want %v", got.Floats(), want)
	}
}
