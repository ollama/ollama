package glimmer

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func testConfig() Config {
	return Config{
		HiddenSize:               6656,
		NumHiddenLayers:          4,
		IntermediateSize:         19968,
		NumAttentionHeads:        32,
		NumKeyValueHeads:         2,
		HeadDim:                  128,
		VocabSize:                202048,
		RMSNormEps:               1e-5,
		PostNormEps:              1e-8,
		RopeTheta:                500000,
		MaxPositionEmbeddings:    16384,
		SlidingWindow:            2048,
		LayerTypes:               []string{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"},
		NoRopeLayers:             []int32{1, 1, 1, 0},
		NormalizeTokenEmbeddings: true,
		UseQKNorm:                true,
		QKScaleFactor:            43.7840518911,
		UseAttentionOutputGate:   true,
		OutputMultiplier:         0.19611613,
		OutputSoftCapTemp:        20,
	}
}

func marshalConfig(t *testing.T, cfg Config) []byte {
	t.Helper()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestParseConfig(t *testing.T) {
	cfg, err := parseConfig(marshalConfig(t, testConfig()))
	if err != nil {
		t.Fatal(err)
	}

	if got, want := cfg.AttentionScale, float32(1/math.Sqrt(128)); math.Abs(float64(got-want)) > 1e-7 {
		t.Errorf("AttentionScale = %g, want %g", got, want)
	}
	if got, want := cfg.QueryScale, float32(float64(cfg.QKScaleFactor)/math.Sqrt(128)); math.Abs(float64(got-want)) > 1e-7 {
		t.Errorf("QueryScale = %g, want %g", got, want)
	}
}

func TestParseOfficialHFConfig(t *testing.T) {
	data := []byte(`{
		"text_config": {
			"hidden_size": 6656,
			"num_hidden_layers": 4,
			"intermediate_size": 19968,
			"num_attention_heads": 32,
			"num_key_value_heads": 2,
			"head_dim": 128,
			"vocab_size": 202048,
			"rms_norm_eps": 0.00001,
			"post_norm_eps": 1e-8,
			"max_position_embeddings": 131072,
			"sliding_window": 2048,
			"layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
			"layer_rope_theta": [500000, 500000, 500000, 0],
			"qk_scale_factor": 3.87,
			"output_multiplier": 0.19611613513818404,
			"final_logit_softcapping": 20,
			"rope_parameters": {"rope_theta": 500000}
		},
		"vision_config": {
			"hidden_size": 1536,
			"intermediate_size": 8960,
			"num_attention_heads": 16,
			"num_hidden_layers": 4,
			"patch_size": 14,
			"patch_temporal": 2,
			"merge_size": 2,
			"pos_emb_height": 32,
			"pos_emb_width": 32,
			"layer_types": ["window_attention", "window_attention", "window_attention", "full_attention"]
		},
		"image_token_id": 200092,
		"out_hidden_size": 6144,
		"projector_hidden_size": 4096
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.QueryScale != 3.87 {
		t.Errorf("QueryScale = %g, want 3.87", cfg.QueryScale)
	}
	if cfg.MaxPositionEmbeddings != 131072 {
		t.Errorf("MaxPositionEmbeddings = %d, want 131072", cfg.MaxPositionEmbeddings)
	}
	if !slices.Equal(cfg.NoRopeLayers, []int32{1, 1, 1, 0}) {
		t.Errorf("NoRopeLayers = %v, want [1 1 1 0]", cfg.NoRopeLayers)
	}
	if cfg.NormalizeTokenEmbeddings {
		t.Error("NormalizeTokenEmbeddings = true, want false for pre-normalized HF embeddings")
	}
	if !cfg.NormalizeVisionEmbeddings {
		t.Error("NormalizeVisionEmbeddings = false, want true")
	}
	if cfg.VisionRoPEInterleaved {
		t.Error("VisionRoPEInterleaved = true, want half-rotation layout")
	}
	if cfg.VisionSparseAttentionFactor != 4 || cfg.VisionOutputDim != 6144 || cfg.VisionAdapterDim != 4096 {
		t.Errorf("vision config = factor %d, output %d, adapter %d", cfg.VisionSparseAttentionFactor, cfg.VisionOutputDim, cfg.VisionAdapterDim)
	}
}

func TestValidateTokenizerEOS(t *testing.T) {
	const tokenizerJSON = `{
		"model": {
			"type": "BPE",
			"vocab": {"x": 0},
			"merges": []
		},
		"added_tokens": [
			{"id": 1, "content": "<|start|>", "special": true},
			{"id": 2, "content": "<|message|>", "special": true},
			{"id": 3, "content": "<|eom|>", "special": true},
			{"id": 4, "content": "<|eot|>", "special": true},
			{"id": 5, "content": "<|end_of_text|>", "special": true}
		]
	}`

	tests := []struct {
		name string
		eos  string
		want string
	}{
		{name: "official boundaries", eos: `[5, 4]`},
		{name: "message boundary is terminal", eos: `[5, 3, 4]`, want: "<|eom|> must be a non-terminal"},
		{name: "end of turn is not terminal", eos: `[5]`, want: "<|eot|> must be an EOS token"},
		{name: "end of text is not terminal", eos: `[4]`, want: "<|end_of_text|> must be an EOS token"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tok, err := tokenizer.LoadFromBytesWithConfig([]byte(tokenizerJSON), &tokenizer.TokenizerConfig{
				GenerationConfigJSON: []byte(`{"eos_token_id":` + tt.eos + `}`),
			})
			if err != nil {
				t.Fatal(err)
			}

			err = validateTokenizer(tok)
			if tt.want == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validateTokenizer() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestParseConfigRejectsInvalidSchedules(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Config)
		want   string
	}{
		{"layer count", func(cfg *Config) { cfg.LayerTypes = cfg.LayerTypes[:3] }, "layer_types has 3 entries"},
		{"layer type", func(cfg *Config) { cfg.LayerTypes[0] = "linear_attention" }, `unsupported value "linear_attention"`},
		{"RoPE flag", func(cfg *Config) { cfg.NoRopeLayers[0] = 2 }, "no_rope_layers[0] has unsupported value 2"},
		{"sliding window", func(cfg *Config) { cfg.SlidingWindow = 0 }, "invalid sliding_window: 0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testConfig()
			tt.mutate(&cfg)
			_, err := parseConfig(marshalConfig(t, cfg))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("parseConfig() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestNewCachesMatchesAttentionSchedule(t *testing.T) {
	cfg := testConfig()
	m := Model{
		Config: &cfg,
		Layers: []*Layer{
			{IsSliding: true},
			{IsSliding: true},
			{IsSliding: true},
			{IsSliding: false},
		},
	}

	caches := m.NewCaches()
	for i := range 3 {
		if _, ok := caches[i].(*cache.RotatingKVCache); !ok {
			t.Errorf("cache %d = %T, want *cache.RotatingKVCache", i, caches[i])
		}
	}
	if _, ok := caches[3].(*cache.KVCache); !ok {
		t.Errorf("cache 3 = %T, want *cache.KVCache", caches[3])
	}
}

func TestUnembedReturnsFloat32Logits(t *testing.T) {
	mlxtest.Run(t, testUnembedReturnsFloat32Logits)
}

func testUnembedReturnsFloat32Logits(t *testing.T) {
	input := mlx.FromValues([]float32{1}, 1, 1, 1).AsType(mlx.DTypeBFloat16)
	weight := mlx.FromValues([]float32{1, 2}, 2, 1).AsType(mlx.DTypeBFloat16)
	m := Model{
		LMHead: nn.NewLinear(weight, nil),
		Config: &Config{
			OutputMultiplier:  0.19611613,
			OutputSoftCapTemp: 20,
		},
	}

	if got := m.Unembed(input).DType(); got != mlx.DTypeFloat32 {
		t.Fatalf("Unembed() dtype = %v, want %v", got, mlx.DTypeFloat32)
	}
}

func TestComputeImageSizeMatchesReference(t *testing.T) {
	tests := []struct {
		width, height             int
		targetWidth, targetHeight int
		tokens                    int
	}{
		{1, 1, 28, 28, 1},
		{100, 100, 112, 112, 16},
		{640, 480, 644, 476, 391},
		{480, 640, 476, 644, 391},
		{1920, 1080, 1932, 1092, 2691},
		{1080, 1920, 1092, 1932, 2691},
		{4000, 4000, 1792, 1792, 4096},
		{8192, 512, 7168, 448, 4096},
		{512, 8192, 448, 7168, 4096},
		{1234, 987, 1260, 1008, 1620},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("%dx%d", tt.width, tt.height), func(t *testing.T) {
			width, height, tokens := computeImageSize(tt.width, tt.height, 28, maxImageTokens)
			if width != tt.targetWidth || height != tt.targetHeight || tokens != tt.tokens {
				t.Fatalf("computeImageSize(%d, %d) = (%d, %d, %d), want (%d, %d, %d)",
					tt.width, tt.height, width, height, tokens,
					tt.targetWidth, tt.targetHeight, tt.tokens)
			}
		})
	}
}

func TestComputeImageSizeDeterministicTieBreak(t *testing.T) {
	width, height, tokens := computeImageSize(128, 128, 28, maxImageTokens)
	if width != 140 || height != 140 || tokens != 25 {
		t.Fatalf("computeImageSize() = %dx%d (%d tokens), want 140x140 (25 tokens)", width, height, tokens)
	}
}

func TestSparseVisionPermutation(t *testing.T) {
	permutation, lengths := sparseVisionPermutation(3, 5, 2, 3)
	wantPermutation := []int32{0, 1, 2, 5, 6, 7, 3, 4, 8, 9, 10, 11, 12, 13, 14}
	wantLengths := []int{6, 4, 3, 2}
	if !slices.Equal(permutation, wantPermutation) {
		t.Fatalf("permutation = %v, want %v", permutation, wantPermutation)
	}
	if !slices.Equal(lengths, wantLengths) {
		t.Fatalf("lengths = %v, want %v", lengths, wantLengths)
	}
}

func TestApplyVisionRoPELayouts(t *testing.T) {
	tests := []struct {
		name        string
		interleaved bool
		want        []float32
	}{
		{
			name: "official half rotation",
			want: []float32{
				1*0.5 - 3*0.75,
				2*0.25 - 4*0.125,
				3*0.5 + 1*0.75,
				4*0.25 + 2*0.125,
			},
		},
		{
			name:        "legacy interleaved rotation",
			interleaved: true,
			want: []float32{
				1*0.5 - 2*0.75,
				1*0.75 + 2*0.5,
				3*0.25 - 4*0.125,
				3*0.125 + 4*0.25,
			},
		},
	}

	for _, tt := range tests {
		mlxtest.RunSubtest(t, tt.name, func(t *testing.T) {
			x := mlx.FromValues([]float32{1, 2, 3, 4}, 1, 1, 1, 4)
			cos := mlx.FromValues([]float32{0.5, 0.25}, 1, 2)
			sin := mlx.FromValues([]float32{0.75, 0.125}, 1, 2)
			got := applyVisionRoPE(x, cos, sin, tt.interleaved)
			mlx.Eval(got)
			if !slices.Equal(got.Floats(), tt.want) {
				t.Fatalf("applyVisionRoPE() = %v, want %v", got.Floats(), tt.want)
			}
		})
	}
}
