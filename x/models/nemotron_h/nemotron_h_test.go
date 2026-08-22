package nemotron_h

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
)

func TestParseConfigNestedWrapper(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"architectures": ["NemotronH_Nano_VL_V2"],
		"model_type": "NemotronH_Nano_VL_V2",
		"llm_config": {
			"model_type": "nemotron_h",
			"hidden_size": 2688,
			"num_hidden_layers": 4,
			"hybrid_override_pattern": "M*E-",
			"num_attention_heads": 32,
			"num_key_value_heads": 2,
			"head_dim": 84,
			"mamba_num_heads": 64,
			"mamba_head_dim": 64,
			"conv_kernel": 4,
			"ssm_state_size": 128,
			"n_groups": 8,
			"n_routed_experts": 128,
			"num_experts_per_tok": 6,
			"routed_scaling_factor": 2.5,
			"layer_norm_epsilon": 0.00001
		}
	}`))
	if err != nil {
		t.Fatalf("parseConfig returned error: %v", err)
	}

	if got, want := cfg.ModelType, "nemotron_h"; got != want {
		t.Fatalf("ModelType = %q, want %q", got, want)
	}
	if got, want := string(cfg.LayerTypes), "M*E-"; got != want {
		t.Fatalf("LayerTypes = %q, want %q", got, want)
	}
	if got, want := cfg.NGroups, int32(8); got != want {
		t.Fatalf("NGroups = %d, want %d", got, want)
	}
	if got, want := cfg.RoutedScalingFactor, float32(2.5); got != want {
		t.Fatalf("RoutedScalingFactor = %v, want %v", got, want)
	}
}

func TestParseConfigRejectsBadPattern(t *testing.T) {
	_, err := parseConfig([]byte(`{
		"hidden_size": 4,
		"num_hidden_layers": 2,
		"hybrid_override_pattern": "M",
		"num_attention_heads": 2
	}`))
	if err == nil || !strings.Contains(err.Error(), "hybrid_override_pattern length") {
		t.Fatalf("parseConfig error = %v, want hybrid pattern length error", err)
	}
}

func TestParseConfigRejectsUndividedExpertGroups(t *testing.T) {
	_, err := parseConfig([]byte(`{
		"hidden_size": 4,
		"num_hidden_layers": 1,
		"hybrid_override_pattern": "E",
		"num_attention_heads": 2,
		"n_routed_experts": 10,
		"n_group": 3,
		"num_experts_per_tok": 2
	}`))
	if err == nil || !strings.Contains(err.Error(), "must be divisible by n_group") {
		t.Fatalf("parseConfig error = %v, want expert-group divisibility error", err)
	}
}

func TestNewCachesLayout(t *testing.T) {
	m := &Model{
		Config: &Config{
			ConvKernel:    4,
			MambaNumHeads: 2,
			MambaHeadDim:  2,
			SSMStateSize:  3,
			NGroups:       1,
		},
		Layers: []*Layer{
			{Type: 'M'},
			{Type: '*'},
			{Type: 'E'},
			{Type: '-'},
		},
	}

	caches := m.NewCaches()
	if got, want := len(caches), 4; got != want {
		t.Fatalf("len(NewCaches()) = %d, want %d", got, want)
	}
	if _, ok := caches[0].(*cache.RecurrentCache); !ok {
		t.Fatalf("caches[0] = %T, want *cache.RecurrentCache", caches[0])
	}
	if _, ok := caches[1].(*cache.KVCache); !ok {
		t.Fatalf("caches[1] = %T, want *cache.KVCache", caches[1])
	}
	if caches[2] != nil || caches[3] != nil {
		t.Fatalf("MLP-only caches = %T/%T, want nil/nil", caches[2], caches[3])
	}
}

func TestNewCachesLayoutWithAttentionMTP(t *testing.T) {
	m := &Model{
		Config: &Config{
			ConvKernel:    4,
			MambaNumHeads: 2,
			MambaHeadDim:  2,
			SSMStateSize:  3,
			NGroups:       1,
		},
		Layers: []*Layer{
			{Type: 'M'},
			{Type: 'A'},
			{Type: 'E'},
			{Type: '-'},
		},
		MTP: &MTPHead{Layers: []*Layer{{Type: 'A'}, {Type: 'E'}}},
	}

	// The model declares only its own layers; the draft declares its own.
	caches := m.NewCaches()
	if got, want := len(caches), 4; got != want {
		t.Fatalf("len(NewCaches()) = %d, want %d", got, want)
	}

	draft := m.SelfDraft()
	if draft == nil {
		t.Fatal("SelfDraft() = nil, want the MTP draft")
	}
	draftCaches := draft.NewCaches()
	if got, want := len(draftCaches), 1; got != want {
		t.Fatalf("len(draft.NewCaches()) = %d, want %d", got, want)
	}
	if _, ok := draftCaches[0].(*cache.KVCache); !ok {
		t.Fatalf("draft.NewCaches()[0] = %T, want *cache.KVCache", draftCaches[0])
	}
}

func TestNewCachesLayoutWithCachelessMTP(t *testing.T) {
	m := &Model{
		Config: &Config{
			ConvKernel:    4,
			MambaNumHeads: 2,
			MambaHeadDim:  2,
			SSMStateSize:  3,
			NGroups:       1,
		},
		Layers: []*Layer{
			{Type: 'M'},
			{Type: 'A'},
			{Type: 'E'},
			{Type: '-'},
		},
		MTP: &MTPHead{Layers: []*Layer{{Type: 'E'}}},
	}

	caches := m.NewCaches()
	if got, want := len(caches), 4; got != want {
		t.Fatalf("len(NewCaches()) = %d, want %d", got, want)
	}

	draft := m.SelfDraft()
	if draft == nil {
		t.Fatal("SelfDraft() = nil, want the MTP draft")
	}
	if draftCaches := draft.NewCaches(); len(draftCaches) != 0 {
		t.Fatalf("draft.NewCaches() = %v, want none for a cacheless MTP", draftCaches)
	}
}

func TestDetectMTPLayerType(t *testing.T) {
	present := &mlx.Array{}
	for _, tt := range []struct {
		name    string
		tensors map[string]*mlx.Array
		want    byte
		wantErr string
	}{
		{
			name:    "attention",
			tensors: map[string]*mlx.Array{"mtp.layers.0.mixer.q_proj.weight": present},
			want:    'A',
		},
		{
			name:    "moe",
			tensors: map[string]*mlx.Array{"mtp.layers.0.mixer.experts.0.up_proj.weight": present},
			want:    'E',
		},
		{
			name:    "dense",
			tensors: map[string]*mlx.Array{"mtp.layers.0.mixer.up_proj.weight": present},
			want:    '-',
		},
		{
			name:    "recurrent unsupported",
			tensors: map[string]*mlx.Array{"mtp.layers.0.mixer.in_proj.weight": present},
			wantErr: "recurrent MTP layers are not supported",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := detectMTPLayerType(tt.tensors, "mtp.layers.0")
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("detectMTPLayerType error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("detectMTPLayerType returned error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("detectMTPLayerType = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMTPLayerPrefixes(t *testing.T) {
	present := &mlx.Array{}
	for _, tt := range []struct {
		name    string
		tensors map[string]*mlx.Array
		want    []string
		wantErr string
	}{
		{
			name:    "nonzero layer index",
			tensors: map[string]*mlx.Array{"mtp.layers.1.mixer.experts.0.up_proj.weight": present},
			want:    []string{"mtp.layers.1"},
		},
		{
			name: "multiple layers",
			tensors: map[string]*mlx.Array{
				"mtp.layers.0.mixer.q_proj.weight":            present,
				"mtp.layers.1.mixer.experts.0.up_proj.weight": present,
			},
			want: []string{"mtp.layers.0", "mtp.layers.1"},
		},
		{
			name:    "missing layer",
			tensors: map[string]*mlx.Array{"mtp.fc.weight": present},
			wantErr: "missing MTP layer tensors",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := mtpLayerPrefixes(tt.tensors)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("mtpLayerPrefixes error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("mtpLayerPrefixes returned error: %v", err)
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("mtpLayerPrefixes = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHasMTPHeadTensors(t *testing.T) {
	present := &mlx.Array{}
	for _, tt := range []struct {
		name    string
		tensors map[string]*mlx.Array
		want    bool
	}{
		{
			name:    "qwen style",
			tensors: map[string]*mlx.Array{"mtp.fc.weight": present},
			want:    true,
		},
		{
			name:    "nemotron native",
			tensors: map[string]*mlx.Array{"mtp.layers.0.eh_proj.weight": present},
			want:    true,
		},
		{
			name:    "mtp layer without fusion",
			tensors: map[string]*mlx.Array{"mtp.layers.0.mixer.q_proj.weight": present},
			want:    false,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := hasMTPHeadTensors(tt.tensors); got != tt.want {
				t.Fatalf("hasMTPHeadTensors() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSupportsGatherQMM(t *testing.T) {
	for _, tt := range []struct {
		mode string
		bits int
		want bool
	}{
		{mode: "affine", bits: 4, want: true},
		{mode: "mxfp4", bits: 4, want: true},
		{mode: "nvfp4", bits: 4, want: true},
		{mode: "mxfp8", bits: 8, want: true},
		{mode: "mxfp8", bits: 4, want: false},
		{mode: "unknown", bits: 4, want: false},
	} {
		if got := supportsGatherQMM(tt.mode, tt.bits); got != tt.want {
			t.Fatalf("supportsGatherQMM(%q, %d) = %v, want %v", tt.mode, tt.bits, got, tt.want)
		}
	}
}

func TestApplyExpertWeightGlobalScale(t *testing.T) {
	mlxtest.Setup(t)

	weight := mlx.FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}, 2, 2, 2)
	scale := mlx.FromValues([]float32{2, 3}, 2)

	got := applyExpertWeightGlobalScale(weight, scale)
	mlx.Eval(got)

	assertAllClose(t, "scaled expert weight", got.Floats(), []float32{
		2, 4,
		6, 8,
		15, 18,
		21, 24,
	}, 1e-5)
}

// The production form leans on MLX's fused fast RMS norm, so check it against
// a reference that spells out every step.
func TestGatedGroupRMSNormMatchesElementwiseReference(t *testing.T) {
	mlxtest.Setup(t)

	cfg := &Config{MambaNumHeads: 4, MambaHeadDim: 2, NGroups: 2, LayerNormEpsilon: 1e-5}
	inner := cfg.MambaNumHeads * cfg.MambaHeadDim
	groupSize := inner / cfg.NGroups
	B, L := int32(2), int32(3)

	y := testGatedValues(0.1, int(B), int(L), int(inner))
	gate := testGatedValues(-0.2, int(B), int(L), int(inner))
	weight := testGatedValues(0.9, int(inner))

	got := gatedGroupRMSNorm(y, gate, weight, cfg, mlx.DTypeFloat32)

	ref := mlx.Mul(y, mlx.SiLU(gate))
	ref = mlx.Reshape(ref, B, L, cfg.NGroups, groupSize)
	variance := mlx.Mean(mlx.Mul(ref, ref), 3, true)
	ref = mlx.Mul(ref, mlx.RSqrt(mlx.AddScalar(variance, cfg.LayerNormEpsilon)))
	ref = mlx.Mul(ref, mlx.Reshape(weight, 1, 1, cfg.NGroups, groupSize))
	ref = mlx.Reshape(ref, B, L, inner)

	mlx.Eval(got, ref)
	assertAllClose(t, "gated group rmsnorm", got.Floats(), ref.Floats(), 1e-5)
}

func testGatedValues(seed float32, shape ...int) *mlx.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = seed + 0.01*float32(i)
	}
	return mlx.FromValues(vals, shape...)
}

func assertAllClose(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func TestParseVisionConfigOmniDefaults(t *testing.T) {
	cfg, err := parseVisionConfig([]byte(`{
		"vision_config": {
			"version": "radio_v2.5-h",
			"patch_size": 16,
			"min_num_patches": 1024,
			"max_num_patches": 13312
		},
		"patch_size": 16,
		"downsample_ratio": 0.5,
		"img_context_token_id": 18,
		"img_context_token": "<image>",
		"img_start_token": "<img>",
		"img_end_token": "</img>",
		"vit_hidden_size": 1280,
		"projector_hidden_size": 20480
	}`), []byte(`{
		"patch_size": 16,
		"downsample_ratio": 0.5,
		"norm_mean": [0.48145466, 0.4578275, 0.40821073],
		"norm_std": [0.26862954, 0.26130258, 0.27577711]
	}`))
	if err != nil {
		t.Fatalf("parseVisionConfig returned error: %v", err)
	}
	if cfg == nil {
		t.Fatal("parseVisionConfig returned nil config")
	}
	if got, want := cfg.HiddenSize, int32(1280); got != want {
		t.Fatalf("HiddenSize = %d, want %d", got, want)
	}
	if got, want := cfg.NumHiddenLayers, int32(32); got != want {
		t.Fatalf("NumHiddenLayers = %d, want %d", got, want)
	}
	if got, want := cfg.HeadDim, int32(80); got != want {
		t.Fatalf("HeadDim = %d, want %d", got, want)
	}
	if got, want := cfg.DownsampleFactor, int32(2); got != want {
		t.Fatalf("DownsampleFactor = %d, want %d", got, want)
	}
	if got, want := cfg.ImageTokenID, int32(18); got != want {
		t.Fatalf("ImageTokenID = %d, want %d", got, want)
	}
	if got, want := cfg.MinNumPatches, 1024; got != want {
		t.Fatalf("MinNumPatches = %d, want %d", got, want)
	}
	if got, want := cfg.MaxNumPatches, 13312; got != want {
		t.Fatalf("MaxNumPatches = %d, want %d", got, want)
	}
	if got, want := cfg.MaxModelLen, 16384; got != want {
		t.Fatalf("MaxModelLen = %d, want %d", got, want)
	}
}

func TestParseVisionConfigRejectsMalformedNormalization(t *testing.T) {
	config := []byte(`{
		"vision_config": {"version": "c-radio_v4-h"},
		"norm_mean": [0.1, 0.2]
	}`)
	if _, err := parseVisionConfig(config, nil); err == nil {
		t.Fatal("expected malformed config norm_mean error")
	}

	config = []byte(`{"vision_config": {"version": "c-radio_v4-h"}}`)
	preprocessor := []byte(`{"norm_std": [0.1, 0.2, 0.3, 0.4]}`)
	if _, err := parseVisionConfig(config, preprocessor); err == nil {
		t.Fatal("expected malformed preprocessor norm_std error")
	}
}

func TestParseVisionConfigRequiresDynamicResolutionBounds(t *testing.T) {
	for _, config := range []string{
		`{"vision_config":{"version":"c-radio_v4-h","min_num_patches":1024}}`,
		`{"vision_config":{"version":"c-radio_v4-h","max_num_patches":13312}}`,
		`{"vision_config":{"version":"c-radio_v4-h"}}`,
	} {
		if _, err := parseVisionConfig([]byte(config), nil); err == nil || !strings.Contains(err.Error(), "requires min_num_patches and max_num_patches") {
			t.Fatalf("parseVisionConfig error = %v, want missing dynamic-resolution bounds", err)
		}
	}

	_, err := parseVisionConfig([]byte(`{
		"vision_config": {
			"version": "c-radio_v4-h",
			"min_num_patches": 2048,
			"max_num_patches": 1024
		}
	}`), nil)
	if err == nil || !strings.Contains(err.Error(), "min_num_patches (2048) exceeds max_num_patches (1024)") {
		t.Fatalf("parseVisionConfig error = %v, want reversed bounds error", err)
	}
}

func TestLoadVisionWeightsRejectsNonSquarePositionGrid(t *testing.T) {
	mlxtest.Setup(t)

	const prefix = "vision_model.radio_model.model."
	tensors := map[string]*mlx.Array{
		prefix + "patch_generator.embedder.weight": mlx.Zeros(mlx.DTypeBFloat16, 4, 3),
		prefix + "patch_generator.cls_token.token": mlx.Zeros(mlx.DTypeBFloat16, 1, 4),
		prefix + "patch_generator.pos_embed":       mlx.Zeros(mlx.DTypeBFloat16, 1, 6, 4),
	}
	m := &Model{
		VisionConfig:  &VisionConfig{},
		VisionEncoder: &RadioVisionEncoder{},
		Projector:     &VisionProjector{},
	}
	linears := model.NewLinearFactory(tensors, 0, 0, "", nil)
	if err := m.loadVisionWeights(tensors, linears); err == nil || !strings.Contains(err.Error(), "6 positions, want a square grid") {
		t.Fatalf("loadVisionWeights error = %v, want non-square position-grid error", err)
	}
}

func TestResizeGrid2DPreservesSourceDType(t *testing.T) {
	mlxtest.Setup(t)

	x := mlx.FromValues([]float32{0, 1, 2, 3}, 1, 2, 2, 1).AsType(mlx.DTypeBFloat16)
	got := resizeGrid2D(x, 3, 3)
	mlx.Eval(got)
	if got.DType() != mlx.DTypeBFloat16 {
		t.Fatalf("resizeGrid2D dtype = %s, want bfloat16", got.DType())
	}
}

func TestNemotronImagePatchBudget(t *testing.T) {
	cfg := &VisionConfig{
		DownsampleFactor: 2,
		MinNumPatches:    1024,
		MaxNumPatches:    13312,
		MaxModelLen:      16384,
	}

	// Context-bound, not memory-bound: base.MediaModel requires PrepareMedia
	// to be deterministic for given segments.
	if got, want := nemotronImagePatchBudget(cfg), 13312; got != want {
		t.Fatalf("budget = %d, want %d", got, want)
	}
	if got, want := nemotronImagePatchBudget(&VisionConfig{DownsampleFactor: 2, MinNumPatches: 1024, MaxNumPatches: 13312, MaxModelLen: 512}), 2032; got != want {
		t.Fatalf("context-limited budget = %d, want %d", got, want)
	}
}

func TestNemotronImagePatchGrid(t *testing.T) {
	cfg := &VisionConfig{
		PatchSize:        16,
		DownsampleFactor: 2,
		MinNumPatches:    1024,
		MaxNumPatches:    13312,
	}

	for _, tt := range []struct {
		name       string
		height     int
		width      int
		budget     int
		wantMax    int
		wantMin    int
		wantFactor int
	}{
		{name: "square minimum", height: 512, width: 512, budget: 1024, wantMax: 1024, wantMin: 1024, wantFactor: 2},
		{name: "large square capped", height: 2048, width: 2048, budget: 13312, wantMax: 13312, wantMin: 1024, wantFactor: 2},
		{name: "wide image capped", height: 512, width: 2048, budget: 4096, wantMax: 4096, wantMin: 1024, wantFactor: 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			patchH, patchW := nemotronImagePatchGrid(tt.height, tt.width, tt.budget, cfg)
			if got := patchH * patchW; got > tt.wantMax || got < tt.wantMin {
				t.Fatalf("patches = %d (%dx%d), want within [%d,%d]", got, patchH, patchW, tt.wantMin, tt.wantMax)
			}
			if patchH%tt.wantFactor != 0 || patchW%tt.wantFactor != 0 {
				t.Fatalf("patch grid = %dx%d, want both divisible by %d", patchH, patchW, tt.wantFactor)
			}
		})
	}
}
