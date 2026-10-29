package nemotron_h

import (
	"fmt"
	"image"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
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
	if cfg.MambaRoundIntermediates {
		t.Fatal("legacy architecture unexpectedly changed Mamba precision")
	}
}

func TestParseConfigLayersBlockType(t *testing.T) {
	for _, tt := range []struct {
		name       string
		layerTypes string
		want       string
	}{
		{"publisher names", `["mamba", "moe", "attention", "mlp"]`, "ME*-"},
		{"legacy aliases", `["linear_attention", "moe", "full_attention"]`, "ME*"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := parseConfig([]byte(fmt.Sprintf(`{
				"model_type": "nemotron_h",
				"hidden_size": 8,
				"num_attention_heads": 2,
				"layers_block_type": %s
			}`, tt.layerTypes)))
			if err != nil {
				t.Fatalf("parseConfig returned error: %v", err)
			}

			if got, want := cfg.NumHiddenLayers, int32(len(tt.want)); got != want {
				t.Fatalf("NumHiddenLayers = %d, want %d", got, want)
			}
			if got, want := cfg.HybridOverridePattern, tt.want; got != want {
				t.Fatalf("HybridOverridePattern = %q, want %q", got, want)
			}
			if got, want := string(cfg.LayerTypes), tt.want; got != want {
				t.Fatalf("LayerTypes = %q, want %q", got, want)
			}
		})
	}
}

func TestParseConfigRejectsUnknownLayerBlockType(t *testing.T) {
	_, err := parseConfig([]byte(`{
		"hidden_size": 8,
		"num_attention_heads": 2,
		"layers_block_type": ["linear_attention", "dense"]
	}`))
	if err == nil || !strings.Contains(err.Error(), `unsupported layers_block_type "dense" at layer 1`) {
		t.Fatalf("parseConfig error = %v, want unsupported layer type error", err)
	}
}

func TestParseSuperConfig(t *testing.T) {
	cfg, err := parseConfig([]byte(`{
		"architectures": ["NemotronH_Omni_Reasoning_V3"],
		"llm_config": {
			"hidden_size": 4096,
			"num_attention_heads": 32,
			"num_key_value_heads": 2,
			"moe_latent_size": 1024,
			"layers_block_type": ["mamba", "moe", "attention"]
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MoELatentSize != 1024 || cfg.NumHiddenLayers != 3 || string(cfg.LayerTypes) != "ME*" || !cfg.MambaRoundIntermediates {
		t.Fatalf("unexpected super config: %+v", cfg)
	}
}

func TestNestedMTPPaths(t *testing.T) {
	present := &mlx.Array{}
	tensors := map[string]*mlx.Array{
		"language_model.mtp.layers.0.eh_proj.weight":                 present,
		"language_model.mtp.layers.0.mixer.q_proj.weight":            present,
		"language_model.mtp.layers.1.mixer.experts.0.up_proj.weight": present,
		"mtp.layers.99.mixer.q_proj.weight":                          present,
	}
	const prefix = "language_model.mtp."
	if !hasMTPHeadTensors(tensors, prefix) {
		t.Fatal("nested MTP fusion not detected")
	}
	got, err := mtpLayerPrefixes(tensors, prefix)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{prefix + "layers.0", prefix + "layers.1"}; !slices.Equal(got, want) {
		t.Fatalf("prefixes = %v, want %v", got, want)
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
			got, err := mtpLayerPrefixes(tt.tensors, "mtp.")
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
			if got := hasMTPHeadTensors(tt.tensors, "mtp."); got != tt.want {
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
	mlxtest.Run(t, func(t *mlxtest.T) {
		weight := mlx.FromValues([]float32{
			1, 2,
			3, 4,
			5, 6,
			7, 8,
		}, 2, 2, 2)
		// Scales reach this function already in MLX's representation.
		scale := mlx.FromValues([]float32{2 * mlx.Nvfp4MaxProduct, 3 * mlx.Nvfp4MaxProduct}, 2)

		got := applyExpertWeightGlobalScale(weight, scale)
		mlx.Eval(got)

		assertAllClose(t, "scaled expert weight", got.Floats(), []float32{
			2, 4,
			6, 8,
			15, 18,
			21, 24,
		}, 1e-5)
	})
}

// The production form leans on MLX's fused fast RMS norm, so check it against
// a reference that spells out every step.
func TestGatedGroupRMSNormMatchesElementwiseReference(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
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
	})
}

func TestMambaIntermediateRounding(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		cfg := &Config{MambaNumHeads: 1, MambaHeadDim: 2, NGroups: 1, MambaRoundIntermediates: true}
		scan := mlx.FromValues([]float32{1.004, 2}, 1, 1, 2)
		gate := mlx.FromValues([]float32{1, 1}, 1, 1, 2).AsType(mlx.DTypeBFloat16)
		weight := mlx.FromValues([]float32{1.015625, 1.015625}, 2).AsType(mlx.DTypeBFloat16)
		got := gatedGroupRMSNorm(scan, gate, weight, cfg, mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32)
		mlx.Eval(got)
		// Scan rounds to [129/128, 2]. Equal gates cancel in RMS normalization
		// with epsilon zero, giving BF16 [163/256, 81/64] before scaling by 65/64.
		assertAllClose(t, "intermediate rounding", got.Floats(), []float32{166.0 / 256, 165.0 / 128}, 0)
	})
}

func TestLatentMoEKeepsSharedExpertAtHiddenWidth(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{HiddenSize: 2, MoELatentSize: 1, NRoutedExperts: 2, NumExpertsPerTok: 2, NormTopKProb: true, RoutedScalingFactor: 1}
		m := &SparseMoE{
			RouterWeight: mlx.Zeros(mlx.DTypeFloat32, 2, 2),
			LatentIn:     nn.NewLinear(mlx.FromValues([]float32{1, 0}, 1, 2), nil),
			LatentOut:    nn.NewLinear(mlx.FromValues([]float32{1, 3}, 2, 1), nil),
			UpWeight:     mlx.FromValues([]float32{1, 2}, 2, 1, 1),
			DownWeight:   mlx.FromValues([]float32{1, 1}, 2, 1, 1),
			SharedUp:     nn.NewLinear(mlx.FromValues([]float32{0, 1}, 1, 2), nil),
			SharedDown:   nn.NewLinear(mlx.FromValues([]float32{1, 2}, 2, 1), nil),
		}
		got := m.Forward(mlx.FromValues([]float32{2, 3}, 1, 1, 2), cfg)
		mlx.Eval(got)
		// Routed: (2² + 4²)/2 = 10, projected to [10, 30].
		// Shared: 3² = 9, projected to [9, 18].
		assertAllClose(t, "latent and shared experts", got.Floats(), []float32{19, 48}, 0)
	})
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

func assertAllClose(t *mlxtest.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

// Folding shared experts into the routed banks must keep the global scale
// banks aligned with the weight banks: rows are added, so scales are added
// too — the shared expert's own scale when it has one, MLX's identity scale
// when it does not.
func TestFoldSharedExpertsExtendsGlobalScales(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const routed, inter, sharedInter, hidden = 3, 16, 32, 32
		cfg := &Config{
			NSharedExperts:                  1,
			MoEIntermediateSize:             inter,
			MoESharedExpertIntermediateSize: sharedInter,
			HiddenSize:                      hidden,
		}

		newMoE := func(routedKernelScales []float32, sharedGlobalScale *mlx.Array) *SparseMoE {
			m := &SparseMoE{
				UseQuantized:  true,
				UpWeightQ:     mlx.Zeros(mlx.DTypeUint32, routed, inter, hidden/8),
				UpScales:      mlx.Zeros(mlx.DTypeUint8, routed, inter, hidden/16),
				UpGroupSize:   16,
				UpBits:        4,
				UpMode:        "nvfp4",
				DownWeightQ:   mlx.Zeros(mlx.DTypeUint32, routed, hidden, inter/8),
				DownScales:    mlx.Zeros(mlx.DTypeUint8, routed, hidden, inter/16),
				DownGroupSize: 16,
				DownBits:      4,
				DownMode:      "nvfp4",
				SharedUp: &nn.QuantizedLinear{
					Weight:      mlx.Zeros(mlx.DTypeUint32, sharedInter, hidden/8),
					Scales:      mlx.Zeros(mlx.DTypeUint8, sharedInter, hidden/16),
					GroupSize:   16,
					Bits:        4,
					Mode:        "nvfp4",
					GlobalScale: sharedGlobalScale,
				},
				SharedDown: &nn.QuantizedLinear{
					Weight:      mlx.Zeros(mlx.DTypeUint32, hidden, sharedInter/8),
					Scales:      mlx.Zeros(mlx.DTypeUint8, hidden, sharedInter/16),
					GroupSize:   16,
					Bits:        4,
					Mode:        "nvfp4",
					GlobalScale: sharedGlobalScale,
				},
			}
			if routedKernelScales != nil {
				m.UpGlobalScales = mlx.FromValues(routedKernelScales, routed)
				m.DownGlobalScales = mlx.FromValues(routedKernelScales, routed)
			}
			return m
		}

		checkScales := func(name string, m *SparseMoE, want []float32) {
			t.Helper()
			for _, gs := range []*mlx.Array{m.UpGlobalScales, m.DownGlobalScales} {
				mlx.Eval(gs)
				if dims := gs.Dims(); len(dims) != 1 || dims[0] != len(want) {
					t.Fatalf("%s dims = %v, want [%d]", name, dims, len(want))
				}
				assertAllClose(t, name, gs.Floats(), want, 1e-5)
			}
		}

		// Routed experts carry kernel scales; the scaleless shared expert folds
		// as the identity scale.
		m := newMoE([]float32{1, 2, 3}, nil)
		if !foldSharedExperts(m, cfg) {
			t.Fatal("foldSharedExperts failed with routed global scales and a scaleless shared expert")
		}
		checkScales("identity extension", m, []float32{1, 2, 3, mlx.Nvfp4MaxProduct, mlx.Nvfp4MaxProduct})

		// The shared expert's own per-tensor scale folds as a routed expert's.
		m = newMoE([]float32{1, 2, 3}, mlx.FromValues([]float32{0.25 * mlx.Nvfp4MaxProduct}, 1))
		if !foldSharedExperts(m, cfg) {
			t.Fatal("foldSharedExperts failed with a per-tensor shared expert scale")
		}
		shared := float32(0.25 * mlx.Nvfp4MaxProduct)
		checkScales("shared scale extension", m, []float32{1, 2, 3, shared, shared})

		// Scaleless routed experts fold against an identity bank when the
		// shared expert carries a scale.
		m = newMoE(nil, mlx.FromValues([]float32{0.25 * mlx.Nvfp4MaxProduct}, 1))
		if !foldSharedExperts(m, cfg) {
			t.Fatal("foldSharedExperts failed with a scaleless routed bank")
		}
		identity := float32(mlx.Nvfp4MaxProduct)
		checkScales("identity bank", m, []float32{identity, identity, identity, shared, shared})

		// Nothing to scale: the banks stay nil.
		m = newMoE(nil, nil)
		if !foldSharedExperts(m, cfg) {
			t.Fatal("foldSharedExperts failed without global scales")
		}
		if m.UpGlobalScales != nil || m.DownGlobalScales != nil {
			t.Fatal("fold without global scales materialized a scale bank")
		}

		// A per-row shared expert scale cannot be represented per expert bank
		// row, so the fold is skipped.
		m = newMoE([]float32{1, 2, 3}, mlx.FromValues([]float32{1 * mlx.Nvfp4MaxProduct, 2 * mlx.Nvfp4MaxProduct}, 2))
		if foldSharedExperts(m, cfg) {
			t.Fatal("foldSharedExperts accepted a per-row shared expert scale")
		}
		if dims := m.UpWeightQ.Dims(); dims[0] != routed {
			t.Fatalf("skipped fold mutated the weight bank: dims = %v", dims)
		}

		cfg.MoELatentSize = hidden / 2
		if foldSharedExperts(newMoE(nil, nil), cfg) {
			t.Fatal("foldSharedExperts accepted different routed and shared widths")
		}
	})
}

func TestParseVisionConfigOmni(t *testing.T) {
	cfg, err := parseVisionConfig([]byte(`{
		"ps_version": "v2",
		"vision_config": {
			"version": "radio_v2.5-h",
			"patch_size": 16,
			"min_num_patches": 1024,
			"max_num_patches": 13312,
			"args": {"model": "vit_huge_patch16_224"}
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
	if got, want := cfg.ModelName, "vit_huge_patch16_224"; got != want {
		t.Fatalf("ModelName = %q, want %q", got, want)
	}
	if got, want := cfg.PSVersion, "v2"; got != want {
		t.Fatalf("PSVersion = %q, want %q", got, want)
	}
	if got, want := cfg.NumAttentionHeads, int32(16); got != want {
		t.Fatalf("NumAttentionHeads = %d, want %d", got, want)
	}
	if cfg.PatchSize != 0 || cfg.HiddenSize != 0 || cfg.NumHiddenLayers != 0 || cfg.HeadDim != 0 {
		t.Fatalf("weight-derived dimensions initialized before weights: %+v", cfg)
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
		"ps_version": "v2",
		"vision_config": {"version": "c-radio_v4-h", "args": {"model": "vit_huge_patch16_224"}},
		"norm_mean": [0.1, 0.2]
	}`)
	if _, err := parseVisionConfig(config, nil); err == nil {
		t.Fatal("expected malformed config norm_mean error")
	}

	config = []byte(`{"ps_version":"v2","vision_config":{"version":"c-radio_v4-h","args":{"model":"vit_huge_patch16_224"}}}`)
	preprocessor := []byte(`{"norm_std": [0.1, 0.2, 0.3, 0.4]}`)
	if _, err := parseVisionConfig(config, preprocessor); err == nil {
		t.Fatal("expected malformed preprocessor norm_std error")
	}
}

func TestParseVisionConfigRequiresDynamicResolutionBounds(t *testing.T) {
	for _, config := range []string{
		`{"ps_version":"v2","vision_config":{"version":"c-radio_v4-h","min_num_patches":1024,"args":{"model":"vit_huge_patch16_224"}}}`,
		`{"ps_version":"v2","vision_config":{"version":"c-radio_v4-h","max_num_patches":13312,"args":{"model":"vit_huge_patch16_224"}}}`,
		`{"ps_version":"v2","vision_config":{"version":"c-radio_v4-h","args":{"model":"vit_huge_patch16_224"}}}`,
	} {
		if _, err := parseVisionConfig([]byte(config), nil); err == nil || !strings.Contains(err.Error(), "requires min_num_patches and max_num_patches") {
			t.Fatalf("parseVisionConfig error = %v, want missing dynamic-resolution bounds", err)
		}
	}

	_, err := parseVisionConfig([]byte(`{
		"ps_version": "v2",
		"vision_config": {
			"version": "c-radio_v4-h",
			"min_num_patches": 2048,
			"max_num_patches": 1024,
			"args": {"model": "vit_huge_patch16_224"}
		}
	}`), nil)
	if err == nil || !strings.Contains(err.Error(), "min_num_patches (2048) exceeds max_num_patches (1024)") {
		t.Fatalf("parseVisionConfig error = %v, want reversed bounds error", err)
	}
}

func TestParseVisionConfigRejectsUnknownArchitecture(t *testing.T) {
	for _, tt := range []struct {
		name   string
		config string
		want   string
	}{
		{
			name:   "missing model",
			config: `{"ps_version":"v2","vision_config":{"min_num_patches":1024,"max_num_patches":13312}}`,
			want:   `unsupported RADIO model ""`,
		},
		{
			name:   "unknown model",
			config: `{"ps_version":"v2","vision_config":{"min_num_patches":1024,"max_num_patches":13312,"args":{"model":"vit_custom_patch16_224"}}}`,
			want:   `unsupported RADIO model "vit_custom_patch16_224"`,
		},
		{
			name:   "missing ps_version",
			config: `{"vision_config":{"min_num_patches":1024,"max_num_patches":13312,"args":{"model":"vit_huge_patch16_224"}}}`,
			want:   `unsupported RADIO ps_version ""`,
		},
		{
			name:   "unsupported ps_version",
			config: `{"ps_version":"v1","vision_config":{"min_num_patches":1024,"max_num_patches":13312,"args":{"model":"vit_huge_patch16_224"}}}`,
			want:   `unsupported RADIO ps_version "v1"`,
		},
		{
			name:   "mismatched head count",
			config: `{"ps_version":"v2","vision_config":{"num_attention_heads":8,"min_num_patches":1024,"max_num_patches":13312,"args":{"model":"vit_huge_patch16_224"}}}`,
			want:   `RADIO model "vit_huge_patch16_224" has num_attention_heads=16, config has 8`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseVisionConfig([]byte(tt.config), nil)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("parseVisionConfig error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestRadioPatchDimensions(t *testing.T) {
	hiddenSize, patchSize, err := radioPatchDimensions([]int{1280, 3 * 16 * 16})
	if err != nil {
		t.Fatal(err)
	}
	if hiddenSize != 1280 || patchSize != 16 {
		t.Fatalf("dimensions = (%d, %d), want (1280, 16)", hiddenSize, patchSize)
	}

	for _, dims := range [][]int{{1280}, {1280, 767}, {1280, 3 * 15}} {
		if _, _, err := radioPatchDimensions(dims); err == nil {
			t.Fatalf("radioPatchDimensions(%v) succeeded, want error", dims)
		}
	}
}

func TestRadioLayerCount(t *testing.T) {
	const prefix = "vision_model.radio_model.model."
	tensors := map[string]*mlx.Array{
		prefix + "blocks.0.norm1.weight":   nil,
		prefix + "blocks.1.norm1.weight":   nil,
		prefix + "blocks.1.mlp.fc1.weight": nil,
		"unrelated.blocks.2.weight":        nil,
	}
	if got, err := radioLayerCount(tensors, prefix); err != nil || got != 2 {
		t.Fatalf("radioLayerCount = %d, %v, want 2, nil", got, err)
	}

	delete(tensors, prefix+"blocks.1.norm1.weight")
	delete(tensors, prefix+"blocks.1.mlp.fc1.weight")
	tensors[prefix+"blocks.2.norm1.weight"] = nil
	if _, err := radioLayerCount(tensors, prefix); err == nil || !strings.Contains(err.Error(), "missing RADIO transformer block 1") {
		t.Fatalf("radioLayerCount error = %v, want missing block 1", err)
	}
}

func TestVisionConfigLoadsArchitectureFromWeights(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const prefix = "vision_model.radio_model.model."
		weight := mlx.Zeros(mlx.DTypeBFloat16, 384, 3*16*16)
		tensors := map[string]*mlx.Array{prefix + "patch_generator.embedder.weight": weight}
		for i := range 12 {
			tensors[fmt.Sprintf("%sblocks.%d.norm1.weight", prefix, i)] = weight
		}

		cfg := &VisionConfig{ModelName: "vit_small_patch16_224", NumAttentionHeads: 6}
		if err := cfg.loadArchitecture(tensors, prefix); err != nil {
			t.Fatal(err)
		}
		if cfg.PatchSize != 16 || cfg.HiddenSize != 384 || cfg.NumHiddenLayers != 12 || cfg.HeadDim != 64 {
			t.Fatalf("derived architecture = %+v", cfg)
		}
	})
}

func TestLoadVisionWeightsRejectsNonSquarePositionGrid(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
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
	})
}

func TestResizeRadioImagePreservesFloatRange(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	copy(src.Pix, []byte{0, 0, 0, 255, 255, 255, 255, 255})
	got := resizeRadioImage(src, 1, 4)
	// The half-pixel bicubic filter overshoots at the ends and produces
	// fractional interior values; neither may be clamped or rounded to uint8.
	want := []float32{-22.5, 255 * 29.0 / 140, 255 * 111.0 / 140, 277.5}
	if len(got) != 3*len(want) {
		t.Fatalf("pixel count = %d, want %d", len(got), 3*len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i%len(want)])) > 3e-5 {
			t.Fatalf("pixel[%d] = %v, want %v", i, v, want[i%len(want)])
		}
	}
}

func TestResizeGrid2DPreservesSourceDType(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		x := mlx.FromValues([]float32{0, 1, 2, 3}, 1, 2, 2, 1).AsType(mlx.DTypeBFloat16)
		got := resizeGrid2D(x, 3, 3)
		mlx.Eval(got)
		if got.DType() != mlx.DTypeBFloat16 {
			t.Fatalf("resizeGrid2D dtype = %s, want bfloat16", got.DType())
		}
	})
}

func TestResizeRadioImageChannels(t *testing.T) {
	src := image.NewNRGBA(image.Rect(3, 5, 5, 7))
	copy(src.Pix, []byte{0, 10, 20, 255, 40, 50, 60, 255, 80, 90, 100, 255, 120, 130, 140, 255})
	for _, tt := range []struct {
		height, width int
		want          []float32
	}{
		{1, 1, []float32{60, 70, 80}},
		{1, 2, []float32{40, 80, 50, 90, 60, 100}},
		{2, 2, []float32{0, 40, 80, 120, 10, 50, 90, 130, 20, 60, 100, 140}},
	} {
		t.Run(fmt.Sprintf("%dx%d", tt.width, tt.height), func(t *testing.T) {
			if got := resizeRadioImage(src, tt.height, tt.width); !slices.Equal(got, tt.want) {
				t.Errorf("pixels = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestResizeRadioImagePixelAccess(t *testing.T) {
	bounds := image.Rect(3, 5, 12, 13)
	crop := image.Rect(5, 7, 10, 11)
	nrgba := image.NewNRGBA(bounds)
	for i := range nrgba.Pix {
		nrgba.Pix[i] = byte(i * 31)
	}
	for i := 3; i < len(nrgba.Pix); i += 4 {
		nrgba.Pix[i] = [4]byte{0, 1, 127, 255}[(i/4)%4]
	}
	images := []image.Image{nrgba.SubImage(crop)}
	for _, ratio := range []image.YCbCrSubsampleRatio{
		image.YCbCrSubsampleRatio444, image.YCbCrSubsampleRatio422,
		image.YCbCrSubsampleRatio420, image.YCbCrSubsampleRatio440,
		image.YCbCrSubsampleRatio411, image.YCbCrSubsampleRatio410,
	} {
		src := image.NewYCbCr(bounds, ratio)
		for i := range src.Y {
			src.Y[i] = byte(i * 31)
		}
		for i := range src.Cb {
			src.Cb[i], src.Cr[i] = byte(i*17), byte(i*43)
		}
		images = append(images, src.SubImage(crop))
	}
	for _, src := range images {
		// Hiding the concrete image type exercises the generic color conversion.
		generic := struct{ image.Image }{src}
		for _, size := range [][2]int{{2, 3}, {4, 5}, {7, 9}} {
			got := resizeRadioImage(src, size[0], size[1])
			want := resizeRadioImage(generic, size[0], size[1])
			if !slices.Equal(got, want) {
				t.Errorf("%T resize to %v differs from generic pixel access", src, size)
			}
		}
	}
}

func TestNemotronImagePatchBudget(t *testing.T) {
	cfg := &VisionConfig{
		DownsampleFactor: 2,
		MinNumPatches:    1024,
		MaxNumPatches:    13312,
		MaxModelLen:      16384,
	}

	// Context-bound, not memory-bound: model.MediaModel requires PrepareMedia
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
		wantErr    bool
	}{
		{name: "square minimum", height: 512, width: 512, budget: 1024, wantMax: 1024, wantMin: 1024, wantFactor: 2},
		{name: "large square capped", height: 2048, width: 2048, budget: 13312, wantMax: 13312, wantMin: 1024, wantFactor: 2},
		{name: "wide image capped", height: 512, width: 2048, budget: 4096, wantMax: 4096, wantMin: 1024, wantFactor: 2},
		{name: "extremely wide image", height: 16, width: 131072, budget: 13312, wantErr: true},
		{name: "extremely tall image", height: 131072, width: 16, budget: 13312, wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			patchH, patchW, err := nemotronImagePatchGrid(tt.height, tt.width, tt.budget, cfg)
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), "exceeds the patch budget") {
					t.Fatalf("error = %v, want patch budget error", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := patchH * patchW; got > tt.wantMax || got < tt.wantMin {
				t.Fatalf("patches = %d (%dx%d), want within [%d,%d]", got, patchH, patchW, tt.wantMin, tt.wantMax)
			}
			if patchH%tt.wantFactor != 0 || patchW%tt.wantFactor != 0 {
				t.Fatalf("patch grid = %dx%d, want both divisible by %d", patchH, patchW, tt.wantFactor)
			}
		})
	}
}
