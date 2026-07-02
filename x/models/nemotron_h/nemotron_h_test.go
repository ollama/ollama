package nemotron_h

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
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
