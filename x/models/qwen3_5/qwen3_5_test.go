//go:build mlx

package qwen3_5

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

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
	cfg := &Config{
		NumHiddenLayers:       6,
		FullAttentionInterval: 3,
		NumExperts:            8,
		DecoderSparseStep:     2,
		MLPOnlyLayers:         []int32{1},
	}

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

func TestModelRuntimeToggles(t *testing.T) {
	m := &Model{}
	if !m.DisablePromptCache() {
		t.Fatal("DisablePromptCache() = false, want true")
	}
	if m.EnableCompile() {
		t.Fatal("EnableCompile() = true, want false")
	}
}

func TestNewCachesLayout(t *testing.T) {
	m := &Model{
		Config: &Config{
			LinearConvKernelDim: 4,
			LinearNumKeyHeads:   2,
			LinearKeyHeadDim:    8,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  16,
		},
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
