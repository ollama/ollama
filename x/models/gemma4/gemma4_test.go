package gemma4

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestParseTextConfigE2B(t *testing.T) {
	skipIfNoMLX(t)
	data := []byte(`{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"hidden_size": 1536,
			"num_hidden_layers": 35,
			"intermediate_size": 6144,
			"num_attention_heads": 8,
			"num_key_value_heads": 1,
			"head_dim": 256,
			"global_head_dim": 512,
			"vocab_size": 262144,
			"rms_norm_eps": 1e-6,
			"max_position_embeddings": 131072,
			"sliding_window": 512,
			"sliding_window_pattern": 5,
			"final_logit_softcapping": 30.0,
			"use_double_wide_mlp": true,
			"num_kv_shared_layers": 20,
			"hidden_size_per_layer_input": 256,
			"vocab_size_per_layer_input": 262144,
			"attention_k_eq_v": false,
			"tie_word_embeddings": true,
			"layer_types": [
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
			],
			"rope_parameters": {
				"full_attention": {
					"partial_rotary_factor": 0.25,
					"rope_theta": 1000000.0,
					"rope_type": "proportional"
				},
				"sliding_attention": {
					"rope_theta": 10000.0,
					"rope_type": "default"
				}
			}
		}
	}`)

	cfg, err := parseTextConfig(data)
	if err != nil {
		t.Fatalf("parseTextConfig failed: %v", err)
	}

	// Basic fields.
	if cfg.HiddenSize != 1536 {
		t.Errorf("HiddenSize = %d, want 1536", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers != 35 {
		t.Errorf("NumHiddenLayers = %d, want 35", cfg.NumHiddenLayers)
	}
	if cfg.GlobalHeadDim != 512 {
		t.Errorf("GlobalHeadDim = %d, want 512", cfg.GlobalHeadDim)
	}
	if cfg.FinalLogitSoftcapping != 30.0 {
		t.Errorf("FinalLogitSoftcapping = %f, want 30.0", cfg.FinalLogitSoftcapping)
	}
	if cfg.NumKVSharedLayers != 20 {
		t.Errorf("NumKVSharedLayers = %d, want 20", cfg.NumKVSharedLayers)
	}
	if cfg.HiddenSizePerLayer != 256 {
		t.Errorf("HiddenSizePerLayer = %d, want 256", cfg.HiddenSizePerLayer)
	}

	// RoPE settings.
	if cfg.SlidingRopeDims != 256 {
		t.Errorf("SlidingRopeDims = %d, want 256", cfg.SlidingRopeDims)
	}
	if cfg.FullRopeDims != 512 {
		t.Errorf("FullRopeDims = %d, want 512 (GlobalHeadDim, partial rotation handled via custom freqs)", cfg.FullRopeDims)
	}
	if cfg.SlidingRopeBase != 10000 {
		t.Errorf("SlidingRopeBase = %f, want 10000", cfg.SlidingRopeBase)
	}
	if cfg.FullRopeBase != 1000000 {
		t.Errorf("FullRopeBase = %f, want 1000000", cfg.FullRopeBase)
	}

	// Attention scale.
	if cfg.SlidingScale == 0 || cfg.FullScale == 0 {
		t.Error("attention scales should be non-zero")
	}

	// KV sharing map.
	// First shared layer is 35 - 20 = 15.
	if donor, ok := cfg.KVShareMap[15]; !ok || donor != 13 {
		t.Errorf("KVShareMap[15] = %d, ok=%v; want 13, true", donor, ok)
	}
	if donor, ok := cfg.KVShareMap[19]; !ok || donor != 14 {
		t.Errorf("KVShareMap[19] = %d, ok=%v; want 14, true (full attn donor)", donor, ok)
	}
	if donor, ok := cfg.KVShareMap[34]; !ok || donor != 14 {
		t.Errorf("KVShareMap[34] = %d, ok=%v; want 14, true (full attn donor)", donor, ok)
	}
	// Layer 14 should not be shared.
	if _, ok := cfg.KVShareMap[14]; ok {
		t.Error("layer 14 should not be in KVShareMap (non-shared)")
	}

	// Donors.
	if !cfg.KVDonors[13] {
		t.Error("layer 13 should be a KV donor")
	}
	if !cfg.KVDonors[14] {
		t.Error("layer 14 should be a KV donor")
	}
}

func TestParseTextConfig26B(t *testing.T) {
	skipIfNoMLX(t)
	data := []byte(`{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"hidden_size": 2816,
			"num_hidden_layers": 30,
			"intermediate_size": 2112,
			"num_attention_heads": 16,
			"num_key_value_heads": 8,
			"num_global_key_value_heads": 2,
			"head_dim": 256,
			"global_head_dim": 512,
			"vocab_size": 262144,
			"rms_norm_eps": 1e-6,
			"max_position_embeddings": 131072,
			"sliding_window": 1024,
			"final_logit_softcapping": 30.0,
			"use_double_wide_mlp": false,
			"num_kv_shared_layers": 0,
			"hidden_size_per_layer_input": null,
			"attention_k_eq_v": true,
			"enable_moe_block": true,
			"num_experts": 128,
			"top_k_experts": 8,
			"moe_intermediate_size": 704,
			"tie_word_embeddings": true,
			"layer_types": [
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
			],
			"rope_parameters": {
				"full_attention": {
					"partial_rotary_factor": 0.25,
					"rope_theta": 1000000.0,
					"rope_type": "proportional"
				},
				"sliding_attention": {
					"rope_theta": 10000.0,
					"rope_type": "default"
				}
			}
		}
	}`)

	cfg, err := parseTextConfig(data)
	if err != nil {
		t.Fatalf("parseTextConfig failed: %v", err)
	}

	if cfg.HiddenSize != 2816 {
		t.Errorf("HiddenSize = %d, want 2816", cfg.HiddenSize)
	}
	if !cfg.AttentionKEqV {
		t.Error("AttentionKEqV should be true")
	}
	if cfg.NumGlobalKeyValueHeads != 2 {
		t.Errorf("NumGlobalKeyValueHeads = %d, want 2", cfg.NumGlobalKeyValueHeads)
	}
	if !cfg.EnableMoeBlock {
		t.Error("EnableMoeBlock should be true")
	}
	if cfg.NumExperts != 128 {
		t.Errorf("NumExperts = %d, want 128", cfg.NumExperts)
	}
	if cfg.TopKExperts != 8 {
		t.Errorf("TopKExperts = %d, want 8", cfg.TopKExperts)
	}
	if cfg.ExpertIntermediateSize != 704 {
		t.Errorf("ExpertIntermediateSize = %d, want 704", cfg.ExpertIntermediateSize)
	}
	if cfg.HiddenSizePerLayer != 0 {
		t.Errorf("HiddenSizePerLayer = %d, want 0 (no PLE)", cfg.HiddenSizePerLayer)
	}
}

func TestParseTextConfig31B(t *testing.T) {
	skipIfNoMLX(t)
	data := []byte(`{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"hidden_size": 5376,
			"num_hidden_layers": 60,
			"intermediate_size": 21504,
			"num_attention_heads": 32,
			"num_key_value_heads": 16,
			"num_global_key_value_heads": 4,
			"head_dim": 256,
			"global_head_dim": 512,
			"vocab_size": 262144,
			"rms_norm_eps": 1e-6,
			"max_position_embeddings": 131072,
			"sliding_window": 1024,
			"final_logit_softcapping": 30.0,
			"use_double_wide_mlp": false,
			"num_kv_shared_layers": 0,
			"hidden_size_per_layer_input": null,
			"attention_k_eq_v": true,
			"tie_word_embeddings": true,
			"layer_types": [
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
			],
			"rope_parameters": {
				"full_attention": {
					"partial_rotary_factor": 0.25,
					"rope_theta": 1000000.0,
					"rope_type": "proportional"
				},
				"sliding_attention": {
					"rope_theta": 10000.0,
					"rope_type": "default"
				}
			}
		}
	}`)

	cfg, err := parseTextConfig(data)
	if err != nil {
		t.Fatalf("parseTextConfig failed: %v", err)
	}

	if cfg.HiddenSize != 5376 {
		t.Errorf("HiddenSize = %d, want 5376", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers != 60 {
		t.Errorf("NumHiddenLayers = %d, want 60", cfg.NumHiddenLayers)
	}
	if !cfg.AttentionKEqV {
		t.Error("AttentionKEqV should be true")
	}
	if cfg.NumGlobalKeyValueHeads != 4 {
		t.Errorf("NumGlobalKeyValueHeads = %d, want 4", cfg.NumGlobalKeyValueHeads)
	}
	if cfg.NumKeyValueHeads != 16 {
		t.Errorf("NumKeyValueHeads = %d, want 16", cfg.NumKeyValueHeads)
	}
	if cfg.NumKVSharedLayers != 0 {
		t.Errorf("NumKVSharedLayers = %d, want 0", cfg.NumKVSharedLayers)
	}
	if cfg.HiddenSizePerLayer != 0 {
		t.Errorf("HiddenSizePerLayer = %d, want 0 (no PLE)", cfg.HiddenSizePerLayer)
	}
	if cfg.SlidingWindow != 1024 {
		t.Errorf("SlidingWindow = %d, want 1024", cfg.SlidingWindow)
	}

	// KV sharing should be empty (no shared layers).
	if len(cfg.KVShareMap) != 0 {
		t.Errorf("KVShareMap should be empty, got %d entries", len(cfg.KVShareMap))
	}

	// Layer types: pattern is 5 sliding + 1 full, repeating 10 times.
	if !isLayerSliding(0, &cfg) {
		t.Error("layer 0 should be sliding")
	}
	if isLayerSliding(5, &cfg) {
		t.Error("layer 5 should be full attention")
	}
	if !isLayerSliding(6, &cfg) {
		t.Error("layer 6 should be sliding")
	}
	if isLayerSliding(59, &cfg) {
		t.Error("layer 59 should be full attention")
	}
}

func TestParseTextConfigE4B(t *testing.T) {
	skipIfNoMLX(t)
	data := []byte(`{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"hidden_size": 2560,
			"num_hidden_layers": 42,
			"intermediate_size": 10240,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 256,
			"global_head_dim": 512,
			"vocab_size": 262144,
			"rms_norm_eps": 1e-6,
			"max_position_embeddings": 131072,
			"sliding_window": 512,
			"final_logit_softcapping": 30.0,
			"use_double_wide_mlp": false,
			"num_kv_shared_layers": 18,
			"hidden_size_per_layer_input": 256,
			"vocab_size_per_layer_input": 262144,
			"attention_k_eq_v": false,
			"enable_moe_block": false,
			"tie_word_embeddings": true,
			"layer_types": [
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
				"sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
			],
			"rope_parameters": {
				"full_attention": {
					"partial_rotary_factor": 0.25,
					"rope_theta": 1000000.0,
					"rope_type": "proportional"
				},
				"sliding_attention": {
					"rope_theta": 10000.0,
					"rope_type": "default"
				}
			}
		}
	}`)

	cfg, err := parseTextConfig(data)
	if err != nil {
		t.Fatalf("parseTextConfig failed: %v", err)
	}

	if cfg.HiddenSize != 2560 {
		t.Errorf("HiddenSize = %d, want 2560", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers != 42 {
		t.Errorf("NumHiddenLayers = %d, want 42", cfg.NumHiddenLayers)
	}
	if cfg.IntermediateSize != 10240 {
		t.Errorf("IntermediateSize = %d, want 10240", cfg.IntermediateSize)
	}
	if cfg.NumKeyValueHeads != 2 {
		t.Errorf("NumKeyValueHeads = %d, want 2", cfg.NumKeyValueHeads)
	}
	if cfg.UseDoubleWideMLP {
		t.Error("UseDoubleWideMLP should be false")
	}
	if cfg.NumKVSharedLayers != 18 {
		t.Errorf("NumKVSharedLayers = %d, want 18", cfg.NumKVSharedLayers)
	}
	if cfg.HiddenSizePerLayer != 256 {
		t.Errorf("HiddenSizePerLayer = %d, want 256 (has PLE)", cfg.HiddenSizePerLayer)
	}
	if cfg.AttentionKEqV {
		t.Error("AttentionKEqV should be false")
	}
	if cfg.EnableMoeBlock {
		t.Error("EnableMoeBlock should be false")
	}
	if cfg.SlidingWindow != 512 {
		t.Errorf("SlidingWindow = %d, want 512", cfg.SlidingWindow)
	}

	// Layer types: pattern is 5 sliding + 1 full, repeating 7 times = 42 layers.
	if !isLayerSliding(0, &cfg) {
		t.Error("layer 0 should be sliding")
	}
	if isLayerSliding(5, &cfg) {
		t.Error("layer 5 should be full attention")
	}
	if !isLayerSliding(6, &cfg) {
		t.Error("layer 6 should be sliding")
	}
	if isLayerSliding(41, &cfg) {
		t.Error("layer 41 should be full attention")
	}

	// KV sharing: first shared = 42 - 18 = 24.
	// Layer 24 is sliding, its donor should be the last non-shared sliding layer.
	// Non-shared layers: 0-23. Last sliding in 0-23 is layer 22 (23=full).
	if donor, ok := cfg.KVShareMap[24]; !ok {
		t.Error("layer 24 should be in KVShareMap")
	} else {
		t.Logf("layer 24 donor = %d", donor)
	}
	// Layer 29 is full_attention (5th full), donor should be the last non-shared full layer.
	// Non-shared full layers: 5, 11, 17, 23.
	if donor, ok := cfg.KVShareMap[29]; !ok || donor != 23 {
		t.Errorf("KVShareMap[29] = %d, ok=%v; want 23, true (full attn donor)", donor, ok)
	}
	// Layer 23 should NOT be shared (it's the last non-shared layer).
	if _, ok := cfg.KVShareMap[23]; ok {
		t.Error("layer 23 should not be in KVShareMap (non-shared)")
	}
}

func TestLayerTypeDetection(t *testing.T) {
	cfg := &TextConfig{
		LayerTypes: []string{
			"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
		},
	}

	if !isLayerSliding(0, cfg) {
		t.Error("layer 0 should be sliding")
	}
	if !isLayerSliding(3, cfg) {
		t.Error("layer 3 should be sliding")
	}
	if isLayerSliding(4, cfg) {
		t.Error("layer 4 should be full attention")
	}
}

func TestNewCachesOmitsSharedKVLayers(t *testing.T) {
	m := &Model{
		Layers: []*DecoderLayer{
			{IsSliding: true, KVShareDonor: -1},
			{IsSliding: false, KVShareDonor: -1},
			{IsSliding: true, KVShareDonor: 0},
			{IsSliding: false, KVShareDonor: 1},
		},
		TextConfig: &TextConfig{SlidingWindow: 512},
	}

	caches := m.NewCaches()
	if got, want := len(caches), 2; got != want {
		t.Fatalf("len(NewCaches()) = %d, want %d", got, want)
	}
}

func TestNewCachesIncludesAllNonSharedLayers(t *testing.T) {
	m := &Model{
		Layers: []*DecoderLayer{
			{IsSliding: true, KVShareDonor: -1},
			{IsSliding: false, KVShareDonor: -1},
			{IsSliding: true, KVShareDonor: -1},
		},
		TextConfig: &TextConfig{SlidingWindow: 512},
	}

	caches := m.NewCaches()
	if got, want := len(caches), len(m.Layers); got != want {
		t.Fatalf("len(NewCaches()) = %d, want %d", got, want)
	}
}

func TestResolveWeightPrefix(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	tests := []struct {
		name    string
		key     string
		wantPfx string
	}{
		{"bare", "embed_tokens.weight", ""},
		{"language_model", "model.language_model.embed_tokens.weight", "model.language_model."},
		{"with_model", "model.embed_tokens.weight", "model."},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dummy := mlx.FromValue(float32(1.0))
			mlx.Eval(dummy)
			tensors := map[string]*mlx.Array{tt.key: dummy}
			got := resolveWeightPrefix(tensors)
			if got != tt.wantPfx {
				t.Errorf("resolveWeightPrefix(%q) = %q, want %q", tt.key, got, tt.wantPfx)
			}
		})
	}
}

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}
