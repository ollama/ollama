package dflash

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestParseConfigYarnRopeScaling(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	data := []byte(`{
		"hidden_size": 2048,
		"num_hidden_layers": 8,
		"num_attention_heads": 32,
		"num_key_value_heads": 4,
		"head_dim": 128,
		"intermediate_size": 6144,
		"vocab_size": 248320,
		"rms_norm_eps": 0.000001,
		"rope_theta": 10000000,
		"rope_scaling": {
			"beta_fast": 32.0,
			"beta_slow": 1.0,
			"factor": 64.0,
			"original_max_position_embeddings": 4096,
			"rope_type": "yarn"
		},
		"block_size": 16,
		"num_target_layers": 40,
		"layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention"],
		"dflash_config": {
			"mask_token_id": 248070,
			"target_layer_ids": [1, 10, 19, 28, 37]
		}
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}
	if cfg.RopeFreqs == nil {
		t.Fatalf("RopeFreqs is nil")
	}
	wantScale := float32(0.1*math.Log(64.0) + 1.0)
	if math.Abs(float64(cfg.RopeScale-wantScale)) > 1e-6 {
		t.Fatalf("RopeScale = %v, want %v", cfg.RopeScale, wantScale)
	}
}
