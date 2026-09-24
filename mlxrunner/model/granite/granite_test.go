package granite

import (
	"testing"
)

func TestParseConfigGranite42(t *testing.T) {
	configJSON := []byte(`{
		"architectures": ["GraniteForCausalLM"],
		"hidden_size": 2560,
		"num_hidden_layers": 40,
		"intermediate_size": 8192,
		"num_attention_heads": 40,
		"num_key_value_heads": 8,
		"tie_word_embeddings": false,
		"vocab_size": 100352,
		"rms_norm_eps": 1e-05,
		"rope_theta": 10000000,
		"max_position_embeddings": 131072,
		"attention_multiplier": 0.015625,
		"embedding_multiplier": 1.0,
		"residual_multiplier": 1.0,
		"logits_scaling": 1.0,
		"rope_parameters": {
			"rope_theta": 10000000,
			"rope_type": "default"
		},
		"rope_scaling": null
	}`)

	cfg, err := parseConfig(configJSON)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	// Verify multiplier fields
	if cfg.AttentionMultiplier != 0.015625 {
		t.Errorf("attention_multiplier = %v, want 0.015625", cfg.AttentionMultiplier)
	}
	if cfg.EmbeddingMultiplier != 1.0 {
		t.Errorf("embedding_multiplier = %v, want 1.0", cfg.EmbeddingMultiplier)
	}
	if cfg.ResidualMultiplier != 1.0 {
		t.Errorf("residual_multiplier = %v, want 1.0", cfg.ResidualMultiplier)
	}
	if cfg.LogitsScaling != 1.0 {
		t.Errorf("logits_scaling = %v, want 1.0", cfg.LogitsScaling)
	}

	// Scale should equal AttentionMultiplier (NOT 1/sqrt(head_dim))
	expectedScale := float32(0.015625)
	if cfg.Scale != expectedScale {
		t.Errorf("Scale = %v, want %v (equals AttentionMultiplier, NOT 1/sqrt(head_dim))", cfg.Scale, expectedScale)
	}

	// rope_parameters.rope_type="default" is a no-op — no YaRN freqs
	if cfg.RopeFreqs != nil {
		t.Errorf("RopeFreqs = %v, want nil (rope_type=\"default\" is a no-op)", cfg.RopeFreqs)
	}
	if cfg.RopeMScale != 1.0 {
		t.Errorf("RopeMScale = %v, want 1.0", cfg.RopeMScale)
	}

	// Verify other config fields
	if cfg.HeadDim != 64 { // 2560 / 40
		t.Errorf("HeadDim = %v, want 64", cfg.HeadDim)
	}
	if cfg.RopeTheta != 10000000 {
		t.Errorf("RopeTheta = %v, want 10000000", cfg.RopeTheta)
	}
	if cfg.RMSNormEps != 1e-05 {
		t.Errorf("RMSNormEps = %v, want 1e-05", cfg.RMSNormEps)
	}
}

func TestParseConfigDefaults(t *testing.T) {
	// Config with the four Granite multiplier keys omitted entirely;
	// they should all default to 1.0 (and Scale therefore also defaults to 1.0,
	// matching AttentionMultiplier's default).
	configJSON := []byte(`{
		"architectures": ["GraniteForCausalLM"],
		"hidden_size": 4096,
		"num_hidden_layers": 32,
		"intermediate_size": 11008,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"tie_word_embeddings": false,
		"vocab_size": 32000,
		"rms_norm_eps": 1e-05,
		"rope_theta": 1000000,
		"max_position_embeddings": 8192,
		"rope_parameters": null,
		"rope_scaling": null
	}`)

	cfg, err := parseConfig(configJSON)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.EmbeddingMultiplier != 1.0 {
		t.Errorf("embedding_multiplier default = %v, want 1.0", cfg.EmbeddingMultiplier)
	}
	if cfg.AttentionMultiplier != 1.0 {
		t.Errorf("attention_multiplier default = %v, want 1.0", cfg.AttentionMultiplier)
	}
	if cfg.ResidualMultiplier != 1.0 {
		t.Errorf("residual_multiplier default = %v, want 1.0", cfg.ResidualMultiplier)
	}
	if cfg.LogitsScaling != 1.0 {
		t.Errorf("logits_scaling default = %v, want 1.0", cfg.LogitsScaling)
	}

	// Scale should equal AttentionMultiplier's default (1.0).
	if cfg.Scale != 1.0 {
		t.Errorf("Scale default = %v, want 1.0 (equals AttentionMultiplier)", cfg.Scale)
	}

	// rope_type="default" is the no-op path; even with nil rope_parameters
	// the defaults should hold.
	if cfg.RopeFreqs != nil {
		t.Errorf("RopeFreqs = %v, want nil", cfg.RopeFreqs)
	}
	if cfg.RopeMScale != 1.0 {
		t.Errorf("RopeMScale = %v, want 1.0", cfg.RopeMScale)
	}
}

func TestParseConfigYaRNScaling(t *testing.T) {
	// Config with rope_type="yarn" — should build YaRN frequencies.
	configJSON := []byte(`{
		"architectures": ["GraniteForCausalLM"],
		"hidden_size": 2560,
		"num_hidden_layers": 40,
		"intermediate_size": 8192,
		"num_attention_heads": 40,
		"num_key_value_heads": 8,
		"tie_word_embeddings": false,
		"vocab_size": 100352,
		"rms_norm_eps": 1e-05,
		"rope_theta": 10000,
		"max_position_embeddings": 131072,
		"attention_multiplier": 0.015625,
		"embedding_multiplier": 1.0,
		"residual_multiplier": 1.0,
		"logits_scaling": 1.0,
		"rope_parameters": {
			"rope_theta": 10000,
			"rope_type": "yarn",
			"type": "yarn",
			"factor": 8,
			"original_max_position_embeddings": 32768
		},
		"rope_scaling": null
	}`)

	cfg, err := parseConfig(configJSON)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.RopeFreqs == nil {
		t.Fatal("RopeFreqs should not be nil when rope_type=\"yarn\"")
	}
	if cfg.RopeMScale == 1.0 {
		t.Error("RopeMScale should differ from 1.0 when rope_type=\"yarn\"")
	}
}

func TestParseConfigRopeScalingFallback(t *testing.T) {
	// rope_scaling should be used as a fallback when rope_parameters is nil.
	configJSON := []byte(`{
		"architectures": ["GraniteForCausalLM"],
		"hidden_size": 2560,
		"num_hidden_layers": 40,
		"intermediate_size": 8192,
		"num_attention_heads": 40,
		"num_key_value_heads": 8,
		"tie_word_embeddings": false,
		"vocab_size": 100352,
		"rms_norm_eps": 1e-05,
		"rope_theta": 10000,
		"max_position_embeddings": 131072,
		"attention_multiplier": 0.015625,
		"embedding_multiplier": 1.0,
		"residual_multiplier": 1.0,
		"logits_scaling": 1.0,
		"rope_parameters": null,
		"rope_scaling": {
			"type": "yarn",
			"factor": 8,
			"original_max_position_embeddings": 32768
		}
	}`)

	cfg, err := parseConfig(configJSON)
	if err != nil {
		t.Fatalf("parseConfig failed: %v", err)
	}

	if cfg.RopeFreqs == nil {
		t.Fatal("RopeFreqs should not be nil when rope_scaling has rope_type=\"yarn\"")
	}
}
