package qwen4_exp

import (
	"math"
	"testing"
)

const testConfig = `{
  "text_config": {
    "model_type": "qwen4_exp_text",
    "hidden_size": 2560,
    "num_hidden_layers": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "rms_norm_eps": 0.000001,
    "vocab_size": 248320,
    "eos_token_id": 248044,
    "max_position_embeddings": 262144,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "full_attention_interval": 4,
    "output_gate_type": "sigmoid",
    "attention_bias": false,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "mamba_ssm_dtype": "float32",
    "tie_word_embeddings": false,
    "use_cache": true,
    "linear_num_value_heads": 48,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "hc_count": 4,
    "hc_lowrank": 320,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "indexer_head_dim": 128,
    "indexer_kv_heads": 1,
    "indexer_n_heads": 4,
    "ple_conv_kernel_size": 4,
    "ple_embed_dim": 2560,
    "ple_layer_ids": [2],
    "heads_per_ngram": 8,
    "make_ngram_vocab_size_divisible_by": 128,
    "ngram_size": 3,
    "ngram_vocab_size_base": 20000000,
    "split_ngram_parts": 128,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "mtp": {
      "hybrid": true,
      "layer_types": ["full_attention"],
      "num_hidden_layers": 1,
      "rope_theta": 10000000
    },
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    }
  }
}`

func TestParseConfig(t *testing.T) {
	cfg, err := parseConfig([]byte(testConfig))
	if err != nil {
		t.Fatal(err)
	}

	if cfg.ModelType != "qwen4_exp_text" || cfg.HiddenSize != 2560 || cfg.HCCount != 4 {
		t.Fatalf("unexpected config identity: model_type=%q hidden=%d hc_count=%d", cfg.ModelType, cfg.HiddenSize, cfg.HCCount)
	}
	if !cfg.layerIsLinear(0) || cfg.layerIsLinear(3) {
		t.Fatalf("unexpected layer classification: %v", cfg.LayerTypes)
	}
	if cfg.RopeDim != 64 {
		t.Fatalf("rope dimension = %d, want 64", cfg.RopeDim)
	}
	wantScale := float32(1 / math.Sqrt(256))
	if cfg.Scale != wantScale {
		t.Fatalf("attention scale = %v, want %v", cfg.Scale, wantScale)
	}
}

func TestParseConfigRejectsUnknownSemantics(t *testing.T) {
	for _, replacement := range []struct {
		name string
		old  string
		new  string
	}{
		{name: "gate", old: `"output_gate_type": "sigmoid"`, new: `"output_gate_type": "swish"`},
		{name: "layer", old: `"full_attention"]`, new: `"sliding_attention"]`},
		{name: "mtp", old: "\"num_hidden_layers\": 1,\n      \"rope_theta\"", new: "\"num_hidden_layers\": 2,\n      \"rope_theta\""},
		{name: "attention head grouping", old: `"num_attention_heads": 24`, new: `"num_attention_heads": 23`},
		{name: "linear convolution", old: `"linear_conv_kernel_dim": 4`, new: `"linear_conv_kernel_dim": 0`},
		{name: "indexer budget", old: `"indexer_budget": 2048`, new: `"indexer_budget": 2`},
		{name: "indexer head broadcast", old: `"indexer_kv_heads": 1`, new: `"indexer_kv_heads": 2`},
		{name: "PLE layer range", old: `"ple_layer_ids": [2]`, new: `"ple_layer_ids": [0]`},
		{name: "duplicate PLE layer", old: `"ple_layer_ids": [2]`, new: `"ple_layer_ids": [2, 2]`},
		{name: "dedicated MTP embeddings", old: `"mtp_use_dedicated_embeddings": false`, new: `"mtp_use_dedicated_embeddings": true`},
		{name: "non-hybrid MTP", old: `"hybrid": true`, new: `"hybrid": false`},
		{name: "MTP hidden-state selection", old: `"hybrid": true,`, new: `"hybrid": true, "mtp_use_hidden_state_from_layer": 1,`},
		{name: "MTP rope", old: "\"rope_theta\": 10000000\n    },\n    \"partial_rotary_factor\"", new: "\"rope_theta\": 10000001\n    },\n    \"partial_rotary_factor\""},
		{name: "non-interleaved MRoPE", old: `"mrope_interleaved": true`, new: `"mrope_interleaved": false`},
		{name: "MRoPE sections", old: `"mrope_section": [11, 11, 10]`, new: `"mrope_section": [10, 11, 10]`},
		{name: "odd rotary dimension", old: "\"partial_rotary_factor\": 0.25,\n      \"rope_theta\"", new: "\"partial_rotary_factor\": 0.25390625,\n      \"rope_theta\""},
	} {
		t.Run(replacement.name, func(t *testing.T) {
			data := []byte(replaceOnce(testConfig, replacement.old, replacement.new))
			if _, err := parseConfig(data); err == nil {
				t.Fatal("parseConfig accepted unsupported semantics")
			}
		})
	}
}

func replaceOnce(s, old, new string) string {
	for i := 0; i+len(old) <= len(s); i++ {
		if s[i:i+len(old)] == old {
			return s[:i] + new + s[i+len(old):]
		}
	}
	return s
}
