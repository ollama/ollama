package convert

import (
	"strings"
	"testing"
)

func TestGptOssCreatesLlamaCppMetadataAndNames(t *testing.T) {
	m := &gptossModel{
		HiddenLayers:          24,
		MaxPositionEmbeddings: 131072,
		HiddenSize:            2880,
		IntermediateSize:      2880,
		AttentionHeads:        64,
		KeyValueHeads:         8,
		HeadDim:               64,
		LocalExperts:          32,
		ExpertsPerToken:       4,
		RopeTheta:             150000,
		InitialContextLength:  4096,
		SlidingWindow:         128,
	}
	m.RopeScaling.Type = "yarn"
	m.RopeScaling.Factor = 32
	m.RopeScaling.OriginalMaxPositionEmbeddings = 4096
	m.RopeScaling.BetaFast = 32
	m.RopeScaling.BetaSlow = 1

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2"}, Pre: "default"})
	for k, want := range map[string]any{
		"general.architecture":                         "gpt-oss",
		"tokenizer.ggml.pre":                           "gpt-4o",
		"gpt-oss.context_length":                       uint32(131072),
		"gpt-oss.expert_feed_forward_length":           uint32(2880),
		"gpt-oss.rope.scaling.type":                    "yarn",
		"gpt-oss.rope.scaling.factor":                  float32(32),
		"gpt-oss.rope.scaling.original_context_length": uint32(4096),
		"gpt-oss.rope.scaling.yarn_beta_fast":          float32(32),
		"gpt-oss.rope.scaling.yarn_beta_slow":          float32(1),
	} {
		if got := kv[k]; got != want {
			t.Fatalf("%s = %v (%T), want %v (%T)", k, got, got, want, want)
		}
	}
	if _, ok := kv["gptoss.context_length"]; ok {
		t.Fatal("unexpected Ollama-format gptoss metadata")
	}

	replacer := strings.NewReplacer(m.Replacements()...)
	for name, want := range map[string]string{
		"model.layers.0.self_attn.o_proj.weight":         "blk.0.attn_output.weight",
		"model.layers.0.self_attn.sinks":                 "blk.0.attn_sinks.weight",
		"model.layers.0.post_attention_layernorm.weight": "blk.0.post_attention_norm.weight",
		"model.layers.0.mlp.experts.gate_up_proj_blocks": "blk.0.ffn_gate_up_exps.blocks",
		"model.layers.0.mlp.experts.down_proj_scales":    "blk.0.ffn_down_exps.scales",
	} {
		if got := replacer.Replace(name); got != want {
			t.Fatalf("Replace(%q) = %q, want %q", name, got, want)
		}
	}

	m.MaxPositionEmbeddings = 0
	replacer = strings.NewReplacer(m.Replacements()...)
	for name, want := range map[string]string{
		"block.0.attn.out.weight": "blk.0.attn_output.weight",
		"block.0.attn.sinks":      "blk.0.attn_sinks.weight",
		"block.0.mlp.norm.weight": "blk.0.post_attention_norm.weight",
	} {
		if got := replacer.Replace(name); got != want {
			t.Fatalf("Replace(%q) = %q, want %q", name, got, want)
		}
	}
}
