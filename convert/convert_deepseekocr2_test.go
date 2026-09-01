package convert

import (
	"strings"
	"testing"
)

func TestDeepseekOCR2CreatesCorrectMetadataAndNames(t *testing.T) {
	m := &deepseekocr2{}
	m.LanguageConfig.MaxPositionEmbeddings = 8192
	m.LanguageConfig.HiddenSize = 1280
	m.LanguageConfig.HiddenLayers = 12
	m.LanguageConfig.IntermediateSize = 6848
	m.LanguageConfig.NumAttentionHeads = 10
	m.LanguageConfig.NumKeyValueHeads = 10
	m.LanguageConfig.NumRoutedExperts = 64
	m.LanguageConfig.NumSharedExperts = 2
	m.LanguageConfig.NumExpertsPerToken = 6
	m.LanguageConfig.FirstKDenseReplace = 1
	m.VisionConfig.Width.Qwen2.Dim = 896
	m.VisionConfig.Width.Sam.Layers = 12
	m.VisionConfig.Width.Sam.Width = 768
	m.VisionConfig.Width.Sam.Heads = 12

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2"}, Pre: "default"})

	for k, want := range map[string]any{
		"general.architecture":              "deepseekocr2",
		"block_count":                       uint32(12),
		"context_length":                    uint32(8192),
		"embedding_length":                  uint32(1280),
		"feed_forward_length":               uint32(6848),
		"attention.head_count":              uint32(10),
		"attention.head_count_kv":           uint32(10),
		"expert_count":                      uint32(64),
		"expert_used_count":                 uint32(6),
		"leading_dense_block_count":         uint32(1),
		"vision.block_count":                uint32(24),
		"qwen2.embedding_length":            uint32(896),
		"qwen2.block_count":                 uint32(24),
		"qwen2.head_count":                  uint32(14),
		"qwen2.head_count_kv":               uint32(2),
		"qwen2.intermediate_size":           uint32(4864),
		"sam.block_count":                   uint32(12),
		"sam.embedding_length":              uint32(768),
		"sam.head_count":                    uint32(12),
	} {
		if got := kv[k]; got != want {
			t.Fatalf("%s = %v (%T), want %v (%T)", k, got, got, want, want)
		}
	}

	replacer := strings.NewReplacer(m.Replacements()...)

	for name, want := range map[string]string{
		// Text model
		"model.embed_tokens.weight":                                    "token_embd.weight",
		"model.layers.0.input_layernorm.weight":                        "blk.0.attn_norm.weight",
		"model.layers.0.self_attn.q_proj.weight":                       "blk.0.attn_q.weight",
		"model.layers.0.self_attn.k_proj.weight":                       "blk.0.attn_k.weight",
		"model.layers.0.self_attn.v_proj.weight":                       "blk.0.attn_v.weight",
		"model.layers.0.self_attn.o_proj.weight":                       "blk.0.attn_output.weight",
		"model.layers.0.post_attention_layernorm.weight":               "blk.0.ffn_norm.weight",
		"model.layers.0.mlp.gate_proj.weight":                          "blk.0.ffn_gate.weight",
		"model.layers.0.mlp.up_proj.weight":                            "blk.0.ffn_up.weight",
		"model.layers.0.mlp.down_proj.weight":                          "blk.0.ffn_down.weight",
		"model.norm.weight":                                            "output_norm.weight",
		"lm_head.weight":                                               "output.weight",

		// Qwen2 vision encoder
		"model.qwen2_model.model.model.norm.weight":                    "q.output_norm.weight",
		"model.qwen2_model.model.model.layers.0.self_attn.q_proj.weight": "q.blk.0.attn_q.weight",
		"model.qwen2_model.query_768":                                  "q.query_768",
		"model.qwen2_model.query_1024":                                 "q.query_1024",

		// Projector
		"model.projector.weight":                                       "mm.weight",
		"model.view_seperator":                                         "mm.view_seperator",

		// SAM
		"model.sam_model.patch_embed.proj.weight":                      "s.patch_embd.weight",
		"model.sam_model.pos_embed":                                    "s.position_embd",
		"model.sam_model.blocks.0.attn.qkv.weight":                    "s.blk.0.attn.qkv.weight",
		"model.sam_model.neck.conv1.weight":                            "s.neck.conv1.weight",
	} {
		if got := replacer.Replace(name); got != want {
			t.Fatalf("Replace(%q) = %q, want %q", name, got, want)
		}
	}
}
