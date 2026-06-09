package convert

import "testing"

func TestGLM4MoeLiteKVUsesLlamaCppMetadata(t *testing.T) {
	p := glm4MoeLiteModel{
		ModelParameters:       ModelParameters{VocabSize: 151552},
		MaxPositionEmbeddings: 202752,
		HiddenSize:            2048,
		HiddenLayers:          47,
		IntermediateSize:      10240,
		NumAttentionHeads:     20,
		NumKeyValueHeads:      20,
		RMSNormEPS:            1e-5,
		RopeTheta:             1000000,
		QKNopeHeadDim:         128,
		QKRopeHeadDim:         64,
		KVLoraRank:            512,
		QLoraRank:             768,
		VHeadDim:              128,
		ExpertCount:           64,
		ExpertSharedCount:     1,
		ExpertUsedCount:       4,
		ExpertWeightsNorm:     true,
		ExpertWeightsScale:    1.8,
	}

	kv := p.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2", Tokens: []string{"a"}}})

	if got := kv.Architecture(); got != "deepseek2" {
		t.Fatalf("architecture = %q, want deepseek2", got)
	}
	for key, want := range map[string]uint32{
		"attention.head_count":       20,
		"attention.head_count_kv":    1,
		"attention.key_length":       576,
		"attention.value_length":     512,
		"attention.key_length_mla":   192,
		"attention.value_length_mla": 128,
		"expert_group_count":         1,
		"expert_group_used_count":    1,
		"expert_gating_func":         2,
		"rope.dimension_count":       64,
	} {
		if got := kv.Uint(key); got != want {
			t.Errorf("%s = %d, want %d", key, got, want)
		}
	}
	if got := kv.String("tokenizer.ggml.pre"); got != "glm4" {
		t.Errorf("tokenizer.ggml.pre = %q, want glm4", got)
	}
}

func TestGLM4MoeLiteKVPromotesExtraEOSIDs(t *testing.T) {
	kv := KV{
		"general.architecture":         "deepseek2",
		"tokenizer.ggml.eos_token_ids": []int32{151329, 151330, 151336},
	}

	setGLM4MoeLiteExtraEOGFromEOSIDs(kv)

	if got := kv.Uint("tokenizer.ggml.eot_token_id"); got != 151330 {
		t.Errorf("eot token = %d, want 151330", got)
	}
	if got := kv.Uint("tokenizer.ggml.eom_token_id"); got != 151336 {
		t.Errorf("eom token = %d, want 151336", got)
	}
}
