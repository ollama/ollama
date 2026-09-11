package convert

import "testing"

func TestLlama4SetsPreTokenizer(t *testing.T) {
	m := &llama4Model{}
	m.TextModel.HiddenSize = 5120
	m.TextModel.NumAttentionHeads = 40
	m.TextModel.NumKeyValueHeads = 8
	m.TextModel.IntermediateSizeMLP = 16384

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2"}, Pre: "default"})

	for k, want := range map[string]any{
		"general.architecture": "llama4",
		"tokenizer.ggml.pre":   "llama4",
	} {
		if got := kv[k]; got != want {
			t.Errorf("kv[%q] = %v, want %v", k, got, want)
		}
	}
}
