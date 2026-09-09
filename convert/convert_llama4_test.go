package convert

import "testing"

func TestLlama4TokenizerPre(t *testing.T) {
	p := llama4Model{}
	p.ModelParameters.VocabSize = 200000
	p.TextModel.NumHiddenLayers = 4
	p.TextModel.HiddenSize = 2048
	p.TextModel.NumAttentionHeads = 16

	kv := p.KV(&Tokenizer{Vocabulary: &Vocabulary{Model: "gpt2"}})

	if got, want := kv["general.architecture"], "llama4"; got != want {
		t.Fatalf("general.architecture = %v, want %v", got, want)
	}
	if got, want := kv["tokenizer.ggml.pre"], "llama4"; got != want {
		t.Fatalf("tokenizer.ggml.pre = %v, want %v", got, want)
	}
}