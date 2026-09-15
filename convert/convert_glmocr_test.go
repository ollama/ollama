package convert

import "testing"

// GLM-OCR ends a turn with `<|user|>` as well as `<|endoftext|>`. llama.cpp only
// treats eos/eot/eom as end-of-generation, so `<|user|>` has to be written as the
// EOT token or generation runs past the end of the answer and repeats it.
func TestGlmOcrTokenizerKVMarksUserTokenAsEOT(t *testing.T) {
	tokenizer := &Tokenizer{Vocabulary: &Vocabulary{
		Model:  "gpt2",
		Tokens: []string{"a", "<|endoftext|>", "b", "<|user|>", "<|assistant|>"},
	}}

	kv := KV{}
	applyGlmOcrTokenizerKV(kv, tokenizer)

	if got := kv.String("tokenizer.ggml.pre"); got != "chatglm-bpe" {
		t.Errorf("tokenizer.ggml.pre = %q, want chatglm-bpe", got)
	}
	for key, want := range map[string]uint32{
		"tokenizer.ggml.eot_token_id":     3,
		"tokenizer.ggml.bos_token_id":     1,
		"tokenizer.ggml.unknown_token_id": 1,
	} {
		if got := kv.Uint(key); got != want {
			t.Errorf("%s = %d, want %d", key, got, want)
		}
	}
}
