package compatmigrate

import (
	"testing"

	"github.com/ollama/ollama/fs/gguf"
)

func TestGLMOCRModelKVEOTToken(t *testing.T) {
	tokens := []string{"<|endoftext|>", "a", "b", "<|user|>", "c"}
	tensors := []*outTensor{
		fixtureTensor("token_embd.weight", gguf.TensorTypeF16, []uint64{2, 8}),
	}

	t.Run("sets eot from <|user|>", func(t *testing.T) {
		src := fixtureSourceModel(t, outKV{
			"general.architecture":  "glmocr",
			"tokenizer.ggml.tokens": tokens,
		}, tensors)
		if got := glmOCRModelKV(src)["tokenizer.ggml.eot_token_id"]; got != uint32(3) {
			t.Fatalf("eot_token_id = %v, want 3", got)
		}
	})

	t.Run("keeps existing eot", func(t *testing.T) {
		src := fixtureSourceModel(t, outKV{
			"general.architecture":        "glmocr",
			"tokenizer.ggml.tokens":       tokens,
			"tokenizer.ggml.eot_token_id": uint32(0),
		}, tensors)
		if got := glmOCRModelKV(src)["tokenizer.ggml.eot_token_id"]; got != uint32(0) {
			t.Fatalf("eot_token_id = %v, want 0", got)
		}
	})

	t.Run("no <|user|> token", func(t *testing.T) {
		src := fixtureSourceModel(t, outKV{
			"general.architecture":  "glmocr",
			"tokenizer.ggml.tokens": tokens[:3],
		}, tensors)
		if got, ok := glmOCRModelKV(src)["tokenizer.ggml.eot_token_id"]; ok {
			t.Fatalf("eot_token_id = %v, want unset", got)
		}
	})
}
