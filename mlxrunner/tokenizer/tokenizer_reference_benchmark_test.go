package tokenizer_test

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func BenchmarkTokenizerReference(b *testing.B) {
	for _, model := range []struct{ name, special string }{
		{"qwen3.5:0.8b-mxfp8", "<|endoftext|>"},
		{"nemotron-3.5-lightning:30b-a3b-nvfp4", "<unk>"},
		{"muse-glimmer:30b-nvfp4", "<|begin_of_text|>"},
	} {
		b.Run(model.name, func(b *testing.B) {
			tok, err := tokenizer.LoadFromBytes(loadTokenizerReference(b, model.name))
			if err != nil {
				b.Fatal(err)
			}
			if _, ok := tok.GetSpecialToken(model.special); !ok {
				b.Fatalf("missing special token %q", model.special)
			}
			for _, input := range []struct {
				name string
				text string
			}{
				{"short", "The quick brown fox jumps over the lazy dog. "},
				{"270KB", strings.Repeat("The quick brown fox jumps over the lazy dog. ", 6000)},
				{"special100", strings.Repeat("x"+model.special, 100)},
				{"special800", strings.Repeat("x"+model.special, 800)},
			} {
				b.Run(input.name, func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64(len(input.text)))
					b.ResetTimer()
					var ids []int32
					for range b.N {
						ids = tok.Encode(input.text, false)
					}
					b.StopTimer()
					if len(ids) == 0 {
						b.Fatal("encoding produced no tokens")
					}
				})
			}
		})
	}
}
