package mllama

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

type TextProcessor struct {
	model.BytePairEncoding
}

func newTextProcessor(c ml.Config) TextProcessor {
	return TextProcessor{
		BytePairEncoding: model.BytePairEncoding{
			Pretokenizer: c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			Vocabulary: &model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    c.Uint("tokenizer.ggml.bos_token_id"),
				EOS:    c.Uint("tokenizer.ggml.eos_token_id"),
			},
		},
	}
}
