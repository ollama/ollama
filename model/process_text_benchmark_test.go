package model

import (
	"testing"
)

// BenchmarkVocabulary is a reusable test vocabulary for benchmarks
var BenchmarkVocabulary = &Vocabulary{
	Values: []string{
		"Hello",
		"World",
		"!",
		"How",
		"are",
		"you",
		"t",
		"o",
		"d",
		"a",
		"y",
		"to",
		"tod",
		"toda",
		"today",
		" ",
		"<s>",
		"</s>",
		"<pad>",
		"'s",
		"'t",
		"'re",
		"'ve",
		"'m",
		"'ll",
		"'d",
	},
	Types: []uint32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1}, // 3 for special tokens
	Merges: []string{
		"to",
		"tod",
		"toda",
		"today",
	},
	BOS: 16, // <s>
	EOS: 17, // </s>
}

func BenchmarkBytePairEncoding(b *testing.B) {
	bpe := BytePairEncoding{
		Pretokenizer: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		Vocabulary:   BenchmarkVocabulary,
	}

	benchmarks := []struct {
		name  string
		input string
	}{
		{
			name:  "simple_hello_world",
			input: "Hello World!",
		},
		{
			name:  "with_special_tokens",
			input: "<s>Hello World!</s>",
		},
		{
			name:  "with_merges",
			input: "today is today and today",
		},
		{
			name:  "with_contractions",
			input: "I'm don't won't can't they're we've you'll he'd",
		},
		{
			name:  "long_text",
			input: "Hello World! How are you today? I'm doing great! This is a longer text to test the performance of the encoding and decoding process with multiple sentences and various tokens including special ones like <s> and </s> and contractions like don't and won't.",
		},
	}

	for _, bm := range benchmarks {
		// Benchmark Encoding
		b.Run("Encode_"+bm.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				tokens, err := bpe.Encode(bm.input)
				if err != nil {
					b.Fatal(err)
				}
				b.SetBytes(int64(len(tokens) * 4)) // Each token is 4 bytes (int32)
			}
		})

		// First encode the input to get tokens for decode benchmark
		tokens, err := bpe.Encode(bm.input)
		if err != nil {
			b.Fatal(err)
		}

		// Benchmark Decoding
		b.Run("Decode_"+bm.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				decoded, err := bpe.Decode(tokens)
				if err != nil {
					b.Fatal(err)
				}
				b.SetBytes(int64(len(decoded)))
			}
		})
	}
}

func BenchmarkBytePairEncodingSplit(b *testing.B) {
	bpe := BytePairEncoding{
		Pretokenizer: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
	}

	benchmarks := []struct {
		name  string
		input string
	}{
		{
			name:  "simple_text",
			input: "Hello World!",
		},
		{
			name:  "with_contractions",
			input: "I'm don't won't",
		},
		{
			name:  "with_numbers",
			input: "In 2024 there are 365 days",
		},
		{
			name:  "with_special_chars",
			input: "Hello!! ...world",
		},
		{
			name:  "with_spaces",
			input: "Hello    World",
		},
		{
			name:  "with_newlines",
			input: "Hello\nWorld\nHow\nAre\nYou",
		},
	}

	for _, bm := range benchmarks {
		b.Run("Split_"+bm.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				splits, err := bpe.split(bm.input)
				if err != nil {
					b.Fatal(err)
				}
				b.SetBytes(int64(len(splits)))
			}
		})
	}
}
