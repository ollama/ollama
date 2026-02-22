//go:build mlx

package tokenizer

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var (
	benchmarkSinkIDs []int32
	benchmarkSinkStr string
	benchmarkSinkTok *Tokenizer
)

const benchmarkWordPieceJSON = `{
  "model": {
    "type": "WordPiece",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "##world": 2,
      "##ly": 3,
      "##hello": 4
    }
  },
  "added_tokens": []
}`

const benchmarkSentencePieceJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "\u2581": 0,
      "h": 1,
      "e": 2,
      "l": 3,
      "o": 4,
      "w": 5,
      "r": 6,
      "d": 7,
      "<0x0A>": 8
    },
    "merges": []
  },
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {
        "type": "Replace",
        "pattern": {
          "String": "\u2581"
        }
      }
    ]
  },
  "added_tokens": []
}`

func benchmarkMiniLlamaPath(tb testing.TB) string {
	tb.Helper()

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		tb.Fatal("failed to resolve benchmark file path")
	}

	return filepath.Join(filepath.Dir(filename), "..", "imagegen", "tokenizer", "testdata", "mini_llama.json")
}

func benchmarkLoadMiniLlama(tb testing.TB) *Tokenizer {
	tb.Helper()

	data := benchmarkLoadMiniLlamaBytes(tb)
	tok, err := LoadFromBytes(data)
	if err != nil {
		tb.Fatalf("failed to load mini llama tokenizer: %v", err)
	}
	return tok
}

func benchmarkLoadMiniLlamaBytes(tb testing.TB) []byte {
	tb.Helper()

	data, err := os.ReadFile(benchmarkMiniLlamaPath(tb))
	if err != nil {
		tb.Fatalf("failed to read mini llama tokenizer: %v", err)
	}
	return data
}

func benchmarkLoadFromBytes(tb testing.TB, data []byte) *Tokenizer {
	tb.Helper()

	tok, err := LoadFromBytes(data)
	if err != nil {
		tb.Fatalf("failed to load tokenizer from bytes: %v", err)
	}
	return tok
}

func BenchmarkTokenizerEncodeBPE(b *testing.B) {
	tok := benchmarkLoadMiniLlama(b)

	inputs := []struct {
		name string
		text string
	}{
		{name: "short", text: "Hello, world!"},
		{name: "medium", text: strings.Repeat("The quick brown fox jumps over the lazy dog. ", 16)},
		{name: "long_sequential", text: strings.Repeat("The quick brown fox jumps over the lazy dog. ", 80)},
		{name: "long_parallel", text: strings.Repeat("The quick brown fox jumps over the lazy dog. ", 160)},
		{name: "huge_parallel", text: strings.Repeat("The quick brown fox jumps over the lazy dog. ", 640)},
		{name: "special_tokens", text: "<|begin_of_text|>system\nYou are concise.<|end_of_text|>"},
	}

	for _, input := range inputs {
		b.Run(input.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(input.text)))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				benchmarkSinkIDs = tok.Encode(input.text, false)
			}
		})
	}
}

func BenchmarkTokenizerDecodeBPE(b *testing.B) {
	tok := benchmarkLoadMiniLlama(b)

	inputs := []struct {
		name string
		text string
	}{
		{name: "medium", text: strings.Repeat("The quick brown fox jumps over the lazy dog. ", 16)},
		{name: "long", text: strings.Repeat("The quick brown fox jumps over the lazy dog. ", 160)},
	}

	for _, input := range inputs {
		ids := tok.Encode(input.text, false)
		b.Run(input.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(input.text)))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				benchmarkSinkStr = tok.Decode(ids)
			}
		})
	}
}

func BenchmarkTokenizerLoadFromBytes(b *testing.B) {
	data := benchmarkLoadMiniLlamaBytes(b)

	config := &TokenizerConfig{
		TokenizerConfigJSON: []byte(`{
			"bos_token": {"content": "<|begin_of_text|>"},
			"eos_token": {"content": "<|end_of_text|>"},
			"add_bos_token": true
		}`),
		GenerationConfigJSON: []byte(`{"bos_token_id": 128000, "eos_token_id": 128001}`),
	}

	b.Run("without_config", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(data)))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			tok, err := LoadFromBytes(data)
			if err != nil {
				b.Fatalf("LoadFromBytes failed: %v", err)
			}
			benchmarkSinkTok = tok
		}
	})

	b.Run("with_config", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(data)))
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			tok, err := LoadFromBytesWithConfig(data, config)
			if err != nil {
				b.Fatalf("LoadFromBytesWithConfig failed: %v", err)
			}
			benchmarkSinkTok = tok
		}
	})
}

func BenchmarkTokenizerEncodeWordPiece(b *testing.B) {
	tok := benchmarkLoadFromBytes(b, []byte(benchmarkWordPieceJSON))
	text := strings.Repeat("helloworldly", 16)

	b.ReportAllocs()
	b.SetBytes(int64(len(text)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchmarkSinkIDs = tok.Encode(text, false)
	}
}

func BenchmarkTokenizerDecodeWordPiece(b *testing.B) {
	tok := benchmarkLoadFromBytes(b, []byte(benchmarkWordPieceJSON))
	text := strings.Repeat("helloworldly", 16)
	ids := tok.Encode(text, false)

	b.ReportAllocs()
	b.SetBytes(int64(len(text)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchmarkSinkStr = tok.Decode(ids)
	}
}

func BenchmarkTokenizerEncodeSentencePiece(b *testing.B) {
	tok := benchmarkLoadFromBytes(b, []byte(benchmarkSentencePieceJSON))
	text := strings.Repeat("hello world\n", 64)

	b.ReportAllocs()
	b.SetBytes(int64(len(text)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchmarkSinkIDs = tok.Encode(text, false)
	}
}

func BenchmarkTokenizerDecodeSentencePiece(b *testing.B) {
	tok := benchmarkLoadFromBytes(b, []byte(benchmarkSentencePieceJSON))
	text := strings.Repeat("hello world\n", 64)
	ids := tok.Encode(text, false)

	b.ReportAllocs()
	b.SetBytes(int64(len(text)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchmarkSinkStr = tok.Decode(ids)
	}
}
