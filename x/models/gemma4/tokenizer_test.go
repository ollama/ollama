package gemma4

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/tokenizer"
)

// TestMLXTokenizerMatchesReference verifies the MLX tokenizer (x/tokenizer)
// matches reference token IDs from the Rust tokenizers library for Gemma 4.
//
// Set GEMMA4_TOKENIZER_DIR to a model directory containing tokenizer.json.
// The test skips when neither the env var nor the default
// "models/gemma-4-e2b-it" path is present.
func TestMLXTokenizerMatchesReference(t *testing.T) {
	modelDir := os.Getenv("GEMMA4_TOKENIZER_DIR")
	if modelDir == "" {
		modelDir = filepath.Join("models", "gemma-4-e2b-it")
	}

	tokJSON := filepath.Join(modelDir, "tokenizer.json")
	data, err := os.ReadFile(tokJSON)
	if err != nil {
		t.Skipf("skipping: cannot read %s: %v", tokJSON, err)
	}

	tok, err := tokenizer.LoadFromBytes(data)
	if err != nil {
		t.Fatalf("LoadFromBytes failed: %v", err)
	}

	// Reference token IDs from the Rust tokenizers library (add_special_tokens=False).
	// Copied from model/models/gemma4/tokenizer_reference_test.go.
	tests := []struct {
		name  string
		input string
		want  []int32
	}{
		// Basic ASCII
		{name: "basic word", input: "hello", want: []int32{23391}},
		{name: "two words", input: "hello world", want: []int32{23391, 1902}},
		{name: "punctuation", input: "Hello, World!", want: []int32{9259, 236764, 4109, 236888}},

		// Space handling
		{name: "leading space", input: " hello", want: []int32{29104}},
		{name: "double leading space", input: "  hello", want: []int32{138, 23391}},
		{name: "double space between words", input: "hello  world", want: []int32{23391, 138, 12392}},
		{name: "only spaces", input: "   ", want: []int32{139}},
		{name: "leading spaces phrase", input: " leading spaces", want: []int32{5830, 9952}},
		{name: "multiple interior spaces", input: "multiple    spaces", want: []int32{43819, 140, 35220}},

		// Polish diacritics (byte fallback)
		{name: "polish diacritics", input: "ąęśćżźółń", want: []int32{237198, 237202, 14732, 237277, 238992, 24875, 238041}},
		{name: "polish sentence", input: "Zażółć gęślą jaźń", want: []int32{236953, 40512, 24875, 237289, 549, 237202, 62081, 237198, 4828, 238992, 238041}},

		// French accents
		{name: "french accents", input: "café résumé naïve", want: []int32{123125, 236859, 118515, 120362}},

		// CJK & Japanese
		{name: "chinese", input: "你好世界", want: []int32{144626, 12811}},
		{name: "japanese hiragana", input: "こんにちは", want: []int32{85141}},

		// Special tokens
		{
			name: "special_tokens", input: "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n",
			want: []int32{105, 2364, 107, 3689, 563, 236743, 236778, 236862, 236778, 236881, 106, 107, 105, 4368, 107},
		},
		{
			name: "tool_declaration", input: "<|tool>declaration:bash{description:<|\"|>Run a command<|\"|>}<tool|>",
			want: []int32{46, 163688, 236787, 42422, 236782, 7777, 236787, 52, 7306, 496, 4991, 52, 236783, 47},
		},
		{
			name: "tool_call", input: "<|tool_call>call:bash{command:<|\"|>ls -la<|\"|>}<tool_call|>",
			want: []int32{48, 6639, 236787, 42422, 236782, 7674, 236787, 52, 5629, 753, 2149, 52, 236783, 49},
		},

		// Code
		{name: "python code", input: "def foo(x): return x + 1", want: []int32{2063, 46293, 236769, 236781, 1473, 994, 1123, 900, 236743, 236770}},
		{name: "json", input: `{"key": "value"}`, want: []int32{14937, 2478, 1083, 623, 2394, 25938}},

		// Misc
		{name: "emoji", input: "hello 👋 world", want: []int32{23391, 155818, 1902}},
		{name: "digits", input: "12345", want: []int32{236770, 236778, 236800, 236812, 236810}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Encode(tt.input, false) // no BOS

			if len(got) != len(tt.want) {
				t.Errorf("token count mismatch: got %d, want %d", len(got), len(tt.want))
				t.Logf("got:  %v", got)
				t.Logf("want: %v", tt.want)
				return
			}

			mismatches := 0
			for i := range got {
				if got[i] != tt.want[i] {
					mismatches++
					if mismatches <= 5 {
						t.Errorf("mismatch at [%d]: got %d, want %d", i, got[i], tt.want[i])
					}
				}
			}
			if mismatches > 5 {
				t.Errorf("... and %d more mismatches", mismatches-5)
			}
		})
	}
}
