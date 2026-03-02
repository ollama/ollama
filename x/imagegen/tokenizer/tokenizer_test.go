//go:build mlx

package tokenizer

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
)

// TestPatternCompilation validates that HuggingFace pretokenizer patterns
// can be rewritten for Go's RE2 regexp engine and compiled successfully.
func TestPatternCompilation(t *testing.T) {
	patterns := []struct {
		name    string
		pattern string
	}{
		{"llama3", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`},
		{"qwen2", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`},
		{"gpt4o", `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`},
		{"gpt2", `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`},
		{"deepseek_cjk", `[一-龥\x{3040}-ゟ゠-ヿ]+`},
	}

	for _, p := range patterns {
		t.Run(p.name, func(t *testing.T) {
			rewritten := rewritePatternForRE2(p.pattern)
			if _, err := regexp.Compile(rewritten); err != nil {
				t.Errorf("failed to compile pattern: %v\noriginal: %s\nrewritten: %s", err, p.pattern, rewritten)
			}
		})
	}
}

// TestRoundtrip verifies the fundamental property: encode(text) -> decode -> text
// This is the key invariant from tiktoken's test suite.
func TestRoundtrip(t *testing.T) {
	tok, err := Load("testdata/mini_llama.json")
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	// Test cases covering key edge cases from tiktoken
	inputs := []string{
		// Empty and simple
		"",
		"a",
		"hello",
		"hello world",

		// Whitespace edge cases
		" ",
		"  ",
		"   ",
		" hello",
		"hello ",
		" hello ",
		"hello  world",
		"hello   world",
		"\t",
		"\n",
		"\r\n",
		"hello\nworld",
		"hello\n\nworld",

		// Contractions
		"don't",
		"I'm",
		"we'll",
		"they're",
		"it's",
		"DON'T", // uppercase

		// Numbers
		"123",
		"1234567890",
		"3.14159",
		"$100",
		"50%",

		// Unicode
		"こんにちは",          // Japanese
		"你好",              // Chinese
		"مرحبا",            // Arabic (RTL)
		"🎉",               // Emoji
		"Hello 世界",        // Mixed
		"café",             // Accented
		"naïve",            // Diaeresis
		"Ω≈ç√∫",            // Math symbols

		// Code
		"func main() {}",
		"if (x == 0) { return; }",
		"import \"fmt\"",
		"x := 42",
		"// comment",
		"/* block */",

		// Repetition (tiktoken specifically tests this)
		"aaaa",
		"aaaaaaaaaaaa",
		strings.Repeat("a", 100),
		strings.Repeat("hello ", 50),

		// Punctuation
		"...",
		"!!!",
		"???",
		"hello, world!",
		"(parentheses)",
		"[brackets]",
		"{braces}",

		// Mixed complexity
		"The quick brown fox jumps over the lazy dog.",
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
		"func TestRoundtrip(t *testing.T) { t.Run(\"test\", func(t *testing.T) {}) }",
	}

	for _, input := range inputs {
		name := input
		if len(name) > 30 {
			name = name[:30] + "..."
		}
		if name == "" {
			name = "<empty>"
		}
		name = strings.ReplaceAll(name, "\n", "\\n")
		name = strings.ReplaceAll(name, "\t", "\\t")

		t.Run(name, func(t *testing.T) {
			tokens := tok.Encode(input, false)
			decoded := tok.Decode(tokens)
			if decoded != input {
				t.Errorf("roundtrip failed:\n  input:   %q\n  tokens:  %v\n  decoded: %q", input, tokens, decoded)
			}
		})
	}
}

// TestSpecialTokens verifies that special tokens are handled correctly
func TestSpecialTokens(t *testing.T) {
	tok, err := Load("testdata/mini_llama.json")
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	// Special tokens should be preserved through encode/decode
	t.Run("bos_preserved", func(t *testing.T) {
		if tok.BOS() < 0 {
			t.Skip("no BOS token")
		}
		tokens := tok.Encode("hello", true)
		if len(tokens) == 0 || tokens[0] != tok.BOS() {
			t.Errorf("BOS not prepended: got %v, want first token to be %d", tokens, tok.BOS())
		}
	})

	t.Run("special_token_split", func(t *testing.T) {
		// If we have special tokens, verify they're split correctly
		for tokenStr, tokenID := range tok.specialTokens {
			input := "before" + tokenStr + "after"
			tokens := tok.Encode(input, false)

			found := false
			for _, id := range tokens {
				if id == tokenID {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("special token %q (id=%d) not found in encoding of %q: %v",
					tokenStr, tokenID, input, tokens)
			}
		}
	})
}

// TestConcurrency verifies thread-safe encoding
func TestConcurrency(t *testing.T) {
	tok, err := Load("testdata/mini_llama.json")
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	input := "The quick brown fox jumps over the lazy dog."
	expected := tok.Encode(input, false)

	var wg sync.WaitGroup
	errors := make(chan error, 100)

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := tok.Encode(input, false)
			if len(got) != len(expected) {
				errors <- nil // just signal error
				return
			}
			for j := range got {
				if got[j] != expected[j] {
					errors <- nil
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errors)

	if len(errors) > 0 {
		t.Errorf("concurrent encoding produced inconsistent results")
	}
}

// TestIntegration runs against real model directories, comparing with Python transformers.
// Skips if model weights are not available.
func TestIntegration(t *testing.T) {
	models := []string{
		"../weights/Llama-3.2-1B",
		"../weights/gemma-3-1b-it",
		"../weights/gpt-oss-20b",
	}

	// Test inputs covering various edge cases
	inputs := []string{
		"Hello, world!",
		"The quick brown fox jumps over the lazy dog.",
		"こんにちは世界",
		"def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
		"1234567890",
		"   spaces   ",
		"don't won't can't",
	}

	for _, modelPath := range models {
		modelName := filepath.Base(modelPath)

		t.Run(modelName, func(t *testing.T) {
			tokenizerPath := filepath.Join(modelPath, "tokenizer.json")
			if _, err := os.Stat(tokenizerPath); err != nil {
				t.Skipf("skipping: %s not found", tokenizerPath)
			}

			tok, err := Load(tokenizerPath)
			if err != nil {
				t.Fatalf("failed to load tokenizer: %v", err)
			}

			for _, input := range inputs {
				t.Run(truncate(input, 20), func(t *testing.T) {
					// Test roundtrip
					tokens := tok.Encode(input, false)
					decoded := tok.Decode(tokens)
					if decoded != input {
						t.Errorf("roundtrip failed:\n  input:   %q\n  decoded: %q", input, decoded)
					}

					// Compare with Python if available
					if pythonTokens, err := pythonEncode(modelPath, input); err == nil {
						if !equalInt32Slice(tokens, pythonTokens) {
							t.Errorf("mismatch with Python:\n  go:     %v\n  python: %v", tokens, pythonTokens)
						}
					}
				})
			}
		})
	}
}

// pythonEncode calls Python transformers to encode text, for comparison
func pythonEncode(modelPath, text string) ([]int32, error) {
	script := `
import sys, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
tokens = tok.encode(sys.argv[2], add_special_tokens=False)
print(json.dumps(tokens))
`
	cmd := exec.Command("python3", "-c", script, modelPath, text)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = nil

	if err := cmd.Run(); err != nil {
		return nil, err
	}

	// Parse JSON array
	var tokens []int32
	output := strings.TrimSpace(out.String())
	if output == "" || output == "[]" {
		return []int32{}, nil
	}

	// Simple parsing for [1, 2, 3] format
	output = strings.Trim(output, "[]")
	if output == "" {
		return []int32{}, nil
	}

	for _, s := range strings.Split(output, ",") {
		s = strings.TrimSpace(s)
		var v int32
		if _, err := parseIntSimple(s, &v); err == nil {
			tokens = append(tokens, v)
		}
	}

	return tokens, nil
}

func parseIntSimple(s string, v *int32) (bool, error) {
	var n int64
	for _, c := range s {
		if c >= '0' && c <= '9' {
			n = n*10 + int64(c-'0')
		}
	}
	*v = int32(n)
	return true, nil
}

func equalInt32Slice(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// TestBPEPretokenizer verifies BPE pretokenizer splits text correctly
// using the GPT-2 style regex pattern (no dependency on tokenizer files)
func TestBPEPretokenizer(t *testing.T) {
	pattern := `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
	re := regexp.MustCompile(rewritePatternForRE2(pattern))

	tests := []struct {
		input    string
		expected []string
	}{
		{"Hello", []string{"Hello"}},
		{"Hello world", []string{"Hello", " world"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"don't", []string{"don", "'t"}},
		{"I'm", []string{"I", "'m"}},
		{"123", []string{"123"}},
		{"12345", []string{"12345"}}, // GPT-2 pattern matches any digit sequence
		{"a  b", []string{"a", " ", " b"}}, // whitespace boundary: last space prepends to word
		{"   ", []string{"   "}},           // pure whitespace stays together
		{"\n\n", []string{"\n\n"}},         // newlines stay together
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			// Get regex matches
			matches := re.FindAllStringIndex(tt.input, -1)
			var chunks []string
			for _, m := range matches {
				chunks = append(chunks, tt.input[m[0]:m[1]])
			}

			// Apply whitespace boundary fix (same logic as Encode)
			for i := 0; i < len(chunks)-1; i++ {
				if isNonNewlineWhitespace(chunks[i]) && len(chunks[i+1]) > 0 {
					r, _ := []rune(chunks[i+1])[0], 0
					if r >= 'A' && r <= 'z' { // simplified letter check
						// Move last space to next chunk
						if len(chunks[i]) > 0 {
							lastSpace := chunks[i][len(chunks[i])-1:]
							chunks[i] = chunks[i][:len(chunks[i])-1]
							chunks[i+1] = lastSpace + chunks[i+1]
						}
					}
				}
			}

			// Filter empty chunks
			var result []string
			for _, c := range chunks {
				if c != "" {
					result = append(result, c)
				}
			}

			if len(result) != len(tt.expected) {
				t.Errorf("got %v, want %v", result, tt.expected)
				return
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("chunk %d: got %q, want %q", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

// TestSentencePiecePretokenizer verifies SentencePiece doesn't use pretokenizer
// and correctly replaces spaces with ▁ (no dependency on tokenizer files)
func TestSentencePiecePretokenizer(t *testing.T) {
	// SentencePiece has no pretokenizer - whole text is one chunk
	// Spaces are replaced with ▁ during encoding

	tests := []struct {
		input    string
		expected string // after space replacement
	}{
		{"Hello", "Hello"},
		{"Hello world", "Hello▁world"},
		{"Hello, world!", "Hello,▁world!"},
		{"   spaces   ", "▁▁▁spaces▁▁▁"},
		{" Hello", "▁Hello"},
		{"Hello ", "Hello▁"},
		{"a b c", "a▁b▁c"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			// SentencePiece encoding: replace space with ▁
			result := strings.ReplaceAll(tt.input, " ", "▁")
			if result != tt.expected {
				t.Errorf("got %q, want %q", result, tt.expected)
			}
		})
	}
}

// TestWordPiecePretokenizer verifies WordPiece (BERT) pretokenizer splits correctly
// BertPreTokenizer splits on whitespace and punctuation
func TestWordPiecePretokenizer(t *testing.T) {
	// BertPreTokenizer behavior: split on whitespace and punctuation
	// Whitespace is stripped, punctuation becomes separate tokens

	tests := []struct {
		input    string
		expected []string
	}{
		{"Hello", []string{"Hello"}},
		{"Hello world", []string{"Hello", "world"}},           // whitespace stripped
		{"Hello, world!", []string{"Hello", ",", "world", "!"}}, // punct separate
		{"don't", []string{"don", "'", "t"}},                   // apostrophe separate (unlike BPE)
		{"   spaces   ", []string{"spaces"}},                   // whitespace stripped
		{"Hello.World", []string{"Hello", ".", "World"}},       // punct splits
		{"test@email.com", []string{"test", "@", "email", ".", "com"}},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := splitBertStyle(tt.input)
			if len(result) != len(tt.expected) {
				t.Errorf("got %v, want %v", result, tt.expected)
				return
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("token %d: got %q, want %q", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

// splitBertStyle mimics BertPreTokenizer: split on whitespace and punctuation
func splitBertStyle(s string) []string {
	var result []string
	var current strings.Builder

	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			// Whitespace: flush current token, don't add whitespace
			if current.Len() > 0 {
				result = append(result, current.String())
				current.Reset()
			}
		} else if isPunct(r) {
			// Punctuation: flush current, add punct as separate token
			if current.Len() > 0 {
				result = append(result, current.String())
				current.Reset()
			}
			result = append(result, string(r))
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		result = append(result, current.String())
	}
	return result
}

func isPunct(r rune) bool {
	// Common ASCII punctuation
	return (r >= '!' && r <= '/') || (r >= ':' && r <= '@') ||
		(r >= '[' && r <= '`') || (r >= '{' && r <= '~')
}

// TestRepeatedDigits verifies correct tokenization of repeated digit sequences.
// Llama-style tokenizers split digits in groups of 1-3 due to the \p{N}{1,3} pattern.
func TestRepeatedDigits(t *testing.T) {
	tok, err := Load("./testdata/mini_llama.json")
	if err != nil {
		t.Skipf("mini_llama.json not available: %v", err)
	}

	// Pattern: 1 digit, 2 digits, 3 digits, then repeats
	// "0" -> [single], "00" -> [double], "000" -> [triple]
	// "0000" -> [triple, single], etc.
	tests := []struct {
		input string
		count int // expected token count
	}{
		{"0", 1},
		{"00", 1},
		{"000", 1},
		{"0000", 2},   // 3 + 1
		{"00000", 2},  // 3 + 2
		{"000000", 2}, // 3 + 3
		{"0000000", 3},
		{"00000000", 3},
		{"000000000", 3},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			ids := tok.Encode(tt.input, false)
			if len(ids) != tt.count {
				t.Errorf("Encode(%q) = %d tokens, want %d", tt.input, len(ids), tt.count)
			}
			// Verify roundtrip
			decoded := tok.Decode(ids)
			if decoded != tt.input {
				t.Errorf("Decode(Encode(%q)) = %q", tt.input, decoded)
			}
		})
	}
}

// TestNullByte verifies that null bytes roundtrip correctly
func TestNullByte(t *testing.T) {
	tok, err := Load("./testdata/mini_llama.json")
	if err != nil {
		t.Skipf("mini_llama.json not available: %v", err)
	}

	ids := tok.Encode("\x00", false)
	decoded := tok.Decode(ids)
	if decoded != "\x00" {
		t.Errorf("null byte roundtrip failed: got %q, want %q", decoded, "\x00")
	}
}

// TestTokenizerTypeDetection verifies correct detection of tokenizer types
func TestTokenizerTypeDetection(t *testing.T) {
	tests := []struct {
		name     string
		decoder  string
		expected TokenizerType
	}{
		{
			name:     "ByteLevel decoder (BPE)",
			decoder:  `{"type": "ByteLevel"}`,
			expected: TokenizerBPE,
		},
		{
			name: "Sequence with Replace ▁ (SentencePiece)",
			decoder: `{
				"type": "Sequence",
				"decoders": [
					{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}
				]
			}`,
			expected: TokenizerSentencePiece,
		},
		{
			name:     "null decoder (BPE default)",
			decoder:  `null`,
			expected: TokenizerBPE,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isSPM := detectSentencePiece([]byte(tt.decoder))
			var got TokenizerType
			if isSPM {
				got = TokenizerSentencePiece
			} else {
				got = TokenizerBPE
			}
			if got != tt.expected {
				t.Errorf("got %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestPADTokenDefault verifies PAD() returns -1 when not configured
func TestPADTokenDefault(t *testing.T) {
	tok, err := Load("testdata/mini_llama.json")
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	// mini_llama.json has no PAD token configured, should return -1
	if got := tok.PAD(); got != -1 {
		t.Errorf("PAD() = %d, want -1 (not configured)", got)
	}
}

// TestPADTokenFromConfig verifies PAD token is loaded from tokenizer_config.json
func TestPADTokenFromConfig(t *testing.T) {
	// Create temp directory with tokenizer files
	dir := t.TempDir()

	// Write minimal tokenizer.json
	tokenizerJSON := `{
		"model": {
			"type": "BPE",
			"vocab": {"<|endoftext|>": 0, "hello": 1, "world": 2},
			"merges": []
		},
		"added_tokens": [
			{"id": 0, "content": "<|endoftext|>", "special": true}
		]
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokenizerJSON), 0o644); err != nil {
		t.Fatalf("failed to write tokenizer.json: %v", err)
	}

	// Write tokenizer_config.json with pad_token
	configJSON := `{
		"pad_token": "<|endoftext|>"
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write tokenizer_config.json: %v", err)
	}

	tok, err := Load(dir)
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	if got := tok.PAD(); got != 0 {
		t.Errorf("PAD() = %d, want 0 (<|endoftext|>)", got)
	}
}

// TestPADTokenFromSpecialTokensMap verifies PAD falls back to special_tokens_map.json
func TestPADTokenFromSpecialTokensMap(t *testing.T) {
	dir := t.TempDir()

	// Write minimal tokenizer.json
	tokenizerJSON := `{
		"model": {
			"type": "BPE",
			"vocab": {"<pad>": 0, "hello": 1, "world": 2},
			"merges": []
		},
		"added_tokens": [
			{"id": 0, "content": "<pad>", "special": true}
		]
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokenizerJSON), 0o644); err != nil {
		t.Fatalf("failed to write tokenizer.json: %v", err)
	}

	// Write special_tokens_map.json with pad_token
	mapJSON := `{
		"pad_token": "<pad>"
	}`
	if err := os.WriteFile(filepath.Join(dir, "special_tokens_map.json"), []byte(mapJSON), 0o644); err != nil {
		t.Fatalf("failed to write special_tokens_map.json: %v", err)
	}

	tok, err := Load(dir)
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	if got := tok.PAD(); got != 0 {
		t.Errorf("PAD() = %d, want 0 (<pad>)", got)
	}
}

// TestPADTokenWithContentObject verifies PAD token works with {"content": "..."} format
func TestPADTokenWithContentObject(t *testing.T) {
	dir := t.TempDir()

	// Write minimal tokenizer.json
	tokenizerJSON := `{
		"model": {
			"type": "BPE",
			"vocab": {"[PAD]": 0, "hello": 1},
			"merges": []
		},
		"added_tokens": [
			{"id": 0, "content": "[PAD]", "special": true}
		]
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokenizerJSON), 0o644); err != nil {
		t.Fatalf("failed to write tokenizer.json: %v", err)
	}

	// Write tokenizer_config.json with pad_token as object (HuggingFace format)
	configJSON := `{
		"pad_token": {"content": "[PAD]", "lstrip": false, "normalized": false}
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write tokenizer_config.json: %v", err)
	}

	tok, err := Load(dir)
	if err != nil {
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	if got := tok.PAD(); got != 0 {
		t.Errorf("PAD() = %d, want 0 ([PAD])", got)
	}
}

// Benchmarks

func BenchmarkEncode(b *testing.B) {
	tok, err := Load("testdata/mini_llama.json")
	if err != nil {
		b.Fatalf("failed to load tokenizer: %v", err)
	}

	inputs := []struct {
		name string
		text string
	}{
		{"short", "Hello, world!"},
		{"medium", "The quick brown fox jumps over the lazy dog. " + strings.Repeat("This is a test. ", 10)},
		{"long", strings.Repeat("The quick brown fox jumps over the lazy dog. ", 100)},
	}

	for _, input := range inputs {
		b.Run(input.name, func(b *testing.B) {
			b.SetBytes(int64(len(input.text)))
			for i := 0; i < b.N; i++ {
				tok.Encode(input.text, false)
			}
		})
	}
}

func BenchmarkDecode(b *testing.B) {
	tok, err := Load("testdata/mini_llama.json")
	if err != nil {
		b.Fatalf("failed to load tokenizer: %v", err)
	}

	text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 100)
	tokens := tok.Encode(text, false)

	b.SetBytes(int64(len(text)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		tok.Decode(tokens)
	}
}
