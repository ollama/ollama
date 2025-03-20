package model

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func llama(t testing.TB) BytePairEncoding {
	t.Helper()

	f, err := os.Open(filepath.Join("testdata", "llama3.2", "encoder.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	vocab := make(map[string]int32)
	if err := json.NewDecoder(f).Decode(&vocab); err != nil {
		t.Fatal(err)
	}

	types := make([]uint32, len(vocab))
	tokens := make([]string, len(vocab))
	for token, id := range vocab {
		tokens[id] = token
		types[id] = 1
	}

	for _, token := range []string{"<|begin_of_text|>", "<|end_of_text|>"} {
		if _, ok := vocab[token]; !ok {
			tokens = append(tokens, token) //nolint:makezero
			types = append(types, 3)       //nolint:makezero
			vocab[token] = int32(len(vocab))
		}
	}

	f, err = os.Open(filepath.Join("testdata", "llama3.2", "vocab.bpe"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	merges := make([]string, 0, 50000)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		if !strings.HasPrefix(scanner.Text(), "#") {
			merges = append(merges, scanner.Text())
		}
	}

	return NewBytePairEncoding(
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		&Vocabulary{
			Values: tokens,
			Types:  types,
			Merges: merges,
		},
	)
}

func TestLlama(t *testing.T) {
	tokenizer := llama(t)

	t.Run("simple", func(t *testing.T) {
		t.Parallel()

		ids, err := tokenizer.Encode("hello world", true)
		if err != nil {
			t.Error(err)
		}

		if diff := cmp.Diff([]int32{15339, 1917}, ids); diff != "" {
			t.Errorf("no match (-theirs +ours):\n%s", diff)
		}

		s, err := tokenizer.Decode([]int32{15339, 1917})
		if err != nil {
			t.Fatal(err)
		}

		if s != "hello world" {
			t.Errorf("got %q, want hello world", s)
		}

		ids, err = tokenizer.Encode("hello <|end_of_text|>", true)
		if err != nil {
			t.Error(err)
		}

		if diff := cmp.Diff([]int32{15339, 220, 128001}, ids); diff != "" {
			t.Errorf("no match (-theirs +ours):\n%s", diff)
		}
	})

	t.Run("simple repeated", func(t *testing.T) {
		t.Parallel()

		cases := map[string][]int32{
			strings.Repeat("0", 1):  {15},
			strings.Repeat("0", 2):  {410},
			strings.Repeat("0", 3):  {931},
			strings.Repeat("0", 4):  {931, 15},
			strings.Repeat("0", 5):  {931, 410},
			strings.Repeat("0", 6):  {931, 931},
			strings.Repeat("0", 7):  {931, 931, 15},
			strings.Repeat("0", 8):  {931, 931, 410},
			strings.Repeat("0", 9):  {931, 931, 931},
			strings.Repeat("0", 10): {931, 931, 931, 15},
			strings.Repeat("0", 11): {931, 931, 931, 410},
			strings.Repeat("0", 12): {931, 931, 931, 931},
			strings.Repeat("0", 13): {931, 931, 931, 931, 15},
			strings.Repeat("0", 14): {931, 931, 931, 931, 410},
			strings.Repeat("0", 15): {931, 931, 931, 931, 931},
			strings.Repeat("0", 16): {931, 931, 931, 931, 931, 15},
			strings.Repeat("0", 17): {931, 931, 931, 931, 931, 410},
		}

		for s, want := range cases {
			ids, err := tokenizer.Encode(s, true)
			if err != nil {
				t.Error(err)
			}

			if diff := cmp.Diff(want, ids); diff != "" {
				t.Errorf("%q no match (-theirs +ours):\n%s", s, diff)
			}
		}
	})

	t.Run("basic roundtrip", func(t *testing.T) {
		t.Parallel()

		cases := []string{
			"hello",
			"hello ",
			"hello  ",
			" hello",
			" hello ",
			" hello  ",
			"hello world",
			"请考试我的软件！12345",
		}

		for _, want := range cases {
			ids, err := tokenizer.Encode(want, true)
			if err != nil {
				t.Error(err)
			}

			if got, err := tokenizer.Decode(ids); err != nil {
				t.Fatal(err)
			} else if got != want {
				t.Errorf("got %q, want %q", got, want)
			}
		}
	})

	t.Run("special", func(t *testing.T) {
		t.Parallel()

		cases := map[string][]int32{
			"<|begin_of_text|>A B!":                                               {128000, 32, 426, 0},
			"<|begin_of_text|>A<|end_of_text|>B!":                                 {128000, 32, 128001, 33, 0},
			"<|begin_of_text|>A<|end_of_text|>B<|begin_of_text|>!":                {128000, 32, 128001, 33, 128000, 0},
			"<|begin_of_text|>A<|end_of_text|>B<|begin_of_text|>!<|end_of_text|>": {128000, 32, 128001, 33, 128000, 0, 128001},
		}

		for s, want := range cases {
			ids, err := tokenizer.Encode(s, true)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(want, ids); diff != "" {
				t.Errorf("no match (-theirs +ours):\n%s", diff)
			}
		}
	})

	t.Run("split", func(t *testing.T) {
		t.Parallel()

		cases := map[string][]string{
			"Hello World!":                   {"Hello", " World", "!"},
			"I'm don't won't":                {"I", "'m", " don", "'t", " won", "'t"},
			"In 2024 there are 366 days":     {"In", " ", "202", "4", " there", " are", " ", "366", " days"},
			"Hello!! ...world":               {"Hello", "!!", " ...", "world"},
			"Hello    World":                 {"Hello", "   ", " World"},
			"Hello\nWorld":                   {"Hello", "\n", "World"},
			"Hello, WORLD!! How's it going?": {"Hello", ",", " WORLD", "!!", " How", "'s", " it", " going", "?"},
		}

		for s, want := range cases {
			got := slices.Collect(tokenizer.split(s))
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("no match (-theirs +ours):\n%s", diff)
			}
		}
	})
}

// tekken loads the Tekken tokenizer for testing
func tekken(t testing.TB) TextProcessor {
	t.Helper()

	// Load tokenizer config from mistral-small
	tokenizerConfigPath := filepath.Join("testdata", "mistral-small", "tokenizer_config.json")
	configFile, err := os.Open(tokenizerConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	defer configFile.Close()

	var config struct {
		AddBosToken bool   `json:"add_bos_token"`
		AddEosToken bool   `json:"add_eos_token"`
		BosToken    string `json:"bos_token"`
		EosToken    string `json:"eos_token"`
	}
	if err := json.NewDecoder(configFile).Decode(&config); err != nil {
		t.Fatal(err)
	}

	// Load tokenizer.json which contains the vocabulary and other settings
	tokenizerJsonPath := filepath.Join("testdata", "mistral-small", "tokenizer.json")
	tokenizerFile, err := os.Open(tokenizerJsonPath)
	if err != nil {
		t.Fatal(err)
	}
	defer tokenizerFile.Close()

	var tokenizerData struct {
		Model struct {
			Type   string           `json:"type"`
			Vocab  map[string]int32 `json:"vocab"`
			Merges []string         `json:"merges"`
		} `json:"model"`
		AddedTokens []struct {
			Id      int32  `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
		PreTokenizer struct {
			Type          string `json:"type"`
			Pretokenizers []struct {
				Type    string `json:"type"`
				Pattern struct {
					String string `json:"String"`
				} `json:"pattern"`
				Behavior string `json:"behavior"`
			} `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
	}
	if err := json.NewDecoder(tokenizerFile).Decode(&tokenizerData); err != nil {
		t.Fatal(err)
	}

	// Extract the pattern from pre_tokenizer if available
	var pattern string
	if tokenizerData.PreTokenizer.Type == "Sequence" && len(tokenizerData.PreTokenizer.Pretokenizers) > 0 {
		pattern = tokenizerData.PreTokenizer.Pretokenizers[0].Pattern.String
	}

	// Combine regular vocab and added tokens
	vocab := tokenizerData.Model.Vocab

	// Add special tokens from added_tokens
	for _, token := range tokenizerData.AddedTokens {
		vocab[token.Content] = token.Id
	}

	// Create vocabulary arrays
	maxId := int32(-1)
	for _, id := range vocab {
		if id > maxId {
			maxId = id
		}
	}

	vocabSize := int(maxId + 1)
	types := make([]uint32, vocabSize)
	tokens := make([]string, vocabSize)
	scores := make([]float32, vocabSize)

	for token, id := range vocab {
		tokens[id] = token
		types[id] = TOKEN_TYPE_NORMAL

		// Assign appropriate token types for special tokens
		if token == "<s>" {
			types[id] = TOKEN_TYPE_CONTROL
		} else if token == "</s>" {
			types[id] = TOKEN_TYPE_CONTROL
		} else if token == "[INST]" || token == "[/INST]" {
			types[id] = TOKEN_TYPE_CONTROL
		}
	}

	// In Tekken, we don't need to load merges separately as they're part of the model
	var merges []string

	// Create vocabulary object
	vocabObj := &Vocabulary{
		Values: tokens,
		Types:  types,
		Scores: scores,
		Merges: merges,
		BOS:    vocab[config.BosToken],
		EOS:    vocab[config.EosToken],
		AddBOS: config.AddBosToken,
		AddEOS: config.AddEosToken,
	}

	// Use pattern from tokenizer.json if available
	if pattern != "" {
		// Ensure pattern has proper escaping for Go regexp
		pattern = strings.ReplaceAll(pattern, "p{", "\\p{")
		return NewBytePairEncoding(pattern, vocabObj)
	}

	// Fallback pattern if not found
	return NewBytePairEncoding(
		`\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+`,
		vocabObj,
	)
}

func TestTekken(t *testing.T) {
	// Skip if the test data isn't available
	if _, err := os.Stat(filepath.Join("testdata", "mistral-small")); os.IsNotExist(err) {
		t.Skip("Mistral-small test data not available")
	}

	tokenizer := tekken(t)

	t.Run("whitespace_handling", func(t *testing.T) {
		t.Parallel()

		// The key difference from SentencePiece is that Tekken doesn't prepend whitespace
		cases := []struct {
			input    string
			expected string
		}{
			{" hello", " hello"},
			{"hello ", "hello "},
			{"hello world", "hello world"},
			{" hello world ", " hello world "},
		}

		for _, tc := range cases {
			ids, err := tokenizer.Encode(tc.input, false)
			if err != nil {
				t.Errorf("Failed to encode %q: %v", tc.input, err)
				continue
			}

			decoded, err := tokenizer.Decode(ids)
			if err != nil {
				t.Errorf("Failed to decode tokens for %q: %v", tc.input, err)
				continue
			}

			if decoded != tc.expected {
				t.Errorf("Whitespace handling: got %q, want %q", decoded, tc.expected)
			}
		}
	})

	t.Run("chat_templates", func(t *testing.T) {
		t.Parallel()

		// Test the Tekken chat template format which doesn't have spaces after special tokens
		templates := []struct {
			input       string
			expectSpace bool // whether we expect a space after special tokens
		}{
			{"<s>[INST]user message[/INST]", false},
			{"<s>[INST] user message[/INST]", true},
			{"<s>[INST]user message [/INST]", true},
		}

		for _, tc := range templates {
			ids, err := tokenizer.Encode(tc.input, false)
			if err != nil {
				t.Errorf("Failed to encode %q: %v", tc.input, err)
				continue
			}

			decoded, err := tokenizer.Decode(ids)
			if err != nil {
				t.Errorf("Failed to decode tokens for %q: %v", tc.input, err)
				continue
			}

			// Check if there's a space after special tokens
			hasSpaceAfterINST := strings.Contains(decoded, "[INST] ")

			if hasSpaceAfterINST != tc.expectSpace {
				t.Errorf("Chat template space handling: got space=%v, want space=%v for %q",
					hasSpaceAfterINST, tc.expectSpace, tc.input)
			}
		}
	})

	t.Run("special_tokens", func(t *testing.T) {
		t.Parallel()

		// Test how Tekken handles special tokens
		cases := []struct {
			input    string
			expected []string // We'll check if these tokens are in the decoded output
		}{
			{"<s>[INST]hello[/INST]", []string{"<s>", "[INST]", "hello", "[/INST]"}},
			{"[INST]hello[/INST]</s>", []string{"[INST]", "hello", "[/INST]", "</s>"}},
			{"<s>[INST]hello[/INST]</s>[INST]again[/INST]", []string{"<s>", "[INST]", "hello", "[/INST]", "</s>", "[INST]", "again", "[/INST]"}},
		}

		for _, tc := range cases {
			ids, err := tokenizer.Encode(tc.input, false)
			if err != nil {
				t.Errorf("Failed to encode %q: %v", tc.input, err)
				continue
			}

			decoded, err := tokenizer.Decode(ids)
			if err != nil {
				t.Errorf("Failed to decode tokens for %q: %v", tc.input, err)
				continue
			}

			for _, expected := range tc.expected {
				if !strings.Contains(decoded, expected) {
					t.Errorf("Special token handling: %q missing in decoded output %q", expected, decoded)
				}
			}
		}
	})

	t.Run("vocabulary_coverage", func(t *testing.T) {
		t.Parallel()

		// Tekken has a larger vocabulary, so test coverage of various token types
		samples := []string{
			"Hello world!",
			"This is a test of the Tekken tokenizer.",
			"It has a considerably larger vocabulary size.",
			"Special characters: !@#$%^&*()",
			"Numbers: 1234567890",
			"Multiple languages: こんにちは 你好 안녕하세요",
			"Code snippets: def function(): return True",
		}

		for _, sample := range samples {
			ids, err := tokenizer.Encode(sample, false)
			if err != nil {
				t.Errorf("Failed to encode %q: %v", sample, err)
				continue
			}

			decoded, err := tokenizer.Decode(ids)
			if err != nil {
				t.Errorf("Failed to decode tokens for %q: %v", sample, err)
				continue
			}

			if decoded != sample {
				t.Errorf("Vocabulary coverage: got %q, want %q", decoded, sample)
			}
		}
	})

	t.Run("splitting_behavior", func(t *testing.T) {
		t.Parallel()

		// Test the splitting behavior which might differ from SentencePiece
		cases := map[string][]string{
			"Hello World!": {"Hello", " World", "!"},
			"user message": {"user", " message"},
			"[INST]hello":  {"[INST]", "hello"},
			"hello[/INST]": {"hello", "[/INST]"},
		}

		for s, want := range cases {
			got := slices.Collect(tokenizer.(*BytePairEncoding).split(s))
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("Splitting behavior no match (-want +got):\n%s", diff)
			}
		}
	})

	t.Run("full_chat_sequence", func(t *testing.T) {
		t.Parallel()

		// Test a complete chat sequence with Tekken's format
		chatSequence := "<s>[INST]user message[/INST]assistant message</s>[INST]new user message[/INST]"

		ids, err := tokenizer.Encode(chatSequence, false)
		if err != nil {
			t.Fatalf("Failed to encode chat sequence: %v", err)
		}

		decoded, err := tokenizer.Decode(ids)
		if err != nil {
			t.Fatalf("Failed to decode chat sequence tokens: %v", err)
		}

		// In Tekken, the whitespace shouldn't be added after special tokens
		if strings.Contains(decoded, "[INST] ") {
			t.Errorf("Tekken chat sequence has unexpected space after [INST]: %q", decoded)
		}

		if strings.Contains(decoded, "[/INST] ") {
			t.Errorf("Tekken chat sequence has unexpected space after [/INST]: %q", decoded)
		}
	})
}

func BenchmarkBytePairEncoding(b *testing.B) {
	tokenizer := llama(b)
	bts, err := os.ReadFile(filepath.Join("testdata", "war-and-peace.txt"))
	if err != nil {
		b.Fatal(err)
	}

	for i := range 8 {
		n := min(int(math.Pow10(i)), len(bts))
		bts := bts[:n]
		b.Run("encode"+strconv.Itoa(n), func(b *testing.B) {
			b.ResetTimer()
			for range b.N {
				_, err := tokenizer.Encode(string(bts), true)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run("decode"+strconv.Itoa(n), func(b *testing.B) {
			ids, err := tokenizer.Encode(string(bts), true)
			if err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			for range b.N {
				_, err := tokenizer.Decode(ids)
				if err != nil {
					b.Fatal(err)
				}
			}
		})

		b.Run("split"+strconv.Itoa(n), func(b *testing.B) {
			b.ResetTimer()
			for range b.N {
				slices.Collect(tokenizer.split(string(bts)))
			}
		})
	}
}
