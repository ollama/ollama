//go:build mlx

package tokenizer

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func llama32GGMLFixturePath(tb testing.TB, file string) string {
	tb.Helper()

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		tb.Fatal("failed to resolve test file path")
	}

	return filepath.Join(filepath.Dir(filename), "..", "..", "tokenizer", "testdata", "llama3.2", file)
}

func loadLlama32FromGGMLFixture(tb testing.TB) *Tokenizer {
	tb.Helper()

	f, err := os.Open(llama32GGMLFixturePath(tb, "encoder.json"))
	if err != nil {
		tb.Fatalf("failed to open encoder.json: %v", err)
	}
	defer f.Close()

	vocab := make(map[string]int32)
	if err := json.NewDecoder(f).Decode(&vocab); err != nil {
		tb.Fatalf("failed to decode encoder.json: %v", err)
	}

	type addedToken struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	}
	var addedTokens []addedToken
	for _, token := range []string{"<|begin_of_text|>", "<|end_of_text|>"} {
		if _, ok := vocab[token]; !ok {
			id := int32(len(vocab))
			vocab[token] = id
			addedTokens = append(addedTokens, addedToken{ID: id, Content: token, Special: true})
		}
	}

	mf, err := os.Open(llama32GGMLFixturePath(tb, "vocab.bpe"))
	if err != nil {
		tb.Fatalf("failed to open vocab.bpe: %v", err)
	}
	defer mf.Close()

	var merges []string
	scanner := bufio.NewScanner(mf)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimSpace(line)
		if line != "" {
			merges = append(merges, line)
		}
	}
	if err := scanner.Err(); err != nil {
		tb.Fatalf("failed to read vocab.bpe: %v", err)
	}

	payload := struct {
		Model struct {
			Type   string           `json:"type"`
			Vocab  map[string]int32 `json:"vocab"`
			Merges []string         `json:"merges"`
		} `json:"model"`
		PreTokenizer struct {
			Type          string `json:"type"`
			Pretokenizers []struct {
				Type    string `json:"type"`
				Pattern struct {
					Regex string `json:"Regex"`
				} `json:"pattern"`
			} `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
		AddedTokens []addedToken `json:"added_tokens"`
	}{}

	payload.Model.Type = "BPE"
	payload.Model.Vocab = vocab
	payload.Model.Merges = merges
	payload.PreTokenizer.Type = "Sequence"
	payload.PreTokenizer.Pretokenizers = []struct {
		Type    string `json:"type"`
		Pattern struct {
			Regex string `json:"Regex"`
		} `json:"pattern"`
	}{
		{
			Type: "Split",
			Pattern: struct {
				Regex string `json:"Regex"`
			}{
				Regex: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
			},
		},
	}
	payload.AddedTokens = addedTokens

	data, err := json.Marshal(payload)
	if err != nil {
		tb.Fatalf("failed to marshal synthetic tokenizer.json: %v", err)
	}

	tok, err := LoadFromBytes(data)
	if err != nil {
		tb.Fatalf("failed to load tokenizer from fixture data: %v", err)
	}
	return tok
}

func TestGGMLLlamaKnownEncodings(t *testing.T) {
	tok := loadLlama32FromGGMLFixture(t)

	cases := map[string][]int32{
		"hello world":                                          {15339, 1917},
		"hello <|end_of_text|>":                                {15339, 220, 128001},
		"<|begin_of_text|>A B!":                                {128000, 32, 426, 0},
		"<|begin_of_text|>A<|end_of_text|>B!":                  {128000, 32, 128001, 33, 0},
		"<|begin_of_text|>A<|end_of_text|>B<|begin_of_text|>!": {128000, 32, 128001, 33, 128000, 0},
		"<|begin_of_text|>A<|end_of_text|>B<|begin_of_text|>!<|end_of_text|>": {128000, 32, 128001, 33, 128000, 0, 128001},
	}

	for input, want := range cases {
		got := tok.Encode(input, false)
		if !equalIDs(got, want) {
			t.Fatalf("encode mismatch for %q:\n got:  %v\n want: %v", input, got, want)
		}
	}
}

func TestGGMLLlamaRepeatedZeros(t *testing.T) {
	tok := loadLlama32FromGGMLFixture(t)

	cases := map[int][]int32{
		1:  {15},
		2:  {410},
		3:  {931},
		4:  {931, 15},
		5:  {931, 410},
		6:  {931, 931},
		7:  {931, 931, 15},
		8:  {931, 931, 410},
		9:  {931, 931, 931},
		10: {931, 931, 931, 15},
		11: {931, 931, 931, 410},
		12: {931, 931, 931, 931},
		13: {931, 931, 931, 931, 15},
		14: {931, 931, 931, 931, 410},
		15: {931, 931, 931, 931, 931},
		16: {931, 931, 931, 931, 931, 15},
		17: {931, 931, 931, 931, 931, 410},
	}

	for n, want := range cases {
		input := strings.Repeat("0", n)
		got := tok.Encode(input, false)
		if !equalIDs(got, want) {
			t.Fatalf("encode mismatch for %q:\n got:  %v\n want: %v", input, got, want)
		}
	}
}

func TestGGMLLlamaRoundtripAndByteBehavior(t *testing.T) {
	tok := loadLlama32FromGGMLFixture(t)

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

	for _, input := range cases {
		ids := tok.Encode(input, false)
		got := tok.Decode(ids)
		if got != input {
			t.Fatalf("roundtrip mismatch for %q: got %q", input, got)
		}
	}

	// Match GGML tokenizer behavior: 0x00 is omitted when decoding.
	ids := tok.Encode(string(rune(0x00)), false)
	got := tok.Decode(ids)
	if got != "" {
		t.Fatalf("expected empty decode for 0x00, got %q (ids=%v)", got, ids)
	}
}
