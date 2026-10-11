package pockettts

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
)

func TestPrepareText(t *testing.T) {
	for _, tc := range []struct {
		input, want string
		options     textOptions
	}{
		{"  hello\nworld  ", "Hello world.", textOptions{}},
		{"wait—", "Wait.", textOptions{}},
		{"\"Hello?\"", "\"Hello?\"", textOptions{}},
		{"hello", "        Hello.", textOptions{PadShortInputs: true}},
		{"hello; world", "Hello, world.", textOptions{RemoveSemicolons: true}},
		{"\"hi?\", she said", "Hi? she said.", textOptions{ReplaceCharacters: map[string]string{"\"": ""}}},
	} {
		got, _, err := tc.options.prepare(tc.input)
		if err != nil || got != tc.want {
			t.Errorf("prepare(%q) = %q, %v; want %q", tc.input, got, err, tc.want)
		}
	}
	if _, _, err := (textOptions{}).prepare(" \n\t"); err == nil {
		t.Fatal("accepted empty text")
	}
}

func TestUnigramMalformedVocabulary(t *testing.T) {
	for _, vocab := range []string{`[]`, `[[]]`, `[["x"]]`, `[["x","not a score"]]`, `[["",0]]`} {
		if _, err := newUnigram([]byte(`{"model":{"type":"Unigram","unk_id":0,"vocab":` + vocab + `}}`)); err == nil {
			t.Errorf("accepted malformed vocabulary %s", vocab)
		}
	}
}

func TestUnigramByteFallback(t *testing.T) {
	vocab := [][]any{{"<unk>", -10.0}, {metaspace, -1.0}, {"hello", -1.0}}
	for i := range 256 {
		vocab = append(vocab, []any{fmt.Sprintf("<0x%02X>", i), -20.0})
	}
	data, _ := json.Marshal(map[string]any{"model": map[string]any{"type": "Unigram", "unk_id": 0, "vocab": vocab}})
	u, err := newUnigram(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"hello", "hello café", "hello 👋", "hello  hello", "日本語"} {
		if got := u.decode(u.encode(text)); got != text {
			t.Errorf("round trip %q => %q", text, got)
		}
	}
}

// Run against a real checkpoint and fixtures generated with Hugging Face
// tokenizers: POCKET_TTS_TOKENIZER=... POCKET_TTS_TOKEN_CASES=... go test ...
func TestCheckpointTokenizer(t *testing.T) {
	path := os.Getenv("POCKET_TTS_TOKENIZER")
	if path == "" {
		t.Skip("set POCKET_TTS_TOKENIZER for checkpoint parity")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	u, err := newUnigram(data)
	if err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(os.Getenv("POCKET_TTS_TOKEN_CASES"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text string
		IDs  []int32
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		if got := u.encode(tc.Text); !slices.Equal(got, tc.IDs) {
			t.Errorf("%q: got %v; want %v", tc.Text, got, tc.IDs)
		}
	}
	chunks, err := u.chunks(strings.Repeat("The value is 3.14, and it matters. ", 20), textOptions{})
	if err != nil || len(chunks) < 2 {
		t.Fatalf("chunking: %v %v", chunks, err)
	}
	for _, chunk := range chunks {
		if !strings.Contains(chunk, "3.14") {
			t.Fatalf("split decimal: %q", chunk)
		}
	}
}
