package phonon

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
		tail        int
	}{
		{"  hello\nworld  ", "Hello world.", 3},
		{"wait—", "Wait—", 3},
		{"\"Hello?\"", "\"Hello?\"", 3},
		{"it's ready", "It's ready.", 3},
		{"it’s ready", "It’s ready.", 3},
		{"hello; world", "Hello; world.", 3},
		{"one  two\tthree\rfour five", "One two three four five.", 1},
		// Rust's full case mapping and Alphabetic property.
		{"ßtart with eszett", "SStart with eszett.", 3},
		{"ﬁnally, a ligature start", "FInally, a ligature start.", 3},
		{"rated ⓐ", "Rated ⓐ.", 3},
	} {
		got, tail, err := prepareText(tc.input)
		if err != nil || got != tc.want || tail != tc.tail {
			t.Errorf("prepare(%q) = %q, %d, %v; want %q, %d", tc.input, got, tail, err, tc.want, tc.tail)
		}
	}
	if _, _, err := prepareText(" \n\t"); err == nil {
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
	text := strings.Repeat("hello ", 300)
	prepared, _, _ := prepareText(text)
	chunks, err := u.chunks(text)
	if err != nil || len(chunks) < 2 || strings.Join(chunks, " ") != prepared {
		t.Fatalf("long sentence was not split losslessly: %v", err)
	}
	for _, chunk := range chunks {
		prepared, _, _ := prepareText(chunk)
		if n := len(u.encode(prepared)); n > 200 {
			t.Fatalf("long sentence chunk has %d tokens", n)
		}
	}
}

func TestUnigramFold(t *testing.T) {
	vocab := [][]any{{"<unk>", -10.0}}
	for _, c := range "abcehilnorstuvACEILORSøé" {
		vocab = append(vocab, []any{string(c), -1.0})
	}
	data, _ := json.Marshal(map[string]any{"model": map[string]any{"type": "Unigram", "unk_id": 0, "vocab": vocab}})
	u, err := newUnigram(data)
	if err != nil {
		t.Fatal(err)
	}
	got := u.fold("École ÉCOLE naïve Søren Æsir ÆSIR straße İstanbul bǎohù 日本")
	if want := "école ECOLE naive Søren Aesir AESIR strasse Istanbul baohu 日本"; got != want {
		t.Errorf("fold = %q; want %q", got, want)
	}
}

// Run against a real checkpoint and fixtures generated with Hugging Face
// tokenizers: PHONON_TOKENIZER=... PHONON_TOKEN_CASES=... go test ...
// encode folds letters the vocabulary has no piece for, which Hugging Face
// does not, so leave such letters out of the fixtures.
func TestCheckpointTokenizer(t *testing.T) {
	path := os.Getenv("PHONON_TOKENIZER")
	if path == "" {
		t.Skip("set PHONON_TOKENIZER for checkpoint parity")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	u, err := newUnigram(data)
	if err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(os.Getenv("PHONON_TOKEN_CASES"))
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
	chunks, err := u.chunks(strings.Repeat("Hello from the new Phonon model. ", 20))
	if err != nil || len(chunks) < 2 {
		t.Fatalf("chunking: %v %v", chunks, err)
	}
	for _, chunk := range chunks {
		if !strings.HasSuffix(chunk, ".") {
			t.Fatalf("split inside sentence: %q", chunk)
		}
	}
}

// Fixtures come from xn-ptts plan::chunks + plan::fit_or_error, using the same
// checkpoint tokenizer. They cover oversized sentences, clauses and UTF-8 words.
func TestCheckpointChunking(t *testing.T) {
	path := os.Getenv("PHONON_CHUNK_CASES")
	if path == "" {
		t.Skip("set PHONON_CHUNK_CASES for upstream chunking parity")
	}
	data, err := os.ReadFile(os.Getenv("PHONON_TOKENIZER"))
	if err != nil {
		t.Fatal(err)
	}
	u, err := newUnigram(data)
	if err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text   string
		Chunks []string
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		got, err := u.chunks(tc.Text)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, tc.Chunks) {
			t.Errorf("chunks for %.40q differ: got %d chunks, want %d", tc.Text, len(got), len(tc.Chunks))
		}
		for _, chunk := range got {
			prepared, _, err := prepareText(chunk)
			if err != nil {
				t.Fatal(err)
			}
			if n := len(u.encode(prepared)); n > 200 {
				t.Errorf("chunk has %d tokens", n)
			}
		}
	}
}

func TestUnigramDecodeSpecialTokens(t *testing.T) {
	u, err := newUnigram([]byte(`{
		"added_tokens":[{"id":1,"content":"\u0000<s>","special":true}],
		"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",-10],["\u0000<s>",0],["▁hello",-1],["▁world",-1]]}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := u.decode([]int32{1, 2, 1, 3, 1}); got != "hello world" {
		t.Fatalf("decode with skipped special tokens = %q", got)
	}
}
