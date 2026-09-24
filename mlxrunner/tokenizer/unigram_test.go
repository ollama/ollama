package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestUnigramBestPath(t *testing.T) {
	u := &Unigram{pieces: map[string]int32{"a": 0, "b": 1, "ab": 2, "c": 3}, scores: []float64{-1, -1, -3, -1}, maxPiece: 2, unknown: 4, unknownScore: -10}
	if got := u.segment("abc"); !slices.Equal(got, []int32{0, 1, 3}) {
		t.Fatal(got)
	}
	if got := u.segment("a😀🤖b"); !slices.Equal(got, []int32{0, 4, 1}) {
		t.Fatal(got)
	}
}

func TestPrecompiledNormalization(t *testing.T) {
	u := &Unigram{chars: make([]uint32, 256), replacements: "e\x00"}
	// A small Darts trie whose sole rule maps UTF-8 é to e.
	u.chars[0] = 1 << 10
	u.chars[194] = 0xc3 | 2<<10
	u.chars[105] = 0xa9 | 2<<10 | 256
	u.chars[107] = 0x80000000
	if got := u.normalize("  café!  "); got != "cafe!" {
		t.Fatal(got)
	}
	u.chars[0] = 0x7ffffc00 // corrupt offsets must not read past the table
	if got := u.normalize("é"); got != "é" {
		t.Fatal(got)
	}
}

func TestUnigramRejectsUnsupportedTokenizer(t *testing.T) {
	for _, data := range []string{`{}`, `{"model":{"type":"BPE"}}`, `{"model":{"type":"Unigram","byte_fallback":true}}`} {
		if _, err := LoadUnigram([]byte(data)); err == nil {
			t.Fatal("accepted", data)
		}
	}
}

func TestDebertaTokenizerParity(t *testing.T) {
	dir := os.Getenv("OLLAMA_GLINER_TEST_MODEL")
	if dir == "" {
		t.Skip("set OLLAMA_GLINER_TEST_MODEL to a GLiNER export with --reference")
	}
	data, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	u, err := LoadUnigram(data)
	if err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(filepath.Join(dir, "tokenizer_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Word string
		IDs  []int32
	}
	if err = json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		if got := u.EncodeWord(c.Word); !slices.Equal(got, c.IDs) {
			t.Errorf("%q: got %v, want %v", c.Word, got, c.IDs)
		}
	}
}
