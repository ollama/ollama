package clef

import (
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func TestEncodeTruncatesOnlyState(t *testing.T) {
	tok, err := tokenizer.LoadFromBytes([]byte(`{"model":{"type":"BPE","vocab":{"x":0},"merges":[]}}`))
	if err != nil {
		t.Fatal(err)
	}
	// Fixed prompt tokens must survive truncation along with the schema.
	prefix := []int32{100, 101, 101, 101, 102}
	input := llm.ScoreRequest{
		MaxTokens: 12,
		Segments:  []string{"prefix", strings.Repeat("x", 100), "xx", "x", "x"},
		Fields:    []llm.ScoreField{{Type: 0, Question: [2]int{2, 3}, Options: [][2]int{{3, 4}, {4, 5}}}},
	}
	got, err := encode(tok, input, prefix)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{100, 101, 101, 101, 102, 0, 0, 0, 0, 0, 0, 0}; !slices.Equal(got.ids, want) {
		t.Fatalf("tokens = %v, want %v", got.ids, want)
	}
	q := got.questions[0]
	if q.instruction != (tokenSpan{8, 10}) || !slices.Equal(q.options, []tokenSpan{{10, 11}, {11, 12}}) {
		t.Fatalf("schema spans shifted incorrectly: %+v", q)
	}
	input.MaxTokens = 8
	if _, err := encode(tok, input, prefix); err == nil {
		t.Fatal("prefix and schema exceeding the context must be rejected")
	}
}

func TestEncodeRejectsInvalidSpan(t *testing.T) {
	tok, err := tokenizer.LoadFromBytes([]byte(`{"model":{"type":"BPE","vocab":{"x":0},"merges":[]}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, bounds := range [][2]int{{-1, 2}, {0, 1}, {2, 2}, {2, 7}, {5, 6}} {
		input := llm.ScoreRequest{
			MaxTokens: 32,
			Segments:  []string{"prefix", "state", "x", "x", "x", ""},
			Fields:    []llm.ScoreField{{Question: bounds, Options: [][2]int{{3, 4}, {4, 5}}}},
		}
		if _, err := encode(tok, input, []int32{100}); err == nil {
			t.Errorf("accepted invalid or empty span %v", bounds)
		}
	}
}
