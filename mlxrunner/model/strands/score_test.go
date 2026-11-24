package strands

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func TestPointerTokens(t *testing.T) {
	// A merged token may straddle the end of an option; it must be excluded.
	tok, err := tokenizer.LoadFromBytes([]byte(`{"model":{"type":"BPE","vocab":{"x":0,"a":1,"b":2,"Ċ":3,"ab":4,"abĊ":5,"Ã":6,"©":7},"merges":["a b","ab Ċ"]}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, prompt string
		spans        [][2]int
		pointers     []int32
	}{
		{"merged", "xab\nxab\nx", [][2]int{{0, 2}, {4, 6}}, []int32{1, 4}},
		{"unicode", "xé\nxé\nx", [][2]int{{0, 3}, {4, 7}}, []int32{3, 7}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := encode(tok, llm.ScorePointerRow{Prefix: "x", Prompt: tc.prompt, Type: 1, Options: tc.spans}, 32)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.Pointers, tc.pointers) {
				t.Fatalf("pointers = %v, want %v (tokens %v)", got.Pointers, tc.pointers, got.Tokens)
			}
		})
	}
	for _, spans := range [][][2]int{
		{{-1, 1}, {1, 2}}, {{1, 1}, {1, 2}}, {{0, 2}, {1, 3}}, {{0, 1}, {2, 99}}, {{1, 2}, {3, 4}},
	} {
		if _, err := encode(tok, llm.ScorePointerRow{Prompt: "xab\nx", Type: 1, Options: spans}, 32); err == nil {
			t.Fatalf("accepted invalid or tokenless spans %v", spans)
		}
	}
	if _, err := encode(tok, llm.ScorePointerRow{Prefix: "xxx", Prompt: "x\nx", Type: 1, Options: [][2]int{{0, 1}, {2, 3}}}, 5); err == nil {
		t.Fatal("accepted input over context")
	}
}

func TestPrepareScoreCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	m := &Model{Model: &qwen3_5.Model{Config: &qwen3_5.Config{MaxPositionEmbeddings: 4096}}, config: config{MaxLength: 4096}}
	_, err := m.PrepareScore(ctx, llm.ScoreRequest{MaxTokens: 4096, PointerRows: []llm.ScorePointerRow{{}}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want cancellation before tokenization", err)
	}
}
