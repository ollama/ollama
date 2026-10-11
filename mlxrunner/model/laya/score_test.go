package laya

import (
	"context"
	"errors"
	"math"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func TestPrepareScoreInputLimits(t *testing.T) {
	// One token per character makes the field and context boundaries explicit.
	tok, err := tokenizer.LoadFromBytes([]byte(`{"model":{"type":"BPE","vocab":{"x":0,"Ġ":1,"c":2,"h":3,"o":4,"i":5,"e":6,"q":7,"u":8,"s":9,"t":10,"n":11,":":12},"merges":[]}}`))
	if err != nil {
		t.Fatal(err)
	}
	m := Model{config: config{MaxLen: 512, HeadMaxLen: 192}, tok: tok, cls: 100, sep: 101, mask: 102, maskToken: "[MASK]"}
	for _, tc := range []struct {
		name         string
		state        string
		instructions string
		options      []string
		limit        int
		wantTokens   int
		wantError    string
	}{
		{"context boundary", strings.Repeat("x", 36), "x", []string{"x", "x"}, 64, 64, ""},
		{"state overflow", strings.Repeat("x", 37), "x", []string{"x", "x"}, 64, 0, "state has 37 tokens; limit is 36"},
		{"question overflow", "", "x", []string{"x", "x"}, 27, 0, "decision question requires 28 tokens; context limit is 27"},
		{"option boundary", "x", "x", []string{strings.Repeat("x", 47), "x"}, 512, 75, ""},
		{"option overflow", "x", "x", []string{strings.Repeat("x", 48), "x"}, 512, 0, "option 0 has 49 tokens; limit is 48"},
		{"options and instructions fit", "x", "", []string{strings.Repeat("x", 42), strings.Repeat("x", 42), strings.Repeat("x", 42), strings.Repeat("x", 41)}, 512, 197, ""},
		{"options leave too little instruction room", "x", "", []string{strings.Repeat("x", 42), strings.Repeat("x", 42), strings.Repeat("x", 42), strings.Repeat("x", 42)}, 512, 0, "instructions with the question type have 17 tokens; limit is 16"},
		{"options budget overflow", "x", "", []string{strings.Repeat("x", 43), strings.Repeat("x", 42), strings.Repeat("x", 42), strings.Repeat("x", 42)}, 512, 0, "decision options exceed the 176-token budget"},
		{"instruction boundary", "x", strings.Repeat("x", 169), []string{"x", "x"}, 512, 197, ""},
		{"instruction overflow", "x", strings.Repeat("x", 170), []string{"x", "x"}, 512, 0, "instructions with the question type have 187 tokens; limit is 186"},
		{"reserved state", "x[MASK]x", "x", []string{"x", "x"}, 512, 0, "state contains reserved token"},
		{"reserved instructions", "x", "x[MASK]x", []string{"x", "x"}, 512, 0, "instructions contain reserved token"},
		{"reserved option", "x", "x", []string{"x", "x[MASK]x"}, 512, 0, "option 1 contains reserved token"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := llm.ScoreRequest{State: tc.state, MaxTokens: tc.limit, Rows: []llm.ScoreRow{{Question: &llm.ScoreQuestion{Type: "choice", Instructions: tc.instructions, Options: tc.options}}}}
			rows, err := m.PrepareScore(t.Context(), input)
			if tc.wantError != "" {
				var status api.StatusError
				if !errors.As(err, &status) || status.StatusCode != http.StatusBadRequest || !strings.Contains(status.ErrorMessage, "question 0: "+tc.wantError) || rows != nil {
					t.Fatalf("rows=%v error=%v, want HTTP 400 containing %q and no prepared rows", rows, err, tc.wantError)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(rows) != 1 || len(rows[0].Tokens) != tc.wantTokens || len(rows[0].Markers) != len(tc.options) {
				t.Fatalf("prepared rows=%+v, want one %d-token row with %d options", rows, tc.wantTokens, len(tc.options))
			}
			row := rows[0]
			if row.Tokens[0] != m.cls || row.Tokens[len(row.Tokens)-1] != m.sep {
				t.Fatalf("missing sequence delimiters: %v", row.Tokens)
			}
		})
	}

	// A late invalid question must fail before Score can forward even the first
	// valid row. This model has no weights, so any inference attempt also fails.
	input := llm.ScoreRequest{State: "x", MaxTokens: 512, Rows: []llm.ScoreRow{
		{Question: &llm.ScoreQuestion{Type: "choice", Options: []string{"x", "x"}}},
		{Question: &llm.ScoreQuestion{Type: "choice", Options: []string{"x", strings.Repeat("x", 48)}}},
	}}
	result, err := m.Score(t.Context(), input)
	var status api.StatusError
	if !errors.As(err, &status) || status.StatusCode != http.StatusBadRequest || !strings.Contains(status.ErrorMessage, "question 1: option 1") || len(result.Logits) != 0 || result.InputTokens != 0 {
		t.Fatalf("Score returned result=%+v error=%v, want rejection before inference", result, err)
	}

	// An ordinary request retains its complete input and marker positions.
	rows, err := m.PrepareScore(t.Context(), llm.ScoreRequest{State: "xx", MaxTokens: 512, Rows: input.Rows[:1]})
	if err != nil {
		t.Fatal(err)
	}
	want := []int32{100, 2, 3, 4, 5, 2, 6, 1, 7, 8, 6, 9, 10, 5, 4, 11, 12, 1, 101, 102, 1, 0, 102, 1, 0, 101, 0, 0, 101}
	if !slices.Equal(rows[0].Tokens, want) || !slices.Equal(rows[0].Markers, []int32{19, 22}) {
		t.Fatalf("prepared row=%+v, want tokens=%v markers=[19 22]", rows[0], want)
	}
}

func TestTemperature(t *testing.T) {
	m := Model{config: config{Temperature: []float32{2, 3, 4}, TemperatureByOptions: map[string]float32{"choice:2": 1.5, "choice:11+": 0.1, "score:3-5": 7, "noul:2": float32(math.NaN())}}}
	for _, tc := range []struct {
		kind  int32
		count int
		want  float32
	}{
		{0, 2, 1.5}, {0, 6, 2}, {0, 26, 0.5}, {1, 3, 5}, {2, 2, 1},
	} {
		if got := m.temperature(tc.kind, tc.count); got != tc.want {
			t.Errorf("type=%d count=%d: got %v want %v", tc.kind, tc.count, got, tc.want)
		}
	}
}

func TestScoreRejectsInvalidInputs(t *testing.T) {
	m := Model{config: config{MaxLen: 512}}
	for _, input := range []llm.ScoreRequest{
		{},
		{MaxTokens: 512, Rows: make([]llm.ScoreRow, 65)},
		{MaxTokens: 513, Rows: []llm.ScoreRow{{}}},
		{MaxTokens: 512, Rows: []llm.ScoreRow{{}}},
		{MaxTokens: 512, Rows: []llm.ScoreRow{{Question: &llm.ScoreQuestion{Type: "chat"}}}},
		{MaxTokens: 512, Rows: []llm.ScoreRow{{Question: &llm.ScoreQuestion{Type: "choice", Options: []string{"one"}}}}},
	} {
		_, err := m.Score(t.Context(), input)
		var status api.StatusError
		if !errors.As(err, &status) || status.StatusCode != http.StatusBadRequest {
			t.Errorf("Score(%+v) error = %v, want bad request", input, err)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := m.Score(ctx, llm.ScoreRequest{MaxTokens: 512, Rows: []llm.ScoreRow{{}}}); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled score: %v", err)
	}
}

func TestPrepareScoreCancellation(t *testing.T) {
	m := Model{config: config{MaxLen: 512}}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	rows, err := m.PrepareScore(ctx, llm.ScoreRequest{
		MaxTokens: 512,
		Rows:      []llm.ScoreRow{{Question: &llm.ScoreQuestion{Type: "choice", Options: []string{"yes", "no"}}}},
	})
	if !errors.Is(err, context.Canceled) || rows != nil {
		t.Fatalf("cancelled preparation: rows=%v, err=%v", rows, err)
	}
}
