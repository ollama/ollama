package laya

import (
	"context"
	"errors"
	"math"
	"net/http"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

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
