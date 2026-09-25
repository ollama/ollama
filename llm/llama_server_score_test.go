package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func newLlamaScoreTestRunner(t *testing.T, completion http.HandlerFunc) *llamaServerRunner {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Content      string `json:"content"`
			AddSpecial   bool   `json:"add_special"`
			ParseSpecial *bool  `json:"parse_special"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
			return
		}
		if req.AddSpecial || req.ParseSpecial == nil {
			t.Error("scoring must explicitly parse special tokens without adding BOS/EOS")
			return
		}
		var tokens []int
		for _, c := range req.Content {
			tokens = append(tokens, int(c))
		}
		if *req.ParseSpecial && strings.HasSuffix(req.Content, "<special>") {
			tokens = append(tokens[:len(tokens)-len("<special>")], 1000)
		}
		// Model a tokenizer that merges the candidate with the prompt's last token.
		if strings.HasSuffix(req.Content, "~") && len(tokens) > 1 {
			tokens = append(tokens[:len(tokens)-2], 1001)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"tokens": tokens})
	})
	mux.HandleFunc("/completion", completion)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return &llamaServerRunner{
		port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client(),
		cmd: fakeRunningCmd(), options: api.Options{Runner: api.Runner{NumCtx: 8}},
		sem: semaphore.NewWeighted(1),
	}
}

func TestLlamaServerScore(t *testing.T) {
	var calls int
	runner := newLlamaScoreTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompt            []int    `json:"prompt"`
			Grammar           string   `json:"grammar"`
			Stream            bool     `json:"stream"`
			CachePrompt       bool     `json:"cache_prompt"`
			NPredict          int      `json:"n_predict"`
			NProbs            int      `json:"n_probs"`
			PostSamplingProbs *bool    `json:"post_sampling_probs"`
			Temperature       float64  `json:"temperature"`
			Samplers          []string `json:"samplers"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
			return
		}
		if req.Stream || !req.CachePrompt || req.NPredict != 1 || req.NProbs != 1 || req.PostSamplingProbs == nil || *req.PostSamplingProbs || req.Temperature != 1 || !slices.Equal(req.Samplers, []string{"temperature"}) {
			t.Errorf("request changes the candidate distribution: %+v", req)
		}
		var candidate string
		if err := json.Unmarshal([]byte(strings.TrimPrefix(req.Grammar, "root ::= ")), &candidate); err != nil {
			t.Error(err)
			return
		}
		want := []int{'a', 'b', 'c'}
		if candidate == "B" {
			want = []int{'x', 'y'}
		}
		if !slices.Equal(req.Prompt, want) {
			t.Errorf("rendered prompt changed: %v", req.Prompt)
		}
		logprob := -1.0
		if candidate == "Z" {
			logprob = -50
		}
		calls++
		_ = json.NewEncoder(w).Encode(map[string]any{
			"tokens_predicted": 1, "tokens_evaluated": len(req.Prompt),
			"completion_probabilities": []any{map[string]any{
				"id": int(candidate[0]), "token": candidate, "logprob": logprob,
				"top_logprobs": []any{map[string]any{"id": 0, "token": "unrelated", "logprob": -0.1}},
			}},
		})
	})
	// Model generation defaults must not affect scoring.
	runner.options.Temperature = 0
	runner.options.Stop = []string{"A"}
	runner.options.RepeatPenalty = 2
	result, err := runner.Score(t.Context(), ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{
		{Prompt: "abc", Candidates: []string{"Z", "A"}},
		{Prompt: "xy", Candidates: []string{"B"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Logits, [][]float32{{-50, -1}, {-1}}) || result.InputTokens != 5 || result.OutputTokens != 3 || calls != 3 {
		t.Fatalf("incomplete or reordered scores/usage: %+v, calls=%d", result, calls)
	}
}

func TestLlamaServerScoreValidation(t *testing.T) {
	for _, tt := range []struct {
		name  string
		input ScoreRequest
	}{
		{"empty rows", ScoreRequest{MaxTokens: 7}},
		{"too many rows", ScoreRequest{MaxTokens: 7, Rows: make([]ScoreRow, 65)}},
		{"zero max tokens", ScoreRequest{Rows: []ScoreRow{{Prompt: "a", Candidates: []string{"A"}}}}},
		{"max exceeds context", ScoreRequest{MaxTokens: 9, Rows: []ScoreRow{{Prompt: "a", Candidates: []string{"A"}}}}},
		{"empty prompt", ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Candidates: []string{"A"}}}}},
		{"oversize prompt", ScoreRequest{MaxTokens: 2, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"A"}}}}},
		{"no scoring token space", ScoreRequest{MaxTokens: 8, Rows: []ScoreRow{{Prompt: "abcdefgh", Candidates: []string{"A"}}}}},
		{"empty candidates", ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Prompt: "abc"}}}},
		{"too many candidates", ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Prompt: "abc", Candidates: make([]string, 27)}}}},
		{"multiple tokens", ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"AB"}}}}},
		{"special token", ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"<special>"}}}}},
		{"changed prefix", ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"~"}}}}},
		{"duplicate token", ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"A", "A"}}}}},
		{"invalid later row", ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"A"}}, {Prompt: "abc", Candidates: []string{"AB"}}}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			runner := newLlamaScoreTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
				t.Error("started inference before validating every row")
			})
			_, err := runner.Score(t.Context(), tt.input)
			var status api.StatusError
			if !errors.As(err, &status) || status.StatusCode != http.StatusBadRequest {
				t.Fatalf("expected validation error, got %v", err)
			}
		})
	}
}

func TestLlamaServerScoreResponse(t *testing.T) {
	const valid = `{"tokens_predicted":1,"tokens_evaluated":3,"completion_probabilities":[{"id":65,"token":"A","logprob":-2}]}`
	for _, tt := range []struct {
		name   string
		body   string
		status int
	}{
		{"missing logprob", strings.Replace(valid, `,"logprob":-2`, "", 1), 200},
		{"missing token id", strings.Replace(valid, `"id":65,`, "", 1), 200},
		{"wrong token id", strings.Replace(valid, `"id":65`, `"id":66`, 1), 200},
		{"wrong token text", strings.Replace(valid, `"token":"A"`, `"token":"B"`, 1), 200},
		{"overflow", strings.Replace(valid, `"logprob":-2`, `"logprob":-1e1000`, 1), 200},
		{"positive logprob", strings.Replace(valid, `"logprob":-2`, `"logprob":1`, 1), 200},
		{"incomplete prompt", strings.Replace(valid, `"tokens_evaluated":3`, `"tokens_evaluated":2`, 1), 200},
		{"truncated prompt", strings.Replace(valid, `"tokens_predicted":1`, `"truncated":true,"tokens_predicted":1`, 1), 200},
		{"extra generated token", strings.Replace(valid, `"tokens_predicted":1`, `"tokens_predicted":2`, 1), 200},
		{"empty probabilities", `{"tokens_predicted":1,"tokens_evaluated":3,"completion_probabilities":[]}`, 200},
		{"malformed JSON", `{`, 200},
		{"upstream error", `{"error":{"message":"out of memory"}}`, 500},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var failed atomic.Bool
			runner := newLlamaScoreTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
				if failed.CompareAndSwap(false, true) {
					w.WriteHeader(tt.status)
					fmt.Fprint(w, tt.body)
					return
				}
				fmt.Fprint(w, valid)
			})
			input := ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"A"}}}}
			if _, err := runner.Score(t.Context(), input); err == nil {
				t.Fatal("accepted invalid scoring response")
			} else if tt.status == 500 {
				var status api.StatusError
				if !errors.As(err, &status) || status.StatusCode != 500 || !IsOutOfMemory(err) {
					t.Fatalf("lost upstream status/OOM diagnostic: %v", err)
				}
			}
			ctx, cancel := context.WithTimeout(t.Context(), time.Second)
			defer cancel()
			if _, err := runner.Score(ctx, input); err != nil {
				t.Fatalf("runner unusable after failed scoring: %v", err)
			}
		})
	}
}

func TestLlamaServerScoreCancellation(t *testing.T) {
	for _, queued := range []bool{true, false} {
		t.Run(fmt.Sprintf("queued=%v", queued), func(t *testing.T) {
			started := make(chan struct{})
			var block atomic.Bool
			block.Store(!queued)
			runner := newLlamaScoreTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
				if block.CompareAndSwap(true, false) {
					_, _ = io.Copy(io.Discard, r.Body)
					close(started)
					<-r.Context().Done()
					return
				}
				fmt.Fprint(w, `{"tokens_predicted":1,"tokens_evaluated":3,"completion_probabilities":[{"id":65,"token":"A","logprob":0}]}`)
			})
			input := ScoreRequest{MaxTokens: 7, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"A"}}}}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if queued {
				if err := runner.sem.Acquire(t.Context(), 1); err != nil {
					t.Fatal(err)
				}
				cancel()
			} else {
				go func() {
					select {
					case <-started:
						cancel()
					case <-ctx.Done():
					}
				}()
			}
			_, err := runner.Score(ctx, input)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation lost: %v", err)
			}
			if queued {
				runner.sem.Release(1)
			}
			next, done := context.WithTimeout(t.Context(), time.Second)
			defer done()
			if _, err := runner.Score(next, input); err != nil {
				t.Fatalf("runner unusable after cancellation: %v", err)
			}
		})
	}
}
