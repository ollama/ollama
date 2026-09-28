package llm

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
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

type scoreCompletionTestRequest struct {
	Prompt            []int        `json:"prompt"`
	Grammar           string       `json:"grammar"`
	Stream            bool         `json:"stream"`
	CachePrompt       bool         `json:"cache_prompt"`
	NPredict          int          `json:"n_predict"`
	NProbs            int          `json:"n_probs"`
	PostSamplingProbs *bool        `json:"post_sampling_probs"`
	Temperature       float64      `json:"temperature"`
	Samplers          []string     `json:"samplers"`
	TopK              int          `json:"top_k"`
	LogitBias         [][2]float64 `json:"logit_bias"`
}

// scoreCompletionTestResponse returns the candidates' probabilities in the
// order llama-server sorts them.
func scoreCompletionTestResponse(prompt int, probs map[int]float64) map[string]any {
	ids := slices.Collect(maps.Keys(probs))
	slices.SortFunc(ids, func(a, b int) int { return cmp.Compare(probs[b], probs[a]) })
	var top []any
	for _, id := range ids {
		top = append(top, map[string]any{"id": id, "token": string(rune(id)), "prob": probs[id]})
	}
	return map[string]any{
		"tokens_predicted": 1, "tokens_evaluated": prompt,
		"completion_probabilities": []any{map[string]any{"id": ids[0], "token": string(rune(ids[0])), "prob": probs[ids[0]], "top_probs": top}},
	}
}

func checkScoreRowRequest(t *testing.T, req scoreCompletionTestRequest, candidates []int, bias float64) {
	t.Helper()
	var biased []int
	for _, b := range req.LogitBias {
		if b[1] != bias {
			t.Errorf("candidates must share bias %g: %v", bias, req.LogitBias)
		}
		biased = append(biased, int(b[0]))
	}
	if req.Stream || !req.CachePrompt || req.NPredict != 1 || req.Grammar != "" || req.NProbs != len(candidates) || req.PostSamplingProbs == nil || !*req.PostSamplingProbs ||
		req.Temperature != 1 || !slices.Equal(req.Samplers, []string{"top_k", "temperature"}) || req.TopK != len(candidates) || !slices.Equal(biased, candidates) {
		t.Errorf("request changes the candidate distribution: %+v", req)
	}
}

func TestLlamaServerScore(t *testing.T) {
	var calls int
	runner := newLlamaScoreTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
		var req scoreCompletionTestRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
			return
		}
		calls++
		switch {
		case slices.Equal(req.Prompt, []int{'a', 'b', 'c'}):
			checkScoreRowRequest(t, req, []int{'Z', 'A'}, scoreBias)
			_ = json.NewEncoder(w).Encode(scoreCompletionTestResponse(len(req.Prompt), map[int]float64{'Z': 0.25, 'A': 0.75}))
		case slices.Equal(req.Prompt, []int{'x', 'y'}):
			checkScoreRowRequest(t, req, []int{'B'}, scoreBias)
			_ = json.NewEncoder(w).Encode(scoreCompletionTestResponse(len(req.Prompt), map[int]float64{'B': 1}))
		default:
			t.Errorf("rendered prompt changed: %v", req.Prompt)
		}
	})
	// Model generation defaults must not affect scoring.
	runner.options.Temperature = 0
	runner.options.Stop = []string{"A"}
	runner.options.RepeatPenalty = 2
	runner.options.NumCtx = 5 // The longest prompt leaves exactly two positions of headroom.
	result, err := runner.Score(t.Context(), ScoreRequest{MaxTokens: 5, Rows: []ScoreRow{
		{Prompt: "abc", Candidates: []string{"Z", "A"}},
		{Prompt: "xy", Candidates: []string{"B"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{float32(math.Log(0.25)), float32(math.Log(0.75))}, {0}}
	if !reflect.DeepEqual(result.Logits, want) || result.InputTokens != 5 || result.OutputTokens != 2 || calls != 2 {
		t.Fatalf("incomplete or reordered scores/usage: %+v, calls=%d", result, calls)
	}
}

func TestLlamaServerScoreSharedPrefix(t *testing.T) {
	var prompts [][]int
	runner := newLlamaScoreTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
		var req scoreCompletionTestRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
			return
		}
		prompts = append(prompts, req.Prompt)
		if len(prompts) == 1 {
			// The native server generates one token even with n_predict: 0.
			if req.NPredict != 0 || req.NProbs != 0 || len(req.LogitBias) != 0 || len(req.Samplers) != 0 || !req.CachePrompt {
				t.Errorf("unexpected primer request: %+v", req)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"tokens_predicted": 1, "tokens_evaluated": len(req.Prompt)})
			return
		}
		candidate := int(req.LogitBias[0][0])
		checkScoreRowRequest(t, req, []int{candidate}, scoreBias)
		_ = json.NewEncoder(w).Encode(scoreCompletionTestResponse(len(req.Prompt), map[int]float64{candidate: 1}))
	})
	runner.options.NumCtx = 64
	result, err := runner.Score(t.Context(), ScoreRequest{MaxTokens: 32, Rows: []ScoreRow{
		{Prompt: "shared:first", Candidates: []string{"A"}},
		{Prompt: "shared:two", Candidates: []string{"B"}},
		{Prompt: "shared:x", Candidates: []string{"C"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	// The primer evaluates the shared prefix plus 4 tokens of the first row,
	// which continues from it; the other rows restore the checkpoint.
	want := [][]int{scoreTestTokens("shared:firs"), scoreTestTokens("shared:first"), scoreTestTokens("shared:two"), scoreTestTokens("shared:x")}
	if !reflect.DeepEqual(prompts, want) {
		t.Fatalf("got prompts %v, want %v", prompts, want)
	}
	if !reflect.DeepEqual(result.Logits, [][]float32{{0}, {0}, {0}}) || result.InputTokens != 30 || result.OutputTokens != 4 {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestLlamaServerScorePrimer(t *testing.T) {
	for _, tt := range []struct {
		name   string
		rows   []string
		primer string
	}{
		{"one row", []string{"abcdefgh"}, ""},
		{"no shared prefix", []string{"abcdefgh", "xbcdefgh"}, ""},
		{"first row too short", []string{"abcdefg", "abcdwxyzuv"}, ""},
		{"identical prompts", []string{"abcdefgh", "abcdefgh"}, ""},
		{"shared prefix", []string{"abcdefghij", "abcxyzuvwt", "abcdefgh"}, "abcdefg"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var primer []int
			runner := newLlamaScoreTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
				var req scoreCompletionTestRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
					return
				}
				if len(req.LogitBias) == 0 {
					primer = req.Prompt
					_ = json.NewEncoder(w).Encode(map[string]any{"tokens_predicted": 1, "tokens_evaluated": len(req.Prompt)})
					return
				}
				_ = json.NewEncoder(w).Encode(scoreCompletionTestResponse(len(req.Prompt), map[int]float64{'A': 1}))
			})
			runner.options.NumCtx = 64
			input := ScoreRequest{MaxTokens: 32}
			for _, row := range tt.rows {
				input.Rows = append(input.Rows, ScoreRow{Prompt: row, Candidates: []string{"A"}})
			}
			if _, err := runner.Score(t.Context(), input); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(primer, scoreTestTokens(tt.primer)) {
				t.Fatalf("primed %v, want %v", primer, scoreTestTokens(tt.primer))
			}
		})
	}
}

// scoreTestTokens tokenizes s the way the test tokenizer does: one token per rune.
func scoreTestTokens(s string) []int {
	var tokens []int
	for _, c := range s {
		tokens = append(tokens, int(c))
	}
	return tokens
}

func TestLlamaServerScoreOutranked(t *testing.T) {
	for _, recovers := range []bool{true, false} {
		t.Run(fmt.Sprintf("recovers=%v", recovers), func(t *testing.T) {
			var calls int
			runner := newLlamaScoreTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
				var req scoreCompletionTestRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
					return
				}
				calls++
				bias := float64(scoreBias)
				if calls > 1 {
					bias = scoreRetryBias
				}
				checkScoreRowRequest(t, req, []int{'A', 'B'}, bias)
				probs := map[int]float64{'A': 0.6, 'x': 0.4}
				if recovers && calls > 1 {
					probs = map[int]float64{'A': 0.6, 'B': 0.4}
				}
				_ = json.NewEncoder(w).Encode(scoreCompletionTestResponse(len(req.Prompt), probs))
			})
			result, err := runner.Score(t.Context(), ScoreRequest{MaxTokens: 5, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"A", "B"}}}})
			if calls != 2 {
				t.Fatalf("got %d attempts, want 2", calls)
			}
			if !recovers {
				if err == nil {
					t.Fatal("accepted a distribution without every candidate")
				}
				return
			}
			if err != nil || result.OutputTokens != 2 || !reflect.DeepEqual(result.Logits, [][]float32{{float32(math.Log(0.6)), float32(math.Log(0.4))}}) {
				t.Fatalf("unexpected result: %+v, %v", result, err)
			}
		})
	}
}

func TestLlamaServerScoreUnderflow(t *testing.T) {
	runner := newLlamaScoreTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
		// llama-server omits candidates whose probability underflows.
		_ = json.NewEncoder(w).Encode(scoreCompletionTestResponse(3, map[int]float64{'B': 1}))
	})
	result, err := runner.Score(t.Context(), ScoreRequest{MaxTokens: 5, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"A", "B"}}}})
	if err != nil || !reflect.DeepEqual(result.Logits, [][]float32{{scoreFloorLogit, 0}}) {
		t.Fatalf("unexpected result: %+v, %v", result, err)
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
		{"no native scoring headroom", ScoreRequest{MaxTokens: 8, Rows: []ScoreRow{{Prompt: "abcdefg", Candidates: []string{"A"}}}}},
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
	const valid = `{"tokens_predicted":1,"tokens_evaluated":3,"completion_probabilities":[{"id":65,"token":"A","prob":1,"top_probs":[{"id":65,"token":"A","prob":1}]}]}`
	for _, tt := range []struct {
		name   string
		body   string
		status int
	}{
		{"missing probability", strings.Replace(valid, `"token":"A","prob":1}]`, `"token":"A"}]`, 1), 200},
		{"missing token id", strings.Replace(valid, `[{"id":65,"token":"A","prob":1}]`, `[{"token":"A","prob":1}]`, 1), 200},
		{"zero probability", strings.Replace(valid, `"token":"A","prob":1}]`, `"token":"A","prob":0}]`, 1), 200},
		{"probability above one", strings.Replace(valid, `"token":"A","prob":1}]`, `"token":"A","prob":1.5}]`, 1), 200},
		{"duplicate candidate", strings.Replace(valid, `[{"id":65,"token":"A","prob":1}]`, `[{"id":65,"token":"A","prob":0.5},{"id":65,"token":"A","prob":0.5}]`, 1), 200},
		{"empty top probabilities", strings.Replace(valid, `[{"id":65,"token":"A","prob":1}]`, `[]`, 1), 200},
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
				fmt.Fprint(w, `{"tokens_predicted":1,"tokens_evaluated":3,"completion_probabilities":[{"id":65,"token":"A","prob":1,"top_probs":[{"id":65,"token":"A","prob":1}]}]}`)
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
