package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"strconv"

	"github.com/ollama/ollama/api"
)

var _ Scorer = (*llamaServerRunner)(nil)

func (s *llamaServerRunner) Score(ctx context.Context, input ScoreRequest) (ScoreResponse, error) {
	var result ScoreResponse
	badRequest := func(format string, args ...any) (ScoreResponse, error) {
		return ScoreResponse{}, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf(format, args...)}
	}
	if len(input.Rows) < 1 || len(input.Rows) > 64 {
		return badRequest("scoring requires 1–64 prompts")
	}
	if input.MaxTokens < 1 || input.MaxTokens > s.ContextLength() {
		return badRequest("max_tokens must be between 1 and %d", s.ContextLength())
	}
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return result, err
	}
	defer s.sem.Release(1)
	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return result, err
	} else if status != ServerStatusReady {
		return result, fmt.Errorf("unexpected server status: %s", status)
	}

	rows := make([]struct{ tokens, candidates []int }, len(input.Rows))
	parseSpecial, ordinary := true, false
	for i, row := range input.Rows {
		if len(row.Candidates) < 1 || len(row.Candidates) > 26 {
			return badRequest("prompt %d requires 1–26 candidates", i)
		}
		tokens, err := s.tokenize(ctx, row.Prompt, false, &parseSpecial)
		if err != nil {
			return result, err
		}
		if len(tokens) == 0 || len(tokens) > input.MaxTokens {
			return badRequest("prompt %d has %d tokens; expected 1–%d (input is never truncated)", i, len(tokens), input.MaxTokens)
		}
		if len(tokens) >= s.ContextLength() {
			return badRequest("prompt %d requires %d context tokens including the scoring token; model has %d", i, len(tokens)+1, s.ContextLength())
		}
		rows[i].tokens = tokens
		result.InputTokens += len(tokens)
		for _, candidate := range row.Candidates {
			joined, err := s.tokenize(ctx, row.Prompt+candidate, false, &parseSpecial)
			if err != nil {
				return result, err
			}
			literal, err := s.tokenize(ctx, candidate, false, &ordinary)
			if err != nil {
				return result, err
			}
			if len(joined) != len(tokens)+1 || !slices.Equal(joined[:len(tokens)], tokens) || len(literal) != 1 || literal[0] != joined[len(tokens)] {
				return badRequest("candidate %q must append exactly one ordinary token to prompt %d", candidate, i)
			}
			id := joined[len(tokens)]
			if id < 0 || slices.Contains(rows[i].candidates, id) {
				return badRequest("prompt %d has invalid or duplicate candidate tokens", i)
			}
			rows[i].candidates = append(rows[i].candidates, id)
		}
	}

	result.Logits = make([][]float32, len(rows))
	for i, row := range rows {
		for j, id := range row.candidates {
			logprob, err := s.scoreCandidate(ctx, row.tokens, input.Rows[i].Candidates[j], id)
			if err != nil {
				return ScoreResponse{}, err
			}
			result.Logits[i] = append(result.Logits[i], logprob)
			result.OutputTokens++
		}
	}
	return result, nil
}

// Forcing a candidate exposes its original logprob even when it is outside the
// top alternatives. Post-sampling probabilities would instead reflect the grammar.
func (s *llamaServerRunner) scoreCandidate(ctx context.Context, tokens []int, candidate string, id int) (float32, error) {
	data, err := json.Marshal(struct {
		Prompt            []int    `json:"prompt"`
		Grammar           string   `json:"grammar"`
		Stream            bool     `json:"stream"`
		CachePrompt       bool     `json:"cache_prompt"`
		NPredict          int      `json:"n_predict"`
		NProbs            int      `json:"n_probs"`
		PostSamplingProbs bool     `json:"post_sampling_probs"`
		Temperature       float32  `json:"temperature"`
		Samplers          []string `json:"samplers"`
		Seed              int      `json:"seed"`
	}{
		Prompt: tokens, Grammar: "root ::= " + strconv.Quote(candidate),
		CachePrompt: true, NPredict: 1, NProbs: 1, Temperature: 1,
		Samplers: []string{"temperature"}, Seed: 1,
	})
	if err != nil {
		return 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/completion", s.port), bytes.NewReader(data))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return 0, ctx.Err()
		}
		if msg := s.lastErrMsg(); msg != "" {
			return 0, fmt.Errorf("scoring failed: %s: %w", msg, err)
		}
		return 0, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return 0, err
	}
	if res.StatusCode != http.StatusOK {
		return 0, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(body)}
	}
	var output struct {
		TokensPredicted int  `json:"tokens_predicted"`
		TokensEvaluated int  `json:"tokens_evaluated"`
		Truncated       bool `json:"truncated"`
		Probabilities   []struct {
			ID      *int     `json:"id"`
			Token   string   `json:"token"`
			Logprob *float32 `json:"logprob"`
		} `json:"completion_probabilities"`
	}
	if err := json.Unmarshal(body, &output); err != nil {
		return 0, fmt.Errorf("invalid scoring response: %w", err)
	}
	if output.Truncated || output.TokensEvaluated != len(tokens) || output.TokensPredicted != 1 || len(output.Probabilities) != 1 {
		return 0, fmt.Errorf("scoring candidate %q did not evaluate the complete prompt and return exactly one token", candidate)
	}
	p := output.Probabilities[0]
	if p.ID == nil || *p.ID != id || p.Token != candidate || p.Logprob == nil || math.IsNaN(float64(*p.Logprob)) || math.IsInf(float64(*p.Logprob), 0) || *p.Logprob > 0 {
		return 0, fmt.Errorf("scoring candidate %q returned an incorrect token or invalid pre-sampling log probability", candidate)
	}
	return *p.Logprob, nil
}
