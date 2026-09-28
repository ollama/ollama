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

	"github.com/ollama/ollama/api"
)

var _ Scorer = (*llamaServerRunner)(nil)

const (
	// Candidates are lifted above every other token so that top_k keeps exactly
	// them; a shared bias does not change their softmax. A larger bias is
	// retried if another token still outranks a candidate.
	scoreBias      = 100
	scoreRetryBias = 1000
)

// A candidate whose probability underflows float32 is omitted by llama-server.
var scoreFloorLogit = float32(math.Log(math.SmallestNonzeroFloat32))

type scoreRowTokens struct {
	tokens, candidates []int
}

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

	rows := make([]scoreRowTokens, len(input.Rows))
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
		// llama-server marks the result truncated when prompt+1 reaches the
		// context limit, so scoring needs two positions of headroom.
		if len(tokens)+2 > s.ContextLength() {
			return badRequest("prompt %d requires %d context tokens for scoring; model has %d", i, len(tokens)+2, s.ContextLength())
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

	result.OutputTokens, err = s.primeSharedPrefix(ctx, rows)
	if err != nil {
		return ScoreResponse{}, err
	}

	// Rows are scored in order: the first row continues from the primer.
	result.Logits = make([][]float32, len(rows))
	for i, row := range rows {
		logits, outputTokens, err := s.scoreRow(ctx, row)
		if err != nil {
			return ScoreResponse{}, err
		}
		result.Logits[i] = logits
		result.OutputTokens += outputTokens
	}
	return result, nil
}

// llama-server saves a checkpoint of state it cannot roll back, from recurrent
// and sliding-window layers, this many tokens before the end of each prompt it
// evaluates. See https://github.com/ggml-org/llama.cpp/pull/20288.
const scoreCheckpointOffset = 4

// primeSharedPrefix makes llama-server save a checkpoint at the end of the
// tokens every row shares, so each row evaluates only its own last few tokens
// instead of its whole prompt.
//
// Rows share the rendered context and schema and differ only near the end.
// llama-server usually reuses a cached prefix by trimming its cache, but it
// cannot trim the state of recurrent layers, as in hybrid models like Qwen 3.5,
// or of sliding-window layers. Such models can only resume from a checkpoint,
// and the checkpoints llama-server saves on its own sit near the end of each
// prompt, past the point where the next row differs. Without a primer, every row would be
// evaluated from the start.
//
// The primer evaluates the shared tokens plus the first row's next
// scoreCheckpointOffset tokens into the cache, so llama-server saves its
// checkpoint exactly at the end of the shared tokens. The pinned server still
// generates one token with n_predict 0; it is discarded but counted in usage.
// The first row continues from the primer; later rows restore the checkpoint.
// With 8 questions, scoring is about 3x faster.
//
// Priming only changes speed: if llama-server stops saving that checkpoint,
// rows are evaluated from the start and scores stay the same. It can be
// removed once llama-server can save a checkpoint at a requested position or
// score rows as one batch.
func (s *llamaServerRunner) primeSharedPrefix(ctx context.Context, rows []scoreRowTokens) (int, error) {
	if len(rows) < 2 {
		return 0, nil
	}
	first := rows[0].tokens
	shared := len(first)
	for _, row := range rows[1:] {
		shared = min(shared, len(row.tokens))
		for i := range shared {
			if row.tokens[i] != first[i] {
				shared = i
				break
			}
		}
	}
	// Nothing is shared, or the first row is too short to continue from the
	// primer.
	if shared == 0 || len(first) <= shared+scoreCheckpointOffset {
		return 0, nil
	}
	tokens := first[:shared+scoreCheckpointOffset]
	output, err := s.scoreCompletion(ctx, scoreCompletionRequest{Prompt: tokens, CachePrompt: true, NPredict: 0})
	if err != nil {
		return 0, err
	}
	if output.Truncated || output.TokensEvaluated != len(tokens) {
		return 0, fmt.Errorf("scoring did not evaluate the complete shared prefix")
	}
	return output.TokensPredicted, nil
}

// scoreCompletionRequest is the part of llama-server's /completion request
// that scoring uses.
type scoreCompletionRequest struct {
	Prompt            []int        `json:"prompt"`
	Stream            bool         `json:"stream"`
	CachePrompt       bool         `json:"cache_prompt"`
	NPredict          int          `json:"n_predict"`
	NProbs            int          `json:"n_probs,omitempty"`
	PostSamplingProbs bool         `json:"post_sampling_probs"`
	Samplers          []string     `json:"samplers,omitempty"`
	TopK              int          `json:"top_k"`
	Temperature       float32      `json:"temperature"`
	LogitBias         [][2]float64 `json:"logit_bias,omitempty"`
}

// scoreCompletionResponse is the part of llama-server's /completion response
// that scoring reads.
type scoreCompletionResponse struct {
	TokensPredicted int  `json:"tokens_predicted"`
	TokensEvaluated int  `json:"tokens_evaluated"`
	Truncated       bool `json:"truncated"`
	Probabilities   []struct {
		TopProbs []struct {
			ID   *int     `json:"id"`
			Prob *float64 `json:"prob"`
		} `json:"top_probs"`
	} `json:"completion_probabilities"`
}

// scoreRow reads every candidate from one next-token distribution. Biasing the
// candidates by the same amount keeps their relative logits, so the returned
// probabilities are the softmax over the candidates' original logits.
func (s *llamaServerRunner) scoreRow(ctx context.Context, row scoreRowTokens) ([]float32, int, error) {
	var outputTokens int
	for _, bias := range []float64{scoreBias, scoreRetryBias} {
		req := scoreCompletionRequest{
			Prompt: row.tokens, CachePrompt: true, NPredict: 1,
			NProbs: len(row.candidates), PostSamplingProbs: true,
			Samplers: []string{"top_k", "temperature"}, TopK: len(row.candidates), Temperature: 1,
		}
		for _, id := range row.candidates {
			req.LogitBias = append(req.LogitBias, [2]float64{float64(id), bias})
		}
		output, err := s.scoreCompletion(ctx, req)
		if err != nil {
			return nil, 0, err
		}
		if output.Truncated || output.TokensEvaluated != len(row.tokens) || output.TokensPredicted != 1 || len(output.Probabilities) != 1 || len(output.Probabilities[0].TopProbs) == 0 {
			return nil, 0, fmt.Errorf("scoring did not evaluate the complete prompt and return one distribution")
		}
		outputTokens += output.TokensPredicted
		logits := slices.Repeat([]float32{scoreFloorLogit}, len(row.candidates))
		seen := make([]bool, len(row.candidates))
		outranked := false
		for _, p := range output.Probabilities[0].TopProbs {
			if p.ID == nil || p.Prob == nil || math.IsNaN(*p.Prob) || *p.Prob <= 0 || *p.Prob > 1 {
				return nil, 0, fmt.Errorf("scoring returned an invalid candidate probability")
			}
			j := slices.Index(row.candidates, *p.ID)
			if j < 0 {
				outranked = true
				break
			}
			if seen[j] {
				return nil, 0, fmt.Errorf("scoring returned a candidate twice")
			}
			seen[j] = true
			logits[j] = max(float32(math.Log(*p.Prob)), scoreFloorLogit)
		}
		if !outranked {
			return logits, outputTokens, nil
		}
	}
	return nil, 0, fmt.Errorf("scoring candidates were outranked by other tokens")
}

func (s *llamaServerRunner) scoreCompletion(ctx context.Context, input scoreCompletionRequest) (scoreCompletionResponse, error) {
	var output scoreCompletionResponse
	data, err := json.Marshal(input)
	if err != nil {
		return output, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/completion", s.port), bytes.NewReader(data))
	if err != nil {
		return output, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return output, ctx.Err()
		}
		if msg := s.lastErrMsg(); msg != "" {
			return output, fmt.Errorf("scoring failed: %s: %w", msg, err)
		}
		return output, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return output, err
	}
	if res.StatusCode != http.StatusOK {
		return output, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(body)}
	}
	if err := json.Unmarshal(body, &output); err != nil {
		return output, fmt.Errorf("invalid scoring response: %w", err)
	}
	return output, nil
}
