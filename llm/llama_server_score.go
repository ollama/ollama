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
	"strings"

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
	if len(input.Segments) == 0 && len(input.Fields) == 0 && (len(input.Rows) < 1 || len(input.Rows) > 64) {
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

	if len(input.Segments) > 0 || len(input.Fields) > 0 {
		return s.scoreFields(ctx, input)
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
	var output scoreCompletionResponse
	if err := s.scoreRequest(ctx, "/completion", scoreCompletionRequest{Prompt: tokens, CachePrompt: true, NPredict: 0}, &output); err != nil {
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
		var output scoreCompletionResponse
		if err := s.scoreRequest(ctx, "/completion", req, &output); err != nil {
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

func (s *llamaServerRunner) scoreRequest(ctx context.Context, path string, input, output any) error {
	data, err := json.Marshal(input)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d%s", s.port, path), bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if msg := s.lastErrMsg(); msg != "" {
			return fmt.Errorf("scoring failed: %s: %w", msg, err)
		}
		return err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return err
	}
	if res.StatusCode != http.StatusOK {
		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(body)}
	}
	if err := json.Unmarshal(body, output); err != nil {
		return fmt.Errorf("invalid scoring response: %w", err)
	}
	return nil
}

func (s *llamaServerRunner) scoreFields(ctx context.Context, input ScoreRequest) (ScoreResponse, error) {
	bad := func(message string) (ScoreResponse, error) {
		return ScoreResponse{}, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: message}
	}
	if len(input.Fields) < 1 || len(input.Fields) > 64 || len(input.Segments) == 0 {
		return bad("invalid decision scoring request")
	}
	var err error
	// Do not concatenate before tokenizing: BPE may merge across the reference
	// encoder's segment boundaries, changing both embeddings and span offsets.
	var tokens []int
	offsets := []int{0}
	cache := map[string][]int{}
	special := true
	for _, segment := range input.Segments {
		ids, ok := cache[segment]
		if !ok {
			ids, err = s.tokenize(ctx, segment, false, &special)
			if err != nil {
				return ScoreResponse{}, err
			}
			cache[segment] = ids
		}
		tokens = append(tokens, ids...)
		offsets = append(offsets, len(tokens))
		if len(tokens) > input.MaxTokens {
			return bad("decision prompt exceeds the model context (input is never truncated)")
		}
	}
	if len(tokens) == 0 {
		return bad("decision prompt must not be empty")
	}
	fields := make([]ScoreField, len(input.Fields))
	for i, f := range input.Fields {
		if f.Type < 0 || f.Type > 2 || len(f.Options) < 2 || len(f.Options) > 26 {
			return bad("invalid decision field")
		}
		// Map the question and its options from segment indexes to token offsets.
		spans := append([][2]int{f.Question}, f.Options...)
		for j, span := range spans {
			start, end := span[0], span[1]
			if start < 0 || start >= end || end >= len(offsets) || offsets[start] == offsets[end] {
				return bad("invalid decision token span")
			}
			spans[j] = [2]int{offsets[start], offsets[end]}
		}
		fields[i] = ScoreField{Type: f.Type, Question: spans[0], Options: spans[1:]}
	}
	var images []api.ImageData
	for _, image := range input.Images {
		data, err := llamaServerMediaBytes(image)
		if err != nil {
			return bad(err.Error())
		}
		if !strings.HasPrefix(http.DetectContentType(data), "image/") {
			return bad("invalid decision image")
		}
		images = append(images, data)
	}
	imagePosition := 0
	if len(images) > 0 {
		if input.ImagePosition < 0 || input.ImagePosition >= len(offsets) {
			return bad("invalid image position")
		}
		imagePosition = offsets[input.ImagePosition]
	}
	request := struct {
		Images        []api.ImageData `json:"images,omitempty"`
		ImagePosition int             `json:"image_position"`
		Input         []int           `json:"input"`
		Fields        []ScoreField    `json:"score_fields"`
	}{images, imagePosition, tokens, fields}
	var result []struct {
		Logits [][]float32 `json:"logits"`
		Tokens int         `json:"tokens_evaluated"`
	}
	if err := s.scoreRequest(ctx, "/embedding", request, &result); err != nil {
		return ScoreResponse{}, err
	}
	if len(result) != 1 || result[0].Tokens > input.MaxTokens || (len(images) == 0 && result[0].Tokens != len(tokens)) || (len(images) > 0 && result[0].Tokens <= len(tokens)) || len(result[0].Logits) != len(fields) {
		return ScoreResponse{}, fmt.Errorf("decision runner did not score the complete request")
	}
	return ScoreResponse{Logits: result[0].Logits, InputTokens: result[0].Tokens}, nil
}
