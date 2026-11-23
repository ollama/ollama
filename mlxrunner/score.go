package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
)

type scoreRow struct {
	tokens     []int32
	candidates []int32
}

// Retain enough scratch space for short decision requests without keeping the
// temporary working set of a large prompt resident between requests.
const scoreScratchLimit = 512 << 20

// Models may project just the requested vocabulary rows at higher precision.
type candidateUnembedder interface {
	UnembedCandidates(hidden, candidates *mlx.Array) *mlx.Array
}

func (r *Runner) scoreHandler(w http.ResponseWriter, req *http.Request) {
	var input llm.ScoreRequest
	if err := json.NewDecoder(req.Body).Decode(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	result, err := r.score(req.Context(), input)
	if err != nil {
		status := http.StatusInternalServerError
		var statusErr api.StatusError
		if errors.As(err, &statusErr) {
			status = statusErr.StatusCode
		}
		http.Error(w, err.Error(), status)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(result)
}

func (r *Runner) score(ctx context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	if scorer, ok := r.Model.(model.Scorer); ok {
		return scorer.Score(ctx, input, r.forwardHidden)
	}
	var result llm.ScoreResponse
	badRequest := func(format string, args ...any) (llm.ScoreResponse, error) {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf(format, args...)}
	}
	if len(input.Rows) < 1 || len(input.Rows) > 64 {
		return badRequest("scoring requires 1–64 prompts")
	}
	if input.MaxTokens < 1 || input.MaxTokens > r.contextLength {
		return badRequest("max_tokens must be between 1 and %d", r.contextLength)
	}
	rows := make([]scoreRow, len(input.Rows))
	for i, row := range input.Rows {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		tokens := r.Tokenizer.Encode(row.Prompt, false)
		if len(tokens) == 0 || len(tokens) > input.MaxTokens {
			return badRequest("prompt %d has %d tokens; expected 1–%d (input is never truncated)", i, len(tokens), input.MaxTokens)
		}
		if len(row.Candidates) < 1 || len(row.Candidates) > 26 {
			return badRequest("prompt %d requires 1–26 candidates", i)
		}
		rows[i].tokens = tokens
		result.InputTokens += len(tokens)
		for _, candidate := range row.Candidates {
			joined := r.Tokenizer.Encode(row.Prompt+candidate, false)
			_, special := r.Tokenizer.GetSpecialToken(candidate)
			if special || len(joined) != len(tokens)+1 || !slices.Equal(joined[:len(tokens)], tokens) {
				return badRequest("candidate %q must append exactly one ordinary token to prompt %d", candidate, i)
			}
			id := joined[len(tokens)]
			if id < 0 || int(id) >= r.Tokenizer.VocabSize() || slices.Contains(rows[i].candidates, id) {
				return badRequest("prompt %d has invalid or duplicate candidate tokens", i)
			}
			rows[i].candidates = append(rows[i].candidates, id)
		}
	}
	logits, cached, err := r.scoreRows(ctx, rows, scorePrefixLength(rows))
	result.CachedTokens = &cached
	result.Logits = logits
	return result, err
}

// Shared boundaries retain both recurrent state and hidden outputs, including
// the complete shorter row when one prompt is a prefix of another.
func scorePrefixLength(rows []scoreRow) int {
	if len(rows) < 2 {
		return 0
	}
	n := len(rows[0].tokens)
	for _, row := range rows[1:] {
		n = min(n, len(row.tokens))
		for i := 0; i < n; i++ {
			if row.tokens[i] != rows[0].tokens[i] {
				n = i
				break
			}
		}
	}
	return max(0, n)
}

func (r *Runner) scoreRows(ctx context.Context, rows []scoreRow, prefix int) ([][]float32, int, error) {
	result := make([][]float32, len(rows))
	cached := 0
	for i, row := range rows {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		restored, err := r.forwardHidden(ctx, &model.PreparedRequest{Tokens: row.tokens, ScorePrefix: prefix}, nil, func(hidden *mlx.Array) error {
			last := hidden.Slice(mlx.Slice(), mlx.Slice(len(row.tokens)-1, len(row.tokens)), mlx.Slice())
			ids := mlx.FromValues(row.candidates, len(row.candidates))
			var selected *mlx.Array
			if m, ok := r.Model.(candidateUnembedder); ok {
				selected = m.UnembedCandidates(last, ids)
			} else {
				selected = r.Model.Unembed(last).Reshape(-1).TakeAxis(ids, 0)
			}
			result[i] = selected.AsType(mlx.DTypeFloat32).Floats()
			return nil
		})
		cached += restored
		if err != nil {
			return nil, 0, err
		}
	}
	return result, cached, ctx.Err()
}

func scoreSnapshots(length, shared int) []int {
	offsets := []int{shared}
	for n := prefillSnapshotInterval; n < length; n += prefillSnapshotInterval {
		offsets = append(offsets, n)
	}
	return offsets
}
