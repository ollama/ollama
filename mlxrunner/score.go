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
	"github.com/ollama/ollama/mlx/mlxthread"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

type scoreRow struct {
	tokens     []int32
	candidates []int32
}

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
	result, err := mlxthread.Call(req.Context(), r.mlxThread, func() (llm.ScoreResponse, error) {
		return r.score(req.Context(), input)
	})
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
	logits, err := r.scoreRows(ctx, rows, scorePrefixLength(rows))
	result.Logits = logits
	return result, err
}

// Leave at least one token for each branch, including identical prompts. Its
// final hidden state predicts the answer; generation's one-token-short prefill
// cannot supply that state.
func scorePrefixLength(rows []scoreRow) int {
	if len(rows) < 2 {
		return 0
	}
	n := len(rows[0].tokens) - 1
	for _, row := range rows[1:] {
		n = min(n, len(row.tokens)-1)
		for i := 0; i < n; i++ {
			if row.tokens[i] != rows[0].tokens[i] {
				n = i
				break
			}
		}
	}
	return max(0, n)
}

// scoreRows runs only on the MLX worker. Caches belong to this request and never
// enter the generation prefix trie. Restore rewinds KV storage and reinstates
// both convolution and recurrent state before each suffix.
func (r *Runner) scoreRows(ctx context.Context, rows []scoreRow, prefix int) ([][]float32, error) {
	caches := r.Model.NewCaches()
	snapshots := make([]cache.Snapshot, len(caches))
	defer func() {
		// Close lazy snapshots before freeing their backing caches.
		for _, snapshot := range snapshots {
			if snapshot != nil {
				snapshot.Close()
			}
		}
		for _, c := range caches {
			if c != nil {
				c.Free()
			}
		}
	}()
	if prefix > 0 {
		if _, err := r.scoreForward(ctx, rows[0].tokens[:prefix], 0, caches, nil); err != nil {
			return nil, err
		}
		for i, c := range caches {
			if c != nil {
				snapshots[i] = c.Snapshot(0)
				if snapshots[i] == nil {
					return nil, fmt.Errorf("cache %d cannot snapshot the scoring prefix", i)
				}
			}
		}
	}
	result := make([][]float32, len(rows))
	for i, row := range rows {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if i > 0 {
			for j, c := range caches {
				if c == nil {
					continue
				}
				if prefix == 0 {
					c.Free()
				} else if !c.Restore(snapshots[j], prefix) {
					return nil, fmt.Errorf("cache %d cannot restore the scoring prefix", j)
				}
			}
		}
		logits, err := r.scoreForward(ctx, row.tokens[prefix:], prefix, caches, row.candidates)
		if err != nil {
			return nil, err
		}
		result[i] = logits
	}
	return result, ctx.Err()
}

func (r *Runner) scoreForward(ctx context.Context, tokens []int32, offset int, caches []cache.Cache, candidates []int32) ([]float32, error) {
	var logits []float32
	for len(tokens) > 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := min(prefillChunkSize(), len(tokens))
		mlx.Scoped(func() {
			output := mlx.ScopedArrays(func() []*mlx.Array {
				hidden, _ := r.Model.Forward(&batch.Batch{
					InputIDs:     mlx.FromValues(tokens[:n], 1, n),
					SeqOffsets:   []int32{int32(offset)},
					SeqQueryLens: []int32{int32(n)},
				}, caches)
				if n != len(tokens) || len(candidates) == 0 {
					return nil
				}
				last := hidden.Slice(mlx.Slice(), mlx.Slice(n-1, n), mlx.Slice())
				ids := mlx.FromValues(candidates, len(candidates))
				var selected *mlx.Array
				if m, ok := r.Model.(candidateUnembedder); ok {
					selected = m.UnembedCandidates(last, ids)
				} else {
					selected = r.Model.Unembed(last).Reshape(-1).TakeAxis(ids, 0)
				}
				return []*mlx.Array{selected.AsType(mlx.DTypeFloat32)}
			})
			state := slices.Clone(output)
			for _, c := range caches {
				if c != nil {
					state = append(state, c.State()...)
				}
			}
			mlx.Eval(state...)
			if len(output) > 0 {
				logits = output[0].Floats()
			}
		})
		tokens = tokens[n:]
		offset += n
	}
	return logits, ctx.Err()
}
