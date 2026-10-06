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
	result, err := mlxthread.Call(req.Context(), r.mlxThread, func() (llm.ScoreResponse, error) {
		if !mlx.MetalIsAvailable() {
			defer mlx.ClearCache()
			return r.score(req.Context(), input)
		}
		keepScratch := false
		defer func() {
			// Keep small reusable buffers, but release large working sets while idle.
			if !keepScratch || mlx.CacheMemory() > scoreScratchLimit {
				mlx.ClearCache()
			}
		}()
		result, err := r.score(req.Context(), input)
		// Eval waits for results, but completion handlers can still free buffers.
		mlx.DefaultStream().Synchronize()
		keepScratch = err == nil
		return result, err
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
	if scorer, ok := r.Model.(model.CachedScorer); ok {
		return scorer.Score(ctx, input, r.scoreHiddenRow)
	}
	if scorer, ok := r.Model.(llm.Scorer); ok {
		return scorer.Score(ctx, input)
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

// Scoring has no draft writes. Keep its model caches separate from generation's
// target/draft pair, using the same prefix matching, snapshots and eviction.
func (r *Runner) scoringCache() *prefixCache {
	if r.scoreCache == nil {
		caches := r.Model.NewCaches()
		r.scoreHidden = cache.NewHiddenCache()
		caches = append(caches, r.scoreHidden)
		r.scoreCache = newPrefixCache(caches)
	}
	return r.scoreCache
}

func (r *Runner) scoreRows(ctx context.Context, rows []scoreRow, prefix int) ([][]float32, int, error) {
	result := make([][]float32, len(rows))
	cached := 0
	for i, row := range rows {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		var err error
		mlx.Scoped(func() {
			session := r.scoringCache().beginScore(row.tokens, nil)
			defer session.close()
			session.schedulePrefillSnapshots(scoreSnapshots(len(row.tokens), prefix))
			offset := len(row.tokens) - len(session.remaining)
			cached += offset
			err = r.scoreForward(ctx, session.remaining, offset, session.caches, nil)
			if err != nil {
				return
			}
			hidden := r.scoreHidden.State()[0]
			last := hidden.Slice(mlx.Slice(), mlx.Slice(len(row.tokens)-1, len(row.tokens)), mlx.Slice())
			ids := mlx.FromValues(row.candidates, len(row.candidates))
			var selected *mlx.Array
			if m, ok := r.Model.(candidateUnembedder); ok {
				selected = m.UnembedCandidates(last, ids)
			} else {
				selected = r.Model.Unembed(last).Reshape(-1).TakeAxis(ids, 0)
			}
			result[i] = selected.AsType(mlx.DTypeFloat32).Floats()
		})
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

func (r *Runner) scoreHiddenRow(ctx context.Context, input *model.PreparedRequest, segments []model.Segment) (*mlx.Array, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	items, err := bindItems(input, segments)
	if err != nil {
		return nil, 0, err
	}
	session := r.scoringCache().beginScore(input.Tokens, items)
	defer session.close()
	session.schedulePrefillSnapshots(scoreSnapshots(len(input.Tokens), 0))
	media := r.openMedia(Request{Tokens: input.Tokens, MediaItems: items, Layout: input.Layout})
	defer media.close()
	offset := len(input.Tokens) - len(session.remaining)
	media.free(offset)
	if err := r.scoreForward(ctx, session.remaining, offset, session.caches, media); err != nil {
		return nil, 0, err
	}
	return r.scoreHidden.State()[0].Slice(mlx.Slice(), mlx.Slice(), mlx.Slice()), offset, nil
}

func (r *Runner) scoreForward(ctx context.Context, tokens []int32, offset int, caches []cache.Cache, media *requestMedia) error {
	for len(tokens) > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := media.extendChunk(offset, min(prefillChunkSize(), len(tokens)))
		mlx.Scoped(func() {
			mlx.Scoped(func() {
				hidden, _ := r.Model.Forward(&batch.Batch{
					InputIDs:     mlx.FromValues(tokens[:n], 1, n),
					SeqOffsets:   []int32{int32(offset)},
					SeqQueryLens: []int32{int32(n)},
					Media:        media.batchMedia(offset, n),
					Layout:       media.rowLayout(),
				}, caches[:len(caches)-1])
				r.scoreHidden.Append(hidden)
			})
			var state []*mlx.Array
			for _, c := range caches {
				if c != nil {
					state = append(state, c.State()...)
				}
			}
			mlx.Eval(state...)
		})
		tokens = tokens[n:]
		offset += n
		media.free(offset)
	}
	return ctx.Err()
}
