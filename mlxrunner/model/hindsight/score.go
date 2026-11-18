package hindsight

import (
	"context"
	"fmt"
	"net/http"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
)

var _ model.CachedScorer = (*Model)(nil)

// Score renders the request and runs it through the runner's cached causal
// forward, which prefills in chunks and reuses a cached prefix shared with an
// earlier request. It returns per question one logit per option: the highest
// of the LM-head logits of its code's spellings at the question's slot.
func (m *Model) Score(ctx context.Context, input llm.ScoreRequest, forward model.ScoreForward) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	cached := 0
	result.CachedTokens = &cached
	badRequest := func(err error) (llm.ScoreResponse, error) {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if input.Native == nil || len(input.Native.Questions) == 0 {
		return badRequest(fmt.Errorf("Hindsight decision models need a native System One request"))
	}
	r, err := m.render(input.Native)
	if err != nil {
		return badRequest(err)
	}
	limit := min(input.MaxTokens, m.config.MaxRequestTokens)
	if len(r.ids) > limit {
		// Hindsight's decision clients split a request and resend it when a 400 names
		// max_tokens_exceeded; any other refusal fails the call.
		return badRequest(fmt.Errorf("max_tokens_exceeded: decision input needs %d tokens; the context limit is %d", len(r.ids), limit))
	}
	mlx.Scoped(func() {
		var hidden *mlx.Array
		var restored int
		hidden, restored, err = forward(ctx, &model.PreparedRequest{Tokens: r.ids}, nil)
		cached = restored
		if err != nil {
			return
		}
		for i, slot := range r.slots {
			result.Logits = append(result.Logits, m.slotLogits(hidden, slot, r.groups[i]))
		}
	})
	if err != nil {
		return result, err
	}
	result.InputTokens = len(r.ids)
	return result, ctx.Err()
}

// slotLogits is one logit per option at row slot of hidden: the highest
// LM-head logit among the token ids of the option code's spellings.
func (m *Model) slotLogits(hidden *mlx.Array, slot int, groups [][]int32) []float32 {
	var candidates []int32
	for _, group := range groups {
		candidates = append(candidates, group...)
	}
	row := hidden.Slice(mlx.Slice(), mlx.Slice(slot, slot+1), mlx.Slice())
	scores := m.UnembedCandidates(row, mlx.FromValues(candidates, len(candidates)))
	mlx.Eval(scores)
	values := scores.Floats()
	logits := make([]float32, len(groups))
	k := 0
	for j, group := range groups {
		best := values[k]
		for _, v := range values[k : k+len(group)] {
			best = max(best, v)
		}
		logits[j] = best
		k += len(group)
	}
	return logits
}
