package laya

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
)

var _ llm.Scorer = (*Model)(nil)

func (m *Model) sequence(state string, q llm.ScoreQuestion, maxLen int) (ids, markers []int32, qtype int32, err error) {
	switch q.Type {
	case "choice":
		qtype = 0
	case "score":
		qtype = 1
	case "noul":
		qtype = 2
	default:
		return nil, nil, 0, fmt.Errorf("unknown decision type %q", q.Type)
	}
	if len(q.Options) < 2 || len(q.Options) > 26 {
		return nil, nil, 0, fmt.Errorf("decision requires 2–26 options")
	}
	encode := func(s string) []int32 { return m.tok.Encode(strings.ReplaceAll(s, m.maskToken, " "), false) }
	head := encode(q.Type + " question: " + q.Instructions)
	opts := make([][]int32, len(q.Options))
	budget := m.config.HeadMaxLen
	for i, text := range q.Options {
		tokens := encode(" " + text)
		opts[i] = append([]int32{m.mask}, tokens[:min(48, len(tokens))]...)
		budget -= len(opts[i])
	}
	if budget < 16 {
		per := max(4, (m.config.HeadMaxLen-16)/len(opts))
		budget = m.config.HeadMaxLen
		for i := range opts {
			opts[i] = opts[i][:min(per, len(opts[i]))]
			budget -= len(opts[i])
		}
	}
	ids = append([]int32{m.cls}, head[:min(len(head), max(8, budget))]...)
	ids = append(ids, m.sep)
	for _, tokens := range opts {
		markers = append(markers, int32(len(ids)))
		ids = append(ids, tokens...)
	}
	ids = append(ids, m.sep)
	if len(ids)+1 > maxLen {
		return nil, nil, 0, fmt.Errorf("decision options exceed the %d-token context", maxLen)
	}
	stateTokens := encode(state)
	// Match the publisher's question/state budget: options and instructions
	// have their own cap, then state is truncated on the right to fit.
	ids = append(ids, stateTokens[:min(len(stateTokens), maxLen-len(ids)-1)]...)
	ids = append(ids, m.sep)
	return ids, markers, qtype, nil
}

func (m *Model) temperature(qtype int32, count int) float32 {
	size := "11+"
	switch {
	case count <= 2:
		size = "2"
	case count <= 5:
		size = "3-5"
	case count <= 10:
		size = "6-10"
	}
	t := m.config.Temperature[qtype]
	if v, ok := m.config.TemperatureByOptions[[]string{"choice", "score", "noul"}[qtype]+":"+size]; ok {
		t = v
	}
	// This is the current author SDK's calibration clamp.
	if math.IsNaN(float64(t)) || math.IsInf(float64(t), 0) {
		return 1
	}
	return max(0.5, min(5, t))
}

// Score executes on the runner's MLX thread. Encoder requests never use the
// decoder prefix cache: changing an option changes every bidirectional state.
func (m *Model) Score(ctx context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	badRequest := func(err error) (llm.ScoreResponse, error) {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if len(input.Rows) < 1 || len(input.Rows) > 64 || input.MaxTokens < 1 || input.MaxTokens > m.config.MaxLen {
		return badRequest(fmt.Errorf("invalid Laya row count or context limit"))
	}
	for i, row := range input.Rows {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if row.Question == nil {
			return badRequest(fmt.Errorf("Laya requires structured decision inputs"))
		}
		ids, markers, qtype, err := m.sequence(input.State, *row.Question, input.MaxTokens)
		if err != nil {
			return badRequest(fmt.Errorf("question %d: %w", i, err))
		}
		var logits []float32
		mlx.Scoped(func() {
			h := m.decisionHidden(&batch.Batch{
				InputIDs:     mlx.FromValues(ids, 1, len(ids)),
				SeqOffsets:   []int32{0},
				SeqQueryLens: []int32{int32(len(ids))},
			}, []int32{qtype})
			selected := h.TakeAxis(mlx.FromValues(markers, len(markers)), 1)
			out := mlx.DivScalar(m.Unembed(selected).AsType(mlx.DTypeFloat32), m.temperature(qtype, len(markers))).Reshape(-1)
			mlx.Eval(out)
			logits = out.Floats()
		})
		result.Logits = append(result.Logits, logits)
		result.InputTokens += len(ids)
	}
	return result, ctx.Err()
}
