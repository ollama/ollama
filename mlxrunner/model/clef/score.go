package clef

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

// ScorePlan holds one schema. Its questions interact in the joint head, so a
// complete schema is one backbone row even when it contains many questions.
type ScorePlan struct {
	Tokens    []int32
	prepared  *model.PreparedRequest
	segments  []model.Segment
	questions []encodedQuestion
}

// PrepareScore is CPU-only and needs no MLX thread or array scope, allowing
// preparation to overlap GPU work. Score currently calls it serially.
func (m *Model) PrepareScore(ctx context.Context, input llm.ScoreRequest) (*ScorePlan, error) {
	if len(input.Segments) < 3 || len(input.Fields) < 1 || len(input.Fields) > 64 || input.MaxTokens < 1 || input.MaxTokens > m.MaxContextLength() {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "invalid Clef question count or context limit"}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(input.Images) > 0 && input.ImagePosition != 1 {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "Clef images must precede the state"}
	}
	segments := []model.Segment{{Tokens: m.Tokenizer().Encode(input.Segments[0], false)}}
	for _, image := range input.Images {
		if len(image) == 0 {
			return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "image must not be empty"}
		}
		segments = append(segments, model.Segment{Kind: "image", Data: image})
	}
	if len(input.Images) > 0 {
		segments = append(segments, model.Segment{Tokens: m.Tokenizer().Encode("\n", false)})
	}
	prepared, err := m.PrepareMedia(segments)
	if err != nil {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	record, err := encode(m.Tokenizer(), input, prepared.Tokens)
	if err != nil {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	prepared.Tokens = record.ids
	return &ScorePlan{Tokens: record.ids, prepared: prepared, segments: segments, questions: record.questions}, nil
}

// FinishScore runs the unpadded joint head for this schema. The backbone can
// batch compatible text rows, but the head's attention is schema-local.
// Call it on the MLX thread, within the caller's scope, with one complete hidden row.
func (m *Model) FinishScore(plan *ScorePlan, hidden *mlx.Array) [][]float32 {
	ids := mlx.FromValues(plan.Tokens, len(plan.Tokens))
	outputs := mlx.ScopedEval(func() []*mlx.Array {
		outputs := m.Head.forward(hidden, ids, plan.questions, m.OutputEmbedding)
		for i := range outputs {
			outputs[i] = outputs[i].AsType(mlx.DTypeFloat32)
		}
		return outputs
	})
	result := make([][]float32, len(outputs))
	for i, output := range outputs {
		result[i] = output.Floats()
	}
	return result
}

// Score retains the serial runner path while sharing preparation and readout.
// Keep preparation and readout model-owned and separate from forward execution
// so scheduling can evolve without duplicating model logic.
func (m *Model) Score(ctx context.Context, input llm.ScoreRequest, forward model.ScoreForward) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	cached := 0
	result.CachedTokens = &cached
	plan, err := m.PrepareScore(ctx, input)
	if err != nil {
		return result, err
	}
	mlx.Scoped(func() {
		var hidden *mlx.Array
		hidden, cached, err = forward(ctx, plan.prepared, plan.segments)
		if err != nil {
			return
		}
		result.Logits = m.FinishScore(plan, hidden)
	})
	if err != nil {
		return result, err
	}
	if len(result.Logits) != len(input.Fields) {
		return result, fmt.Errorf("Clef returned an incomplete schema")
	}
	result.InputTokens = len(plan.Tokens)
	return result, ctx.Err()
}
