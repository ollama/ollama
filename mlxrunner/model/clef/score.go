package clef

import (
	"context"
	"fmt"
	"net/http"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
)

var _ llm.Scorer = (*Model)(nil)

// ScorePlan holds one schema. Its questions interact in the joint head, so a
// complete schema is one backbone row even when it contains many questions.
type ScorePlan struct {
	Tokens    []int32
	prepared  *model.PreparedRequest
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
	// Reserve the entire image prefix before truncating only the state.
	record, err := encode(m.Tokenizer(), input, prepared.Tokens)
	if err != nil {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &ScorePlan{Tokens: record.ids, prepared: prepared, questions: record.questions}, nil
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
func (m *Model) Score(ctx context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	plan, err := m.PrepareScore(ctx, input)
	if err != nil {
		return result, err
	}
	mlx.Scoped(func() {
		ids := mlx.FromValues(plan.Tokens, 1, len(plan.Tokens))
		var hidden *mlx.Array
		hidden, err = m.prefill(ctx, ids, plan.prepared)
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

// prefill evaluates bounded token chunks so cancellation can release the worker.
// Keep all hidden states: the joint head uses span means and cross-attention.
func (m *Model) prefill(ctx context.Context, ids *mlx.Array, prepared *model.PreparedRequest) (*mlx.Array, error) {
	caches := m.Model.NewCaches()
	defer func() {
		for _, c := range caches {
			if c != nil {
				c.Free()
			}
		}
	}()
	var err error
	out := mlx.ScopedArrays(func() []*mlx.Array {
		var media []batch.MediaItem
		for i := range prepared.Items {
			if err = ctx.Err(); err != nil {
				return nil
			}
			item := &prepared.Items[i]
			features := mlx.ScopedEval(func() []*mlx.Array {
				pixels := mlx.FromValues(item.MediaData, item.Dims...)
				return []*mlx.Array{m.EncodeMedia(item, pixels)}
			})[0]
			media = append(media, batch.MediaItem{Seq: 0, Pos: item.Range[0], Features: features, Opaque: item.Opaque})
		}
		var chunks []*mlx.Array
		const chunkSize = 2048
		for pos := 0; pos < ids.Dim(1); pos += chunkSize {
			if err = ctx.Err(); err != nil {
				return nil
			}
			n := min(chunkSize, ids.Dim(1)-pos)
			hidden := mlx.ScopedArrays(func() []*mlx.Array {
				hidden, _ := m.Model.Forward(&batch.Batch{
					InputIDs:     ids.Slice(mlx.Slice(), mlx.Slice(pos, pos+n)),
					SeqOffsets:   []int32{int32(pos)},
					SeqQueryLens: []int32{int32(n)},
					Media:        media,
					Layout:       []any{prepared.Layout},
				}, caches)
				return []*mlx.Array{hidden}
			})[0]
			state := []*mlx.Array{hidden}
			for _, c := range caches {
				if c != nil {
					state = append(state, c.State()...)
				}
			}
			mlx.Eval(state...)
			chunks = append(chunks, hidden)
		}
		if err = ctx.Err(); err != nil {
			return nil
		}
		return []*mlx.Array{mlx.Concatenate(chunks, 1)}
	})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
