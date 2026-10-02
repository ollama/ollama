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

func (m *Model) Score(ctx context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	if len(input.Segments) < 3 || len(input.Fields) < 1 || len(input.Fields) > 64 || input.MaxTokens < 1 || input.MaxTokens > m.MaxContextLength() {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "invalid Clef question count or context limit"}
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if len(input.Images) > 0 && input.ImagePosition != 1 {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "Clef images must precede the state"}
	}
	segments := []model.Segment{{Tokens: m.Tokenizer().Encode(input.Segments[0], false)}}
	for _, image := range input.Images {
		if len(image) == 0 {
			return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "image must not be empty"}
		}
		segments = append(segments, model.Segment{Kind: "image", Data: image})
	}
	if len(input.Images) > 0 {
		segments = append(segments, model.Segment{Tokens: m.Tokenizer().Encode("\n", false)})
	}
	prepared, err := m.PrepareMedia(segments)
	if err != nil {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	// Reserve the entire image prefix before truncating only the state.
	record, err := encode(m.Tokenizer(), input, prepared.Tokens)
	if err != nil {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	mlx.Scoped(func() {
		ids := mlx.FromValues(record.ids, 1, len(record.ids))
		var hidden *mlx.Array
		hidden, err = m.prefill(ctx, ids, prepared)
		if err != nil {
			return
		}
		outputs := mlx.ScopedEval(func() []*mlx.Array {
			outputs := m.Head.forward(hidden, ids.Squeeze(0), record.questions, m.OutputEmbedding)
			for i := range outputs {
				outputs[i] = outputs[i].AsType(mlx.DTypeFloat32)
			}
			return outputs
		})
		for _, output := range outputs {
			result.Logits = append(result.Logits, output.Floats())
		}
	})
	if err != nil {
		return result, err
	}
	if len(result.Logits) != len(input.Fields) {
		return result, fmt.Errorf("Clef returned an incomplete schema")
	}
	result.InputTokens = len(record.ids)
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
