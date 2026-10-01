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
		outputs := mlx.ScopedEval(func() []*mlx.Array {
			// Keep the full token sequence: both span means and cross-attention need it.
			ids := mlx.FromValues(record.ids, 1, len(record.ids))
			b := &batch.Batch{InputIDs: ids, SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(len(record.ids))}, Layout: []any{prepared.Layout}}
			for i := range prepared.Items {
				item := &prepared.Items[i]
				pixels := mlx.FromValues(item.MediaData, item.Dims...)
				b.Media = append(b.Media, batch.MediaItem{Seq: 0, Pos: item.Range[0], Features: m.EncodeMedia(item, pixels), Opaque: item.Opaque})
			}
			hidden, _ := m.Model.Forward(b, nil)
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
	if len(result.Logits) != len(input.Fields) {
		return result, fmt.Errorf("Clef returned an incomplete schema")
	}
	result.InputTokens = len(record.ids)
	return result, ctx.Err()
}
