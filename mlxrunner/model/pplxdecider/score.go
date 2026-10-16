package pplxdecider

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/model/renderers"
)

var _ llm.Scorer = (*Model)(nil)

const imageMarker = "<|vision_start|><|image_pad|><|vision_end|>"

func (m *Model) scorePrompt(input llm.ScoreRequest, row llm.ScoreRow) (string, int, error) {
	if row.Question == nil || len(row.Question.Options) < 1 || len(row.Question.Options) > len(m.config.Codes) {
		return "", 0, fmt.Errorf("invalid Perplexity question or option count")
	}
	var body strings.Builder
	// The server uses [img] placeholders for other model families. This
	// checkpoint's processor expects native Qwen image markers without spaces.
	for range input.Images {
		body.WriteString(imageMarker)
	}
	body.WriteString("State:\n")
	body.WriteString(input.State)
	body.WriteString("\n\nQuestion:\n")
	body.WriteString(row.Question.Instructions)
	body.WriteString("\n\nOptions:\n")
	for i, option := range row.Question.Options {
		if i > 0 {
			body.WriteByte('\n')
		}
		fmt.Fprintf(&body, "%s: %s", m.config.Codes[i], option)
	}
	body.WriteString("\n\nReturn only the letter code of the best option.")
	userBody := body.String()
	prompt, err := renderers.RenderWithRenderer("qwen3.8", []api.Message{
		{Role: "system", Content: input.System},
		{Role: "user", Content: userBody},
	}, nil, &api.ThinkValue{Value: false})
	if err != nil {
		return "", 0, err
	}
	start := strings.LastIndex(prompt, userBody)
	if start < 0 {
		return "", 0, fmt.Errorf("rendered Perplexity prompt lost its user message")
	}
	return prompt, start, nil
}

func (m *Model) prepareRow(input llm.ScoreRequest, row llm.ScoreRow) (*model.PreparedRequest, error) {
	prompt, imageStart, err := m.scorePrompt(input, row)
	if err != nil {
		return nil, err
	}
	if len(input.Images) == 0 {
		return m.PrepareMedia([]model.Segment{{Tokens: m.Tokenizer().Encode(prompt, false)}})
	}
	// The rendered user body starts with exactly one marker per attached image.
	// Markers in the system or state text are not image attachments.
	var segments []model.Segment
	start := 0
	for i, image := range input.Images {
		end := imageStart + i*len(imageMarker)
		if !strings.HasPrefix(prompt[end:], imageMarker) {
			return nil, fmt.Errorf("image %d is missing from the rendered prompt", i)
		}
		segments = append(segments, model.Segment{Tokens: m.Tokenizer().Encode(prompt[start:end], false)})
		segments = append(segments, model.Segment{Kind: "image", Data: image})
		start = end + len(imageMarker)
	}
	segments = append(segments, model.Segment{Tokens: m.Tokenizer().Encode(prompt[start:], false)})
	return m.PrepareMedia(segments)
}

// Score evaluates every complete row. Noncausal full attention makes a prefix
// snapshot invalid when the question or state changes, even if tokens match.
func (m *Model) Score(ctx context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	cached := 0
	result.CachedTokens = &cached
	badRequest := func(err error) (llm.ScoreResponse, error) {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if input.System == "" || len(input.Rows) < 1 || len(input.Rows) > 64 || input.MaxTokens < 1 || input.MaxTokens > m.MaxContextLength() {
		return badRequest(fmt.Errorf("invalid Perplexity system prompt, row count, or context limit"))
	}
	for i, row := range input.Rows {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		prepared, err := m.prepareRow(input, row)
		if err != nil {
			return badRequest(fmt.Errorf("question %d: %w", i, err))
		}
		if len(prepared.Tokens) < 1 || len(prepared.Tokens) > input.MaxTokens {
			return badRequest(fmt.Errorf("question %d has %d tokens; limit is %d", i, len(prepared.Tokens), input.MaxTokens))
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		var logits []float32
		mlx.Scoped(func() {
			media := make([]batch.MediaItem, len(prepared.Items))
			for j := range prepared.Items {
				item := &prepared.Items[j]
				data := mlx.FromValues(item.MediaData, item.Dims...)
				media[j] = batch.MediaItem{Pos: item.Range[0], Opaque: item.Opaque, Features: m.EncodeMedia(item, data)}
				item.MediaData = nil
			}
			b := &batch.Batch{
				InputIDs:     mlx.FromValues(prepared.Tokens, 1, len(prepared.Tokens)),
				SeqOffsets:   []int32{0},
				SeqQueryLens: []int32{int32(len(prepared.Tokens))},
				Media:        media,
			}
			if prepared.Layout != nil {
				b.Layout = []any{prepared.Layout}
			}
			hidden, _ := m.Forward(b, nil)
			last := hidden.Slice(mlx.Slice(), mlx.Slice(len(prepared.Tokens)-1, len(prepared.Tokens)), mlx.Slice())
			count := len(row.Question.Options)
			output := mlx.DivScalar(m.Readout.Forward(last).AsType(mlx.DTypeFloat32), m.config.Temperature)
			output = output.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, count)).Reshape(count)
			mlx.Eval(output)
			logits = output.Floats()
		})
		result.Logits = append(result.Logits, logits)
		result.InputTokens += len(prepared.Tokens)
	}
	return result, ctx.Err()
}
