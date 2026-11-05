package pplxdecider

import (
	"context"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/model/renderers"
)

var _ llm.Scorer = (*Model)(nil)

const (
	imageMarker   = "<|vision_start|><|image_pad|><|vision_end|>"
	readoutMarker = "<|fim_suffix|>"
)

var specialTokenPattern = regexp.MustCompile(`<\|[^<>]+\|>|<tts_[a-z_]+>`)

// User text must not introduce a readout or chat control token.
func escapeDecisionText(value string) string {
	return specialTokenPattern.ReplaceAllStringFunc(value, func(token string) string {
		return "＜" + token[1:len(token)-1] + "＞"
	})
}

func (m *Model) scorePrompt(input llm.ScoreRequest, rows []llm.ScoreRow) (string, int, error) {
	var body strings.Builder
	for range input.Images {
		body.WriteString(imageMarker)
	}
	body.WriteString("State:\n")
	body.WriteString(escapeDecisionText(input.State))
	for i, row := range rows {
		if row.Question == nil || len(row.Question.Options) < 1 || len(row.Question.Options) > len(m.config.Codes) {
			return "", 0, fmt.Errorf("question %d has an invalid option count", i)
		}
		body.WriteString("\n\nQuestion ")
		fmt.Fprintf(&body, "%d:\n", i)
		body.WriteString(escapeDecisionText(row.Question.Instructions))
		body.WriteString("\nOptions:\n")
		for j, option := range row.Question.Options {
			if j > 0 {
				body.WriteByte('\n')
			}
			fmt.Fprintf(&body, "%s: %s", m.config.Codes[j], escapeDecisionText(option))
		}
		body.WriteByte('\n')
		body.WriteString(readoutMarker)
	}
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

func (m *Model) prepareJoint(input llm.ScoreRequest, rows []llm.ScoreRow) (*model.PreparedRequest, []int32, error) {
	prompt, imageStart, err := m.scorePrompt(input, rows)
	if err != nil {
		return nil, nil, err
	}
	var prepared *model.PreparedRequest
	if len(input.Images) == 0 {
		prepared, err = m.PrepareMedia([]model.Segment{{Tokens: m.Tokenizer().Encode(prompt, false)}})
	} else {
		var segments []model.Segment
		start := 0
		for i, image := range input.Images {
			end := imageStart + i*len(imageMarker)
			if !strings.HasPrefix(prompt[end:], imageMarker) {
				return nil, nil, fmt.Errorf("image %d is missing from the rendered prompt", i)
			}
			segments = append(segments, model.Segment{Tokens: m.Tokenizer().Encode(prompt[start:end], false)})
			segments = append(segments, model.Segment{Kind: "image", Data: image})
			start = end + len(imageMarker)
		}
		segments = append(segments, model.Segment{Tokens: m.Tokenizer().Encode(prompt[start:], false)})
		prepared, err = m.PrepareMedia(segments)
	}
	if err != nil {
		return nil, nil, err
	}
	var markers []int32
	for i, token := range prepared.Tokens {
		if token == m.markerID {
			markers = append(markers, int32(i))
		}
	}
	if len(markers) != len(rows) {
		return nil, nil, fmt.Errorf("Perplexity readout markers do not match questions")
	}
	return prepared, markers, nil
}

type scoreChunk struct {
	rows     []llm.ScoreRow
	prepared *model.PreparedRequest
	markers  []int32
}

// A request shares one noncausal sequence. Split only when its complete
// sequence exceeds the context limit, keeping the state and images in each part.
func (m *Model) planScore(input llm.ScoreRequest) ([]scoreChunk, error) {
	var chunks []scoreChunk
	var split func([]llm.ScoreRow) error
	split = func(rows []llm.ScoreRow) error {
		prepared, markers, err := m.prepareJoint(input, rows)
		if err != nil {
			return err
		}
		if len(prepared.Tokens) <= input.MaxTokens {
			chunks = append(chunks, scoreChunk{rows: rows, prepared: prepared, markers: markers})
			return nil
		}
		if len(rows) == 1 {
			return fmt.Errorf("question needs %d tokens with state and images; limit is %d", len(prepared.Tokens), input.MaxTokens)
		}
		middle := len(rows) / 2
		if err := split(rows[:middle]); err != nil {
			return err
		}
		return split(rows[middle:])
	}
	if err := split(input.Rows); err != nil {
		return nil, err
	}
	return chunks, nil
}

func (m *Model) Score(ctx context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	cached := 0
	result.CachedTokens = &cached
	badRequest := func(err error) (llm.ScoreResponse, error) {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if input.System == "" || len(input.Rows) < 1 || len(input.Rows) > 64 || input.MaxTokens < 1 || input.MaxTokens > m.MaxContextLength() {
		return badRequest(fmt.Errorf("invalid Perplexity system prompt, question count, or context limit"))
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	chunks, err := m.planScore(input)
	if err != nil {
		return badRequest(err)
	}
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		var logits []float32
		mlx.Scoped(func() {
			prepared := chunk.prepared
			media := make([]batch.MediaItem, len(prepared.Items))
			for i := range prepared.Items {
				item := &prepared.Items[i]
				data := mlx.FromValues(item.MediaData, item.Dims...)
				media[i] = batch.MediaItem{Pos: item.Range[0], Opaque: item.Opaque, Features: m.EncodeMedia(item, data)}
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
			readouts := hidden.TakeAxis(mlx.FromValues(chunk.markers, len(chunk.markers)), 1)
			output := mlx.DivScalar(m.Readout.Forward(readouts).AsType(mlx.DTypeFloat32), m.config.Temperature)
			output = output.Reshape(len(chunk.rows) * len(m.config.Codes))
			mlx.Eval(output)
			logits = output.Floats()
		})
		for i, row := range chunk.rows {
			start := i * len(m.config.Codes)
			count := len(row.Question.Options)
			result.Logits = append(result.Logits, logits[start:start+count])
		}
		result.InputTokens += len(chunk.prepared.Tokens)
	}
	return result, ctx.Err()
}
