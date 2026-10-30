package autotrustdecision

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

var _ model.CachedScorer = (*Model)(nil)

// The publisher's bare-v1 prompt has no chat delimiters or generation suffix.
const imageMarker = "<|vision_start|><|image_pad|><|vision_end|>"

// A request body cap alone still permits many tiny images and prompt markers.
const maxAutoTrustDecisionImages = 64

func choiceLabels(tok *tokenizer.Tokenizer) ([]string, []int32) {
	var codes []string
	var ids []int32
	labels := make([]string, 0, 702)
	for a := 'A'; a <= 'Z'; a++ {
		labels = append(labels, string(a))
	}
	for a := 'A'; a <= 'Z'; a++ {
		for b := 'A'; b <= 'Z'; b++ {
			labels = append(labels, string(a)+string(b))
		}
	}
	for _, code := range labels {
		candidate := tok.Encode(code, false)
		if len(candidate) != 1 || !slices.Contains(tok.Encode("x\n"+code+") y", false), candidate[0]) {
			continue
		}
		codes = append(codes, code)
		ids = append(ids, candidate[0])
		if len(ids) == 255 {
			return codes, ids
		}
	}
	return codes, ids
}

func (m *Model) prompt(input llm.ScoreRequest, q *llm.ScoreQuestion) (string, string, []int32, []float32, error) {
	if q == nil {
		return "", "", nil, nil, fmt.Errorf("missing AutoTrust decision question")
	}
	if len(input.Images) > maxAutoTrustDecisionImages {
		return "", "", nil, nil, fmt.Errorf("AutoTrust decision supports at most %d images", maxAutoTrustDecisionImages)
	}
	var options []string
	var ids []int32
	var bias []float32
	switch q.Type {
	case "noul":
		if len(q.Options) != 2 {
			return "", "", nil, nil, fmt.Errorf("AutoTrust decision noul requires two options")
		}
		options = []string{"false", "true"}
		ids = m.head.VerbalizerIDs[:2]
		bias = m.head.Bias[:2]
	case "score":
		if len(q.Options) != 6 {
			return "", "", nil, nil, fmt.Errorf("AutoTrust decision score requires six levels")
		}
		options = []string{"0", "1", "2", "3", "4", "5"}
		ids = m.head.VerbalizerIDs[2:8]
		bias = m.head.Bias[2:8]
	case "choice":
		if len(q.Options) < 2 || len(q.Options) > len(m.choiceIDs) {
			return "", "", nil, nil, fmt.Errorf("AutoTrust decision choice requires 2–%d options", len(m.choiceIDs))
		}
		options = make([]string, len(q.Options))
		for i, option := range q.Options {
			options[i] = m.choiceCodes[i] + ") " + option
		}
		ids = m.choiceIDs[:len(options)]
		bias = make([]float32, len(options))
		copy(bias, m.head.Bias[8:24])
	default:
		return "", "", nil, nil, fmt.Errorf("unsupported AutoTrust decision question type %q", q.Type)
	}
	prefix := fmt.Sprintf("[kind] %s\n[state] ", q.Type)
	if len(input.Images) > 0 {
		prefix += strings.Repeat(imageMarker, len(input.Images))
	}
	prefix += input.State + "\n[question] "
	full := prefix + q.Instructions + "\n[options]\n" + strings.Join(options, "\n") + "\n[decision]:"
	return full, prefix, ids, bias, nil
}

func (m *Model) prepareRow(input llm.ScoreRequest, q *llm.ScoreQuestion) (*model.PreparedRequest, []model.Segment, []int32, []float32, error) {
	full, prefix, ids, bias, err := m.prompt(input, q)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	tok := m.Tokenizer()
	if len(input.Images) == 0 {
		prepared, err := m.PrepareMedia([]model.Segment{{Tokens: tok.Encode(full, false)}})
		if err != nil {
			return nil, nil, nil, nil, err
		}
		want := tok.Encode(prefix, false)
		for prepared.ScorePrefix < len(want) && prepared.ScorePrefix < len(prepared.Tokens) && want[prepared.ScorePrefix] == prepared.Tokens[prepared.ScorePrefix] {
			prepared.ScorePrefix++
		}
		return prepared, nil, ids, bias, nil
	}
	before := fmt.Sprintf("[kind] %s\n[state] ", q.Type)
	segments := []model.Segment{{Tokens: tok.Encode(before, false)}}
	for _, data := range input.Images {
		segments = append(segments, model.Segment{Kind: "image", Data: data})
	}
	after := full[len(before)+len(input.Images)*len(imageMarker):]
	segments = append(segments, model.Segment{Tokens: tok.Encode(after, false)})
	prepared, err := m.PrepareMedia(segments)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	// The cache still recognizes identical media requests and safe boundaries.
	// The text-only prefix boundary is omitted here because multimodal token
	// expansion changes it and the caller's images are separate from state.
	return prepared, segments, ids, bias, nil
}

func (m *Model) finishScore(hidden *mlx.Array, n int, ids []int32, bias []float32, kind string) []float32 {
	last := hidden.Slice(mlx.Slice(), mlx.Slice(n-1, n), mlx.Slice())
	selected := m.Model.UnembedCandidates(last, mlx.FromValues(ids, len(ids)))
	logits := mlx.DivScalar(mlx.Add(selected, mlx.FromValues(bias, len(bias))), m.temperature[kind])
	mlx.Eval(logits)
	return logits.Floats()
}

func (m *Model) Score(ctx context.Context, input llm.ScoreRequest, forward model.ScoreForward) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	cached := 0
	result.CachedTokens = &cached
	badRequest := func(err error) (llm.ScoreResponse, error) {
		return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if len(input.Rows) < 1 || len(input.Rows) > 64 || input.MaxTokens < 1 || input.MaxTokens > m.MaxContextLength() {
		return badRequest(fmt.Errorf("invalid AutoTrust decision question count or context limit"))
	}
	for i, row := range input.Rows {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		prepared, segments, ids, bias, err := m.prepareRow(input, row.Question)
		if err != nil {
			return badRequest(fmt.Errorf("question %d: %w", i, err))
		}
		if len(prepared.Tokens) > input.MaxTokens {
			return badRequest(fmt.Errorf("question %d needs %d tokens; limit is %d", i, len(prepared.Tokens), input.MaxTokens))
		}
		mlx.Scoped(func() {
			var hidden *mlx.Array
			var restored int
			hidden, restored, err = forward(ctx, prepared, segments)
			cached += restored
			if err == nil {
				result.Logits = append(result.Logits, m.finishScore(hidden, len(prepared.Tokens), ids, bias, row.Question.Type))
			}
		})
		if err != nil {
			return result, err
		}
		result.InputTokens += len(prepared.Tokens)
	}
	return result, ctx.Err()
}
