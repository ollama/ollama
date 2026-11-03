package strands

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

var _ model.CachedScorer = (*Model)(nil)

// ScoreRow is one independent question. Pointers refer to real option tokens;
// the query is the final real token, even when the forward row is padded.
type ScoreRow struct {
	Tokens   []int32
	Pointers []int32
	Type     int
}

// PrepareScore needs no MLX thread or array scope and can overlap GPU execution.
func (m *Model) PrepareScore(ctx context.Context, input llm.ScoreRequest) ([]ScoreRow, error) {
	if len(input.PointerRows) < 1 || len(input.PointerRows) > 64 || len(input.Images) != 0 || input.MaxTokens < 1 || input.MaxTokens > m.MaxContextLength() {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "invalid Strands Decider rows, images, or context limit"}
	}
	rows := make([]ScoreRow, len(input.PointerRows))
	for i, row := range input.PointerRows {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		encoded, err := encode(m.Tokenizer(), row, input.MaxTokens)
		if err != nil {
			return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf("question %d: %v", i, err)}
		}
		rows[i] = encoded
	}
	return rows, nil
}

func encode(tok *tokenizer.Tokenizer, row llm.ScorePointerRow, limit int) (ScoreRow, error) {
	result := ScoreRow{Type: row.Type}
	count := len(row.Options)
	if row.Type < 0 || row.Type > 2 || count < 2 || count > 255 || (row.Type == 0 && count != 2) || (row.Type == 2 && count > 10) {
		return result, fmt.Errorf("invalid pointer question type or option count")
	}
	prefix := tok.Encode(row.Prefix, true)
	suffix := tok.Encode(row.Prompt, false)
	if len(suffix) == 0 || len(prefix)+len(suffix) > limit {
		return result, fmt.Errorf("decision input exceeds the %d-token context or has no question", limit)
	}
	// Tokenize the entire question once. Encoding options separately changes BPE
	// boundaries, and the head reads the last token wholly inside each option.
	ends := make([]int, len(suffix))
	var decoded strings.Builder
	for i, id := range suffix {
		decoded.WriteString(tok.Decode([]int32{id}))
		ends[i] = decoded.Len()
	}
	if decoded.String() != row.Prompt {
		return result, fmt.Errorf("cannot map decision option spans to tokenizer bytes")
	}
	previous := 0
	for _, span := range row.Options {
		if span[0] < previous || span[0] >= span[1] || span[1] > len(row.Prompt) {
			return result, fmt.Errorf("invalid decision option span")
		}
		pointer, start := -1, 0
		for j, end := range ends {
			if start >= span[0] && end <= span[1] && end > start {
				pointer = j
			}
			start = end
		}
		if pointer < 0 {
			return result, fmt.Errorf("decision option contains no complete token")
		}
		result.Pointers = append(result.Pointers, int32(len(prefix)+pointer))
		previous = span[1]
	}
	result.Tokens = append(prefix, suffix...)
	return result, nil
}

// FinishScore reads the option pointers and final query from one complete row.
// Call on the MLX thread within the caller's scope; padded tail states are ignored.
func (m *Model) FinishScore(row ScoreRow, hidden *mlx.Array) []float32 {
	options := hidden.TakeAxis(mlx.FromValues(row.Pointers, len(row.Pointers)), 1)
	n := len(row.Tokens)
	query := hidden.Slice(mlx.Slice(), mlx.Slice(n-1, n), mlx.Slice())
	output := m.pointer(query, options, row.Type).Reshape(-1)
	mlx.Eval(output)
	return output.Floats()
}

func (m *Model) Score(ctx context.Context, input llm.ScoreRequest, forward model.ScoreForward) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	cached := 0
	result.CachedTokens = &cached
	rows, err := m.PrepareScore(ctx, input)
	if err != nil {
		return result, err
	}
	for _, row := range rows {
		mlx.Scoped(func() {
			var hidden *mlx.Array
			var restored int
			hidden, restored, err = forward(ctx, &model.PreparedRequest{Tokens: row.Tokens}, nil)
			cached += restored
			if err == nil {
				result.Logits = append(result.Logits, m.FinishScore(row, hidden))
			}
		})
		if err != nil {
			return result, err
		}
		result.InputTokens += len(row.Tokens)
	}
	return result, ctx.Err()
}
