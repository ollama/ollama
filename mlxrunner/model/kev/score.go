package kev

import (
	"context"
	"fmt"
	"net/http"
	"regexp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

var _ model.CachedScorer = (*Model)(nil)

// Callers cannot insert the tokens Kev uses to delimit its own fields.
var controlToken = regexp.MustCompile(`<\|([A-Za-z0-9_]+)\|>`)

func userTokens(tok *tokenizer.Tokenizer, value string) []int32 {
	return tok.Encode(controlToken.ReplaceAllString(value, "<¦${1}¦>"), false)
}

type ScoreRow struct {
	Tokens      []int32
	Pointers    []int32
	StateTokens int
}

func (m *Model) PrepareScore(ctx context.Context, input llm.ScoreRequest) ([]ScoreRow, error) {
	badRequest := func(format string, args ...any) ([]ScoreRow, error) {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf(format, args...)}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(input.Rows) < 1 || len(input.Rows) > 64 || len(input.Images) != 0 || input.MaxTokens < 1 || input.MaxTokens > m.MaxContextLength() {
		return badRequest("invalid Kev questions, images, or context limit")
	}
	tok := m.Tokenizer()
	markers := [5]int32{}
	for i, name := range [...]string{"<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"} {
		var ok bool
		markers[i], ok = tok.GetSpecialToken(name)
		if !ok {
			return badRequest("Kev tokenizer is missing %s", name)
		}
	}
	state := append([]int32{markers[0]}, userTokens(tok, input.State)...)
	if len(state) > m.config.MaxState || len(state) > input.MaxTokens {
		return badRequest("Kev state has %d tokens; limit is %d", len(state), min(m.config.MaxState, input.MaxTokens))
	}
	rows := make([]ScoreRow, len(input.Rows))
	for i, request := range input.Rows {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		q := request.Question
		if q == nil || (q.Type != "noul" && q.Type != "choice" && q.Type != "score") || len(q.Options) < 1 || len(q.Options) > 255 || (q.Type == "noul" && len(q.Options) != 2) {
			return badRequest("question %d has an invalid type or option count", i)
		}
		row := ScoreRow{Tokens: append([]int32(nil), state...), StateTokens: len(state)}
		row.Tokens = append(row.Tokens, markers[1])
		row.Tokens = append(row.Tokens, userTokens(tok, q.Instructions)...)
		for _, option := range q.Options {
			row.Tokens = append(row.Tokens, markers[2])
			row.Tokens = append(row.Tokens, userTokens(tok, option)...)
			row.Pointers = append(row.Pointers, int32(len(row.Tokens)))
			row.Tokens = append(row.Tokens, markers[3])
		}
		row.Tokens = append(row.Tokens, markers[4])
		if len(row.Tokens) > m.config.MaxRow || len(row.Tokens) > input.MaxTokens {
			return badRequest("question %d needs %d tokens; row limit is %d", i, len(row.Tokens), min(m.config.MaxRow, input.MaxTokens))
		}
		rows[i] = row
	}
	return rows, nil
}

// FinishScore reads the closing delimiter of each option and the final query.
// It runs on the MLX thread, after the complete causal row has been evaluated.
func (m *Model) FinishScore(row ScoreRow, hidden *mlx.Array) []float32 {
	options := hidden.TakeAxis(mlx.FromValues(row.Pointers, len(row.Pointers)), 1)
	last := len(row.Tokens) - 1
	query := hidden.Slice(mlx.Slice(), mlx.Slice(last, last+1), mlx.Slice())
	logits := m.pointer(query, options).Reshape(-1)
	mlx.Eval(logits)
	return logits.Floats()
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
			hidden, restored, err = forward(ctx, &model.PreparedRequest{Tokens: row.Tokens, ScorePrefix: row.StateTokens}, nil)
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
