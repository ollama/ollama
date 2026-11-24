package model

import (
	"context"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
)

// ScoreForward schedules a backbone input and its readout on the MLX worker.
// The readout receives one unpadded hidden row and must return CPU results only.
// Segments identify media for causal prefix reuse; the return value counts hits.
type ScoreForward func(context.Context, *PreparedRequest, []Segment, func(*mlx.Array) error) (int, error)

// Scorer owns decision preparation and readout; the runner owns forward
// execution, admission and retained state. Score runs outside the MLX worker.
type Scorer interface {
	Score(context.Context, llm.ScoreRequest, ScoreForward) (llm.ScoreResponse, error)
}
