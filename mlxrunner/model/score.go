package model

import (
	"context"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
)

// ScoreForward evaluates a causal backbone through the runner's prefix cache,
// returning the complete hidden row and the number of restored tokens. Segments supply
// the original media bytes for cache identity. Call within an MLX scope.
type ScoreForward func(context.Context, *PreparedRequest, []Segment) (*mlx.Array, int, error)

// CachedScorer owns decision preparation and readout; the runner owns forward
// execution and retained state. Bidirectional encoders cannot use this path.
type CachedScorer interface {
	Score(context.Context, llm.ScoreRequest, ScoreForward) (llm.ScoreResponse, error)
}
