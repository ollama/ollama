package llm

import (
	"context"

	"github.com/ollama/ollama/decision"
)

// Scorer is an optional runner capability for bounded decision scoring.
// Inputs are already encoded; no chat template or generation options apply.
type Scorer interface {
	Score(context.Context, ScoreRequest) (ScoreResponse, error)
}

// DecisionRunner uses a model's native System One template and readout.
type DecisionRunner interface {
	SystemOne(context.Context, decision.Request) (decision.Response, error)
}

type (
	ScoreRequest    = decision.ScoreRequest
	ScoreRow        = decision.ScoreRow
	ScoreQuestion   = decision.ScoreQuestion
	ScoreField      = decision.ScoreField
	ScorePointerRow = decision.ScorePointerRow
	ScoreResponse   = decision.ScoreResponse
)
