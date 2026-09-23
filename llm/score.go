package llm

import "context"

// Scorer is an optional runner capability for next-token candidate scoring.
// Prompts are already rendered; no chat template or generation options apply.
type Scorer interface {
	Score(context.Context, ScoreRequest) (ScoreResponse, error)
}

type ScoreRequest struct {
	Rows      []ScoreRow `json:"rows"`
	MaxTokens int        `json:"max_tokens"`
}

type ScoreRow struct {
	Prompt     string   `json:"prompt"`
	Candidates []string `json:"candidates"`
}

type ScoreResponse struct {
	Logits      [][]float32 `json:"logits"`
	InputTokens int         `json:"input_tokens"` // Sum of complete prompt lengths, including shared tokens.
}
