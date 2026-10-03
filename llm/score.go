package llm

import (
	"context"

	"github.com/ollama/ollama/api"
)

// Scorer is an optional runner capability for bounded decision scoring.
// Inputs are already encoded; no chat template or generation options apply.
type Scorer interface {
	Score(context.Context, ScoreRequest) (ScoreResponse, error)
}

// ScoreRequest carries prompt rows or model-native decision head inputs.
type ScoreRequest struct {
	PointerRows   []ScorePointerRow `json:"pointer_rows,omitempty"`
	State         string            `json:"state,omitempty"`
	Rows          []ScoreRow        `json:"rows,omitempty"`
	Segments      []string          `json:"segments,omitempty"`
	Fields        []ScoreField      `json:"fields,omitempty"`
	MaxTokens     int               `json:"max_tokens"`
	Images        []api.ImageData   `json:"images,omitempty"`
	ImagePosition int               `json:"image_position,omitempty"` // Segment boundary at which images are inserted.
}

type ScoreRow struct {
	Prompt     string         `json:"prompt"`
	Candidates []string       `json:"candidates"`
	Question   *ScoreQuestion `json:"question,omitempty"`
}

// ScoreQuestion preserves a field's schema for model-native decision heads.
type ScoreQuestion struct {
	Type         string   `json:"type"`
	Instructions string   `json:"instructions"`
	Options      []string `json:"options"`
}

type ScoreResponse struct {
	// Logits may be log probabilities: a shared offset within a row does not
	// affect the softmax over its candidates.
	Logits       [][]float32 `json:"logits"`
	InputTokens  int         `json:"input_tokens"`            // Sum of complete prompt lengths, including shared tokens.
	OutputTokens int         `json:"output_tokens,omitempty"` // Tokens generated internally to obtain candidate scores.
	CachedTokens *int        `json:"cached_tokens,omitempty"` // Subset of InputTokens restored instead of evaluated; nil when unavailable.
}

// ScoreField identifies a question and its allowed options for a decision head.
// Spans index ScoreRequest.Segments; the runner maps them to token offsets.
// Type is 0 for noul, 1 for choice, and 2 for score.
type ScoreField struct {
	Type     int      `json:"type"`
	Question [2]int   `json:"question"`
	Options  [][2]int `json:"options"`
}

// ScorePointerRow is an independent state/question pair for a pointer head.
// Options are byte spans in Prompt; Type is 0 for noul, 1 for choice, 2 for score.
type ScorePointerRow struct {
	Prefix  string   `json:"prefix"`
	Prompt  string   `json:"prompt"`
	Type    int      `json:"type"`
	Options [][2]int `json:"options"`
}
