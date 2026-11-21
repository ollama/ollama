package llm

import (
	"context"
	"encoding/json"

	"github.com/ollama/ollama/api"
)

// Scorer is an optional runner capability for bounded decision scoring.
// Inputs are already encoded; no chat template or generation options apply.
type Scorer interface {
	Score(context.Context, ScoreRequest) (ScoreResponse, error)
}

// ScoreRequest carries prompt rows or model-native decision head inputs.
type ScoreRequest struct {
	// System is the Modelfile instruction for scorers that assemble a native prompt.
	// Generic candidate rows already include it in their rendered prompts.
	System        string            `json:"system,omitempty"`
	PointerRows   []ScorePointerRow `json:"pointer_rows,omitempty"`
	State         string            `json:"state,omitempty"`
	Rows          []ScoreRow        `json:"rows,omitempty"`
	Segments      []string          `json:"segments,omitempty"`
	Fields        []ScoreField      `json:"fields,omitempty"`
	MaxTokens     int               `json:"max_tokens"`
	Images        []api.ImageData   `json:"images,omitempty"`
	ImagePosition int               `json:"image_position,omitempty"` // Segment boundary at which images are inserted.
	// Native carries a whole System One request to a model that renders and
	// answers it itself; Images are its media in order.
	Native *NativeDecision `json:"native,omitempty"`
}

// NativeDecision is a System One request as its caller wrote it.
type NativeDecision struct {
	State     json.RawMessage  `json:"state,omitempty"`
	Questions []NativeQuestion `json:"questions"`
}

// NativeQuestion keeps one question's JSON in request order.
type NativeQuestion struct {
	Name     string          `json:"name"`
	Question json.RawMessage `json:"question"`
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
	// Answers are complete System One answers, one per NativeDecision question.
	Answers []json.RawMessage `json:"answers,omitempty"`
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
