package decision

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/orderedmap"
)

// Questions preserves request order, which determines the order of scored fields.
// Its zero value is ready to use with Set.
type Questions = api.SystemOneQuestions

type Request = api.SystemOneRequest
type Question = api.SystemOneQuestion

// Choice maps a single-token answer code to a value in the question's schema.
type Choice struct {
	Code        string `json:"code"`
	Value       any    `json:"value"`
	Description any    `json:"description"`
}

// Field is a question's schema as presented to the model.
type Field struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Choices     []Choice `json:"choices"`
}

type (
	Answers       = orderedmap.Map[string, any]
	Probabilities = orderedmap.Map[string, float64]
	Legend        = orderedmap.Map[string, any]
)

type Response struct {
	Model   string   `json:"model"`
	Answers *Answers `json:"answers"`
	Usage   Usage    `json:"usage"`
}

type Usage = api.SystemOneUsage

type NoulAnswer struct {
	Type string  `json:"type"`
	Noul float64 `json:"noul"`
}

type ChoiceAnswer struct {
	Type          string         `json:"type"`
	Choice        string         `json:"choice"`
	Probabilities *Probabilities `json:"probabilities"`
	Confidence    float64        `json:"confidence"`
}

type ScoreAnswer struct {
	Type          string         `json:"type"`
	Score         float64        `json:"score"`
	Legend        *Legend        `json:"legend"`
	Probabilities *Probabilities `json:"probabilities"`
	Confidence    float64        `json:"confidence"`
}
