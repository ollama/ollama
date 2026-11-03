package model

import (
	"context"

	"github.com/ollama/ollama/llm"
)

// SpeechModel generates speech instead of text. The runner loads it like any
// model and serves it with Speak, which emits 24 kHz mono samples.
type SpeechModel interface {
	Speak(ctx context.Context, req llm.SpeechRequest, emit func(samples []float32) error) error
}
