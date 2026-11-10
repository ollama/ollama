package model

import (
	"context"

	"github.com/ollama/ollama/llm"
)

// SpeechModel batches generation over independently owned speech requests.
// All methods run on the MLX worker. DecodeSpeech shares model computation
// across the supplied states; output rows correspond to input rows.
type SpeechModel interface {
	NewSpeech(context.Context, llm.SpeechRequest) (SpeechState, error)
	DecodeSpeech([]SpeechState) ([]SpeechOutput, error)
}

// SpeechState owns one request's caches, sampling history and codec stream.
// Close runs on the MLX worker after completion, cancellation or failure.
type SpeechState interface {
	Close()
}

type SpeechOutput struct {
	Samples []float32 // CPU-owned 24 kHz mono audio.
	Done    bool
	Err     error // A request-local failure; peers may continue.
}
