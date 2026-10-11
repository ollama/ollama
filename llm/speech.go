package llm

import "context"

// SpeechRequest asks a speech model to read Input aloud.
type SpeechRequest struct {
	Input string `json:"input"`
	Voice string `json:"voice,omitempty"`
}

// Speaker is implemented by runners that serve speech models. Speech streams
// 24 kHz mono 16-bit little-endian PCM to fn as it is generated.
type Speaker interface {
	Speech(ctx context.Context, req SpeechRequest, fn func(pcm []byte) error) error
}
