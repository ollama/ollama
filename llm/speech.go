package llm

import (
	"context"

	"github.com/ollama/ollama/model/renderers"
)

// SpeechRequest asks a speech model to read Input aloud.
type SpeechRequest struct {
	Input string `json:"input"`
	Voice string `json:"voice,omitempty"`
	// Prompter maps the request onto llama-server's /tts; runners with their
	// own speech models ignore it.
	Prompter renderers.SpeechPrompter `json:"-"`
}

// SpeechSampleRate is the rate of the PCM a Speaker streams.
const SpeechSampleRate = 24000

// Speaker is implemented by runners that serve speech models. Speech streams
// SpeechSampleRate mono 16-bit little-endian PCM to fn as it is generated.
type Speaker interface {
	Speech(ctx context.Context, req SpeechRequest, fn func(pcm []byte) error) error
}
