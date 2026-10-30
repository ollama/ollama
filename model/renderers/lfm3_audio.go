package renderers

import (
	"strings"
	"time"

	"golang.org/x/text/language"

	"github.com/ollama/ollama/api"
)

// The publisher documents no clip limit; 30 s matches the other speech models
// and keeps each generation short.
func (*LFM3Renderer) AudioClipLimit() time.Duration { return 30 * time.Second }

// AudioPrompt follows the publisher's recipes: the instruction is the system
// turn and the user turn holds only the audio. Client prompt text is left
// out; any user text pulls the model toward describing or answering the clip.
func (r *LFM3Renderer) AudioPrompt(req AudioRequest) (AudioPrompt, error) {
	var system string
	switch {
	case req.Task == AudioTranslate && req.Source == language.Und:
		system = "Convert the speech to written " + LanguageName(req.Target) + "."
	case req.Task == AudioTranslate:
		system = "Convert the speech in " + LanguageName(req.Source) + " to written " + LanguageName(req.Target) + "."
	case req.Source == language.Und:
		system = "Perform ASR."
	default:
		system = "Perform ASR in " + LanguageName(req.Source) + "."
	}
	prompt := AudioPrompt{Messages: []api.Message{
		{Role: "system", Content: system},
		{Role: "user", Images: []api.ImageData{req.Audio}},
	}}
	// Audio and Omni don't think; any other variant must not reason here.
	if r.Thinking() != nil {
		prompt.Think = &api.ThinkValue{Value: false}
	}
	return prompt, nil
}

func (*LFM3Renderer) AudioText(_ AudioRequest, output string) string {
	return strings.TrimSpace(output)
}
