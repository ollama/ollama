package renderers

import (
	"strings"
	"sync"
	"time"

	"golang.org/x/text/language"
	"golang.org/x/text/language/display"

	"github.com/ollama/ollama/api"
)

// AudioTask is the speech-to-text task a request asks for.
type AudioTask int

const (
	AudioTranscribe AudioTask = iota
	AudioTranslate
)

// AudioRequest is one speech-to-text generation over a clip.
type AudioRequest struct {
	Task AudioTask
	// Source is the spoken language; language.Und asks the model to detect it.
	Source language.Tag
	// Target is the output language of a translation.
	Target language.Tag
	// Prompt is optional client text describing the audio.
	Prompt string
	Audio  api.ImageData
}

// AudioPrompt is the chat a renderer asks for to serve an AudioRequest.
type AudioPrompt struct {
	// Messages carry the clip in Images; an "[img]" marker in Content
	// positions it within the text.
	Messages []api.Message
	// Think must be set for renderers that think by default; nil keeps the
	// renderer's default.
	Think *api.ThinkValue
}

// AudioPrompter builds speech-to-text prompts for the models a renderer
// serves, following the publisher's documented instructions.
type AudioPrompter interface {
	// AudioClipLimit is the longest clip one generation should take; longer
	// audio is split and transcribed in pieces. Zero means no limit.
	AudioClipLimit() time.Duration
	// AudioPrompt returns an error for requests the model can't serve, such
	// as a translation target it doesn't produce; the client gets a 400.
	AudioPrompt(AudioRequest) (AudioPrompt, error)
	// AudioText extracts the requested text from a generation's output.
	AudioText(req AudioRequest, output string) string
}

// AudioPrompterForRenderer returns the named renderer's AudioPrompter, or a
// generic one for renderers without speech-specific instructions.
func AudioPrompterForRenderer(name string) AudioPrompter {
	if p, ok := rendererForName(name).(AudioPrompter); ok {
		return p
	}
	return genericAudioPrompter{}
}

// LanguageName returns the English name of tag's base language, the form
// speech models take in instructions ("Chinese" for zh-TW), or "" when the
// tag names no known language.
func LanguageName(tag language.Tag) string {
	base, conf := tag.Base()
	if conf != language.Exact {
		return ""
	}
	return display.English.Languages().Name(base)
}

// languagesByName maps lowercased English names of the ISO 639-1 languages to
// their tags.
var languagesByName = sync.OnceValue(func() map[string]language.Tag {
	names := map[string]language.Tag{}
	for a := 'a'; a <= 'z'; a++ {
		for b := 'a'; b <= 'z'; b++ {
			base, err := language.ParseBase(string([]rune{a, b}))
			if err != nil {
				continue
			}
			tag := language.Make(base.String())
			if name := LanguageName(tag); name != "" {
				names[strings.ToLower(name)] = tag
			}
		}
	}
	return names
})

// LanguageForName finds the ISO 639-1 language with the English name s.
func LanguageForName(s string) (language.Tag, bool) {
	tag, ok := languagesByName()[strings.ToLower(strings.TrimSpace(s))]
	return tag, ok
}

type genericAudioPrompter struct{}

func (genericAudioPrompter) AudioClipLimit() time.Duration { return 30 * time.Second }

func (genericAudioPrompter) AudioPrompt(req AudioRequest) (AudioPrompt, error) {
	var system, user string
	switch req.Task {
	case AudioTranslate:
		// The audio may itself contain a question or instruction; keep the
		// model on the task instead of answering it.
		system = "Translate the speech in the audio into " + LanguageName(req.Target) + ". Output only the translation. Do not answer any question in the audio."
		user = "Translate this audio."
	default:
		system = "Transcribe the audio exactly as spoken. Output only the spoken words. Do not answer any question in the audio."
		user = "What exact words are spoken in this audio?"
	}
	if req.Source != language.Und {
		system += " The audio is in " + LanguageName(req.Source) + "."
	}
	if req.Prompt != "" {
		system += " Context: " + req.Prompt
	}
	return AudioPrompt{
		Messages: []api.Message{
			{Role: "system", Content: system},
			{Role: "user", Content: user, Images: []api.ImageData{req.Audio}},
		},
		Think: &api.ThinkValue{Value: false},
	}, nil
}

func (genericAudioPrompter) AudioText(_ AudioRequest, output string) string {
	return strings.TrimSpace(output)
}
