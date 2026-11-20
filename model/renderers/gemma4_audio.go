package renderers

import (
	"strings"
	"time"

	"golang.org/x/text/language"

	"github.com/ollama/ollama/api"
)

// Gemma 4 takes up to 30 s of audio, placed after the instruction in a
// single user turn.
func (*Gemma4Renderer) AudioClipLimit() time.Duration { return 30 * time.Second }

const gemma4AudioFormat = "Follow these specific instructions for formatting the answer:\n" +
	"* Only output the transcription, with no newlines.\n" +
	"* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."

func (r *Gemma4Renderer) AudioPrompt(req AudioRequest) (AudioPrompt, error) {
	source := "its original language"
	if req.Source != language.Und {
		source = LanguageName(req.Source)
	}
	target := LanguageName(req.Target)

	// Client prompt text is not used: the model card documents no context
	// field. The model card's transcribe-then-translate format is followed by
	// the large models only; the small ones stop after the transcription, so
	// they get a plain translation instruction of our own.
	var text string
	switch {
	case req.Task == AudioTranslate && r.transcribeThenTranslate:
		text = "Transcribe the following speech segment in " + source + ", then translate it into " + target + ".\n" +
			"When formatting the answer, first output the transcription in " + source + ", then one newline, " +
			"then output the string '" + target + ": ', then the translation in " + target + "."
	case req.Task == AudioTranslate && req.Source == language.Und:
		text = "Translate the following speech segment into " + target + ". Only output the translation."
	case req.Task == AudioTranslate:
		text = "Translate the following speech segment in " + source + " into " + target + ". Only output the translation."
	case req.Source == language.Und:
		text = "Transcribe the following speech segment in its original language. " + gemma4AudioFormat
	default:
		text = "Transcribe the following speech segment in " + source + " into " + source + " text.\n\n" + gemma4AudioFormat
	}
	return AudioPrompt{
		Messages: []api.Message{{Role: "user", Content: text + "[img]", Images: []api.ImageData{req.Audio}}},
		Think:    &api.ThinkValue{Value: false},
	}, nil
}

// AudioText keeps only the translation from a transcribe-then-translate
// answer.
func (r *Gemma4Renderer) AudioText(req AudioRequest, output string) string {
	if req.Task != AudioTranslate || !r.transcribeThenTranslate {
		return strings.TrimSpace(output)
	}
	// Without the requested label the answer's shape is unknown, so it is
	// returned whole rather than guessed at.
	label := LanguageName(req.Target) + ": "
	if i := strings.LastIndex(output, label); i >= 0 {
		output = output[i+len(label):]
	}
	return strings.TrimSpace(output)
}
