package renderers

// SpeechVoice is the voice and language a llama-server /tts request asks for.
type SpeechVoice struct{ Voice, Language string }

// SpeechChunk is one /tts generation of a speech request.
type SpeechChunk struct {
	// Text is read aloud, escaped for the model's tts template.
	Text string
	// MaxFrames caps the codec frames the generation may produce.
	MaxFrames int
}

// SpeechPrompter maps speech requests onto llama-server's /tts for the models
// a renderer serves; llama-server renders the model's own TTS template.
type SpeechPrompter interface {
	// SpeechVoices lists the voice names to advertise; renderers whose models
	// do not speak return none.
	SpeechVoices() []string
	// SpeechVoice resolves a voice name; the client gets a 400 on error.
	SpeechVoice(name string) (SpeechVoice, error)
	// SpeechChunks splits input into generations, or returns nil when there is
	// no text to speak; count returns the token count of a piece of text.
	SpeechChunks(input string, count func(string) int) []SpeechChunk
}

// SpeechPrompterForRenderer returns the named renderer's SpeechPrompter if
// its models read text aloud through llama-server.
func SpeechPrompterForRenderer(name string) (SpeechPrompter, bool) {
	p, ok := rendererForName(name).(SpeechPrompter)
	if !ok || len(p.SpeechVoices()) == 0 {
		return nil, false
	}
	return p, true
}
