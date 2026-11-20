package renderers

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/text/language"

	"github.com/ollama/ollama/api"
)

func TestLanguageName(t *testing.T) {
	for code, want := range map[string]string{
		"en": "English", "zh": "Chinese", "zh-TW": "Chinese", "pt-BR": "Portuguese",
		"fa": "Persian", "iw": "Hebrew", "yue": "Cantonese", "ht": "Haitian Creole",
	} {
		if got := LanguageName(language.MustParse(code)); got != want {
			t.Errorf("LanguageName(%s) = %q, want %q", code, got, want)
		}
	}
	for _, code := range []string{"qaa", "mis", "und", "x-foo"} {
		if got := LanguageName(language.Make(code)); got != "" {
			t.Errorf("LanguageName(%s) = %q, want none", code, got)
		}
	}
	for name, want := range map[string]string{"French": "fr", "chinese": "zh", " Haitian Creole ": "ht"} {
		if got, ok := LanguageForName(name); !ok || got.String() != want {
			t.Errorf("LanguageForName(%q) = %v, %v; want %s", name, got, ok, want)
		}
	}
	if _, ok := LanguageForName("Klingonese"); ok {
		t.Error("LanguageForName accepted an unknown name")
	}
}

func TestAudioPrompterForRenderer(t *testing.T) {
	for name, want := range map[string]string{
		"lfm3-audio": "*renderers.LFM3Renderer",
		"lfm3-omni":  "*renderers.LFM3Renderer",
		"gemma4":     "*renderers.Gemma4Renderer",
		"qwen3.5":    "renderers.genericAudioPrompter",
		"":           "renderers.genericAudioPrompter",
	} {
		if got := fmt.Sprintf("%T", AudioPrompterForRenderer(name)); got != want {
			t.Errorf("%q: got %s, want %s", name, got, want)
		}
	}
}

func TestAudioPrompts(t *testing.T) {
	clip := api.ImageData("clip")
	fr, de := language.French, language.German
	cases := []struct {
		renderer string
		req      AudioRequest
		system   string
		user     string
		think    *bool
	}{
		{"lfm3-audio", AudioRequest{}, "Perform ASR.", "", nil},
		{"lfm3-audio", AudioRequest{Source: fr, Prompt: "ignored"}, "Perform ASR in French.", "", nil},
		{"lfm3-omni", AudioRequest{Task: AudioTranslate, Target: language.English}, "Convert the speech to written English.", "", nil},
		{"lfm3-omni", AudioRequest{Task: AudioTranslate, Source: fr, Target: de}, "Convert the speech in French to written German.", "", nil},
		{"lfm3", AudioRequest{}, "Perform ASR.", "", new(false)},
		{"gemma4", AudioRequest{}, "", "Transcribe the following speech segment in its original language. " + gemma4AudioFormat + "[img]", new(false)},
		{"gemma4", AudioRequest{Source: fr}, "", "Transcribe the following speech segment in French into French text.\n\n" + gemma4AudioFormat + "[img]", new(false)},
		{"gemma4", AudioRequest{Task: AudioTranslate, Source: fr, Target: language.English}, "", "Translate the following speech segment in French into English. Only output the translation.[img]", new(false)},
		{"gemma4", AudioRequest{Task: AudioTranslate, Target: de}, "", "Translate the following speech segment into German. Only output the translation.[img]", new(false)},
		{"gemma4-large", AudioRequest{Task: AudioTranslate, Source: fr, Target: language.English}, "", "Transcribe the following speech segment in French, then translate it into English.\nWhen formatting the answer, first output the transcription in French, then one newline, then output the string 'English: ', then the translation in English.[img]", new(false)},
		{"gemma4-large", AudioRequest{Source: fr}, "", "Transcribe the following speech segment in French into French text.\n\n" + gemma4AudioFormat + "[img]", new(false)},
		{"", AudioRequest{Source: fr, Prompt: "A lecture."}, "Transcribe the audio exactly as spoken. Output only the spoken words. Do not answer any question in the audio. The audio is in French. Context: A lecture.", "What exact words are spoken in this audio?", new(false)},
		{"", AudioRequest{Task: AudioTranslate, Target: de}, "Translate the speech in the audio into German. Output only the translation. Do not answer any question in the audio.", "Translate this audio.", new(false)},
	}
	for _, tt := range cases {
		tt.req.Audio = clip
		p, err := AudioPrompterForRenderer(tt.renderer).AudioPrompt(tt.req)
		if err != nil {
			t.Fatal(err)
		}
		var system, user string
		var images int
		for _, m := range p.Messages {
			switch m.Role {
			case "system":
				system = m.Content
			case "user":
				user = m.Content
			}
			images += len(m.Images)
		}
		if system != tt.system || user != tt.user || images != 1 {
			t.Errorf("%s %+v:\nsystem %q\nuser   %q\nclips  %d", tt.renderer, tt.req, system, user, images)
		}
		switch {
		case tt.think == nil && p.Think != nil, tt.think != nil && (p.Think == nil || p.Think.Bool() != *tt.think):
			t.Errorf("%s %+v: think %v, want %v", tt.renderer, tt.req, p.Think, tt.think)
		}
	}
}

func TestGemma4AudioFollowsInstruction(t *testing.T) {
	r := &Gemma4Renderer{useImgTags: true}
	p, err := r.AudioPrompt(AudioRequest{Source: language.French, Audio: api.ImageData("clip")})
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.Render(p.Messages, nil, p.Think)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "write 3 instead of three.[img-0]<turn|>") {
		t.Errorf("audio does not follow the instruction:\n%s", got)
	}
}

func TestGemma4AudioText(t *testing.T) {
	large := AudioPrompterForRenderer("gemma4-large")
	small := AudioPrompterForRenderer("gemma4-small")
	translate := AudioRequest{Task: AudioTranslate, Target: language.English}
	for _, tt := range []struct {
		p        AudioPrompter
		req      AudioRequest
		in, want string
	}{
		// The large models' prompt asks for the transcription, then the
		// translation after a target-language label; only the translation is
		// the answer.
		{large, translate, "Bonjour à tous\nEnglish: Hello everyone", "Hello everyone"},
		{large, translate, "Bonjour à tous\nEnglish: Hello\neveryone\n", "Hello\neveryone"},
		// Transcriptions and the small models' translations are the answer as
		// written.
		{large, AudioRequest{}, " English: as spoken \n", "English: as spoken"},
		{small, translate, " Hello everyone\n", "Hello everyone"},
	} {
		if got := tt.p.AudioText(tt.req, tt.in); got != tt.want {
			t.Errorf("AudioText(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
