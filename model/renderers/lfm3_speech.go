package renderers

// The text side of LFM3 speech, shared by the MLX and llama.cpp runners: voice
// names, splitting long input, and escaping the text read aloud.

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	// LFM3SpeechChunkTokens bounds one generation; longer input is read in sentence
	// groups of about 20 seconds each.
	LFM3SpeechChunkTokens = 80

	// A chunk may speak for 0.8 s per text token plus 2 s before the frame
	// budget cuts it off; speech runs at about 0.3 s per token.
	lfm3SpeechFramesPerToken = 10
	lfm3SpeechLeadFrames     = 25
)

// LFM3SpeechFrameBudget is the most codec frames one chunk of textTokens may generate.
func LFM3SpeechFrameBudget(textTokens int) int {
	return lfm3SpeechLeadFrames + lfm3SpeechFramesPerToken*textTokens
}

// lfm3SpeechLanguages maps voice-name language suffixes to the names the publisher's
// TTS prompts use; English defaults to American English.
var lfm3SpeechLanguages = map[string]string{
	"english": "American English", "en": "American English",
	"american-english": "American English", "en-us": "American English",
	"british-english": "British English", "en-gb": "British English",
	"arabic": "Arabic", "ar": "Arabic",
	"chinese": "Chinese", "zh": "Chinese",
	"danish": "Danish", "da": "Danish",
	"dutch": "Dutch", "nl": "Dutch",
	"finnish": "Finnish", "fi": "Finnish",
	"french": "French", "fr": "French",
	"german": "German", "de": "German",
	"greek": "Greek", "el": "Greek",
	"hebrew": "Hebrew", "he": "Hebrew",
	"italian": "Italian", "it": "Italian",
	"japanese": "Japanese", "ja": "Japanese",
	"korean": "Korean", "ko": "Korean",
	"polish": "Polish", "pl": "Polish",
	"portuguese": "Portuguese", "pt": "Portuguese",
	"spanish": "Spanish", "es": "Spanish",
	"swedish": "Swedish", "sv": "Swedish",
	"turkish": "Turkish", "tr": "Turkish",
}

// LFM3SpeechVoices lists the voices to advertise: the default, then a female and a
// male voice for each language under its full name ("male-british-english").
func LFM3SpeechVoices() []string {
	var languages []string
	for _, name := range lfm3SpeechLanguages {
		key := strings.ToLower(strings.ReplaceAll(name, " ", "-"))
		if !slices.Contains(languages, key) {
			languages = append(languages, key)
		}
	}
	slices.Sort(languages)
	voices := []string{"default"}
	for _, language := range languages {
		voices = append(voices, "female-"+language, "male-"+language)
	}
	return voices
}

// LFM3Voice is the speaker the TTS prompt asks for.
type LFM3Voice struct{ Gender, Language string }

// ParseLFM3Voice reads "female" or "male", optionally followed by a language
// ("male-french", "female-en-gb"), or a language alone. The default, also
// named "default" as with other speech models, is a female American English
// voice.
func ParseLFM3Voice(name string) (LFM3Voice, error) {
	v := LFM3Voice{Gender: "female", Language: "American English"}
	key := strings.ToLower(strings.Join(strings.FieldsFunc(name, func(r rune) bool {
		return unicode.IsSpace(r) || r == '_' || r == '-'
	}), "-"))
	if key == "" || key == "default" {
		return v, nil
	}
	gender, rest, _ := strings.Cut(key, "-")
	switch gender {
	case "female", "male":
		v.Gender = gender
	default:
		rest = key
	}
	if rest != "" {
		language, ok := lfm3SpeechLanguages[rest]
		if !ok {
			return v, fmt.Errorf("voice %q is not supported; use female or male, optionally followed by a language such as female-british-english or male-french (languages: %s)", name, strings.Join(lfm3SpeechLanguageNames(), ", "))
		}
		v.Language = language
	}
	return v, nil
}

func lfm3SpeechLanguageNames() []string {
	var names []string
	for key := range lfm3SpeechLanguages {
		if len(key) > 2 && !strings.HasPrefix(key, "en-") {
			names = append(names, key)
		}
	}
	slices.Sort(names)
	return names
}

var lfm3ControlToken = regexp.MustCompile(`<\|[^<>]+\|>`)

// LFM3EscapeSpeechText swaps the brackets of control-token spellings for full-width
// ones, so text to read aloud cannot end its turn or switch modes.
func LFM3EscapeSpeechText(text string) string {
	return lfm3ControlToken.ReplaceAllStringFunc(text, func(token string) string {
		return "＜" + token[1:len(token)-1] + "＞"
	})
}

// LFM3SpeechChunks splits text into groups of whole sentences of at most limit tokens;
// sentences over the limit split between words, and words between characters.
func LFM3SpeechChunks(text string, limit int, count func(string) int) []string {
	var units []string
	for _, sentence := range lfm3Sentences(text) {
		if count(sentence) <= limit {
			units = append(units, sentence)
			continue
		}
		var words []string
		for _, word := range strings.Fields(sentence) {
			if count(word) <= limit {
				words = append(words, word)
				continue
			}
			var chars []string
			for _, r := range word {
				chars = append(chars, string(r))
			}
			words = append(words, lfm3SpeechPack(chars, limit, count, "")...)
		}
		units = append(units, lfm3SpeechPack(words, limit, count, " ")...)
	}
	return lfm3SpeechPack(units, limit, count, " ")
}

// lfm3SpeechPack joins consecutive units while the result fits the limit.
func lfm3SpeechPack(units []string, limit int, count func(string) int, sep string) []string {
	var out []string
	var cur string
	for _, u := range units {
		next := u
		if cur != "" {
			join := sep
			if last, _ := utf8.DecodeLastRuneInString(cur); lfm3FullWidthStop(last) {
				join = ""
			}
			next = cur + join + u
			if count(next) > limit {
				out = append(out, cur)
				next = u
			}
		}
		cur = next
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

// lfm3Sentences splits text at line breaks and after sentence-ending punctuation,
// collapsing other whitespace.
func lfm3Sentences(text string) []string {
	var out []string
	var cur []rune
	flush := func() {
		if s := strings.Join(strings.Fields(string(cur)), " "); s != "" {
			out = append(out, s)
		}
		cur = cur[:0]
	}
	runes := []rune(text)
	for i, r := range runes {
		if r == '\n' || r == '\r' {
			flush()
			continue
		}
		cur = append(cur, r)
		if lfm3FullWidthStop(r) || (strings.ContainsRune(".!?…", r) && (i+1 == len(runes) || unicode.IsSpace(runes[i+1]))) {
			flush()
		}
	}
	flush()
	return out
}

func lfm3FullWidthStop(r rune) bool { return strings.ContainsRune("。！？", r) }

// LFM3-Audio and LFM3-Omni GGUFs carry a "tts" chat template that llama-server
// renders from a gender and a full language name.

func (r *LFM3Renderer) SpeechVoices() []string {
	if !r.speech {
		return nil
	}
	return LFM3SpeechVoices()
}

func (*LFM3Renderer) SpeechVoice(name string) (SpeechVoice, error) {
	v, err := ParseLFM3Voice(name)
	return SpeechVoice{Voice: v.Gender, Language: v.Language}, err
}

func (*LFM3Renderer) SpeechChunks(input string, count func(string) int) []SpeechChunk {
	var chunks []SpeechChunk
	for _, text := range LFM3SpeechChunks(input, LFM3SpeechChunkTokens, count) {
		chunks = append(chunks, SpeechChunk{Text: LFM3EscapeSpeechText(text), MaxFrames: LFM3SpeechFrameBudget(count(text))})
	}
	return chunks
}
