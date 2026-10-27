package lfm3speech

import (
	"slices"
	"strings"
)

// Languages maps voice-name language suffixes to the names the publisher's
// TTS prompts use; English defaults to American English.
var Languages = map[string]string{
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

// VoiceNames lists the voices to advertise: the default, then a female and a
// male voice for each language under its full name ("male-british-english").
func VoiceNames() []string {
	var languages []string
	for _, name := range Languages {
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
