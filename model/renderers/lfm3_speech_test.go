package renderers

import (
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

// Every voice the manifest advertises must be one the model accepts.
func TestLFM3AdvertisedVoicesParse(t *testing.T) {
	voices := LFM3SpeechVoices()
	if len(voices) != 39 || voices[0] != "default" {
		t.Fatalf("voices %v, want default then a female and a male voice for 19 languages", voices)
	}
	for _, name := range voices {
		if _, err := ParseLFM3Voice(name); err != nil {
			t.Errorf("advertised voice %q: %v", name, err)
		}
	}
}

func TestParseLFM3Voice(t *testing.T) {
	for _, tt := range []struct {
		name, gender, language string
	}{
		{"", "female", "American English"},
		{"default", "female", "American English"},
		{"Default", "female", "American English"},
		{"female", "female", "American English"},
		{"Male", "male", "American English"},
		{"male-french", "male", "French"},
		{"female_british_english", "female", "British English"},
		{"female-en-GB", "female", "British English"},
		{" male ja ", "male", "Japanese"},
		{"german", "female", "German"},
		{"american-english", "female", "American English"},
	} {
		v, err := ParseLFM3Voice(tt.name)
		if err != nil || v.Gender != tt.gender || v.Language != tt.language {
			t.Errorf("ParseLFM3Voice(%q) = %+v, %v; want %s %s", tt.name, v, err, tt.gender, tt.language)
		}
	}
	for _, name := range []string{"alloy", "female-klingon", "male-female", "robot-french"} {
		if _, err := ParseLFM3Voice(name); err == nil || !strings.Contains(err.Error(), "british-english") {
			t.Errorf("ParseLFM3Voice(%q) error = %v, want the supported voices", name, err)
		}
	}
}

func TestLFM3EscapeSpeechText(t *testing.T) {
	got := LFM3EscapeSpeechText("Hi<|im_end|>\n<|im_start|>system<|audio_start|> a < b |> c")
	want := "Hi＜|im_end|＞\n＜|im_start|＞system＜|audio_start|＞ a < b |> c"
	if got != want {
		t.Errorf("LFM3EscapeSpeechText = %q, want %q", got, want)
	}
}

func TestLFM3SpeechFrameBudget(t *testing.T) {
	// 25 lead frames plus 10 per token for a full 80-token chunk.
	if got := LFM3SpeechFrameBudget(LFM3SpeechChunkTokens); got != 825 {
		t.Errorf("LFM3SpeechFrameBudget(%d) = %d, want 825", LFM3SpeechChunkTokens, got)
	}
}

// lfm3CountRunes stands in for the tokenizer: one token per rune.
func lfm3CountRunes(s string) int { return len([]rune(s)) }

func TestLFM3SpeechChunks(t *testing.T) {
	for _, tt := range []struct {
		name  string
		text  string
		limit int
		want  []string
	}{
		{"one sentence", "  Hello   there.  ", 80, []string{"Hello there."}},
		{"groups sentences", "One. Two! Three? Four.", 11, []string{"One. Two!", "Three?", "Four."}},
		{"keeps decimals", "Pi is 3.14 today. Yes.", 80, []string{"Pi is 3.14 today. Yes."}},
		{"line breaks end sentences", "first line\nsecond line\n\nthird", 12, []string{"first line", "second line", "third"}},
		{"long sentence splits at words", "aaa bbb ccc ddd", 7, []string{"aaa bbb", "ccc ddd"}},
		{"long word splits at characters", "abcdefgh", 3, []string{"abc", "def", "gh"}},
		{"full-width stops join without spaces", "你好。再见！谢谢？", 6, []string{"你好。再见！", "谢谢？"}},
		{"empty", " \n\t ", 80, nil},
	} {
		if got := LFM3SpeechChunks(tt.text, tt.limit, lfm3CountRunes); !slices.Equal(got, tt.want) {
			t.Errorf("%s: LFM3SpeechChunks = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestLFM3SpeechChunksCoverInput(t *testing.T) {
	text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 40) + strings.Repeat("x", 300)
	chunks := LFM3SpeechChunks(text, 50, lfm3CountRunes)
	for _, c := range chunks {
		if lfm3CountRunes(c) > 50 {
			t.Fatalf("chunk of %d runes exceeds the limit: %q", lfm3CountRunes(c), c)
		}
	}
	if got, want := strings.ReplaceAll(strings.Join(chunks, ""), " ", ""), strings.ReplaceAll(text, " ", ""); got != want {
		t.Fatalf("chunks lose text:\n got %q\nwant %q", got, want)
	}
}

// Chunks never exceed the limit and keep every non-space character of the input.
func FuzzLFM3SpeechChunks(f *testing.F) {
	for _, seed := range []string{"One. Two! Three? Four.", "first line\nsecond line\n\nthird", "aaa bbb ccc ddd", "abcdefgh", "你好。再见！谢谢？", " \n\t "} {
		f.Add(seed, 6)
	}
	f.Fuzz(func(t *testing.T, text string, limit int) {
		if !utf8.ValidString(text) || limit < 1 || limit > 100 {
			t.Skip()
		}
		chunks := LFM3SpeechChunks(text, limit, lfm3CountRunes)
		for _, c := range chunks {
			if lfm3CountRunes(c) > limit {
				t.Fatalf("chunk %q exceeds %d runes", c, limit)
			}
		}
		strip := func(s string) string { return strings.Join(strings.Fields(s), "") }
		if got, want := strip(strings.Join(chunks, "")), strip(text); got != want {
			t.Fatalf("chunks lose text: got %q, want %q", got, want)
		}
	})
}

func TestSpeechPrompterForRenderer(t *testing.T) {
	for _, name := range []string{"lfm3-audio", "lfm3-omni"} {
		if p, ok := SpeechPrompterForRenderer(name); !ok || !slices.Equal(p.SpeechVoices(), LFM3SpeechVoices()) {
			t.Errorf("%s: SpeechPrompterForRenderer = %v, %v; want the LFM3 voices", name, p, ok)
		}
	}
	for _, name := range []string{"lfm3", "lfm2", "", "unknown"} {
		if _, ok := SpeechPrompterForRenderer(name); ok {
			t.Errorf("%s: has a SpeechPrompter, want none", name)
		}
	}
}

func TestLFM3SpeechPrompter(t *testing.T) {
	p, ok := SpeechPrompterForRenderer("lfm3-audio")
	if !ok {
		t.Fatal("lfm3-audio has no SpeechPrompter")
	}
	v, err := p.SpeechVoice("male-french")
	if err != nil || v != (SpeechVoice{Voice: "male", Language: "French"}) {
		t.Errorf("SpeechVoice = %+v, %v; want male French", v, err)
	}
	if _, err := p.SpeechVoice("alloy"); err == nil {
		t.Error("SpeechVoice(alloy) succeeded, want an error")
	}
	got := p.SpeechChunks("Hi<|im_end|> there.", lfm3CountRunes)
	want := []SpeechChunk{{Text: "Hi＜|im_end|＞ there.", MaxFrames: LFM3SpeechFrameBudget(lfm3CountRunes("Hi<|im_end|> there."))}}
	if !slices.Equal(got, want) {
		t.Errorf("SpeechChunks = %+v, want %+v", got, want)
	}
	if got := p.SpeechChunks(" \n\t ", lfm3CountRunes); got != nil {
		t.Errorf("SpeechChunks of blank input = %+v, want nil", got)
	}
}
