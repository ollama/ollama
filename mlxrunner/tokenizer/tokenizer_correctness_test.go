package tokenizer

import (
	"runtime"
	"slices"
	"strings"
	"testing"
)

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestEncodeRoundtripMiniLlama(t *testing.T) {
	tok := benchmarkLoadMiniLlama(t)

	inputs := []string{
		"",
		"hello",
		"hello world",
		" hello  world ",
		"don't we'll they're",
		"1234567890",
		"こんにちは世界",
		"Hello 世界",
		"func main() {}",
		"<|begin_of_text|>system\nYou are concise.<|end_of_text|>",
		strings.Repeat("The quick brown fox jumps over the lazy dog. ", 32),
	}

	for _, input := range inputs {
		ids := tok.Encode(input, false)
		got := tok.Decode(ids)
		if got != input {
			t.Fatalf("roundtrip mismatch for %q: got %q", input, got)
		}
	}
}

func TestSplitBySpecialTokensGreedyLongest(t *testing.T) {
	data := []byte(`{
		"model": {
			"type": "BPE",
			"vocab": {"a": 0, "b": 1},
			"merges": []
		},
		"added_tokens": [
			{"id": 2, "content": "<tag>", "special": true},
			{"id": 3, "content": "<tag>x", "special": true}
		]
	}`)

	tok, err := LoadFromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []struct {
		text string
		want []encodeChunk
	}{
		{"a<tag>xb", []encodeChunk{{text: "a"}, {text: "<tag>x", isSpecial: true}, {text: "b"}}},
		{"<tag>x<tag>", []encodeChunk{{text: "<tag>x", isSpecial: true}, {text: "<tag>", isSpecial: true}}},
		{"a<tag><tag>xb", []encodeChunk{{text: "a"}, {text: "<tag>", isSpecial: true}, {text: "<tag>x", isSpecial: true}, {text: "b"}}},
		{"a<tag", []encodeChunk{{text: "a<tag"}}},
		{"<<tag>x", []encodeChunk{{text: "<"}, {text: "<tag>x", isSpecial: true}}},
		{"<tag><ta", []encodeChunk{{text: "<tag>", isSpecial: true}, {text: "<ta"}}},
		{strings.Repeat("a<tag>x", 800), slices.Repeat([]encodeChunk{{text: "a"}, {text: "<tag>x", isSpecial: true}}, 800)},
		{"", nil},
	} {
		if got := tok.specialTokenMatcher.split(input.text); !slices.Equal(got, input.want) {
			t.Errorf("split(%q) = %v, want %v", input.text, got, input.want)
		}
	}
}

func TestEncodeDeterministicAcrossGOMAXPROCS(t *testing.T) {
	tok := benchmarkLoadMiniLlama(t)

	prev := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(prev)

	for _, tc := range []struct {
		name, text string
	}{
		{"ASCII", strings.Repeat("The quick brown fox jumps over the lazy dog. ", 640)},
		{"Unicode and special tokens", strings.Repeat("e\u0301日本😀<|end_of_text|>", 640)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runtime.GOMAXPROCS(1)
			seq := tok.Encode(tc.text, false)

			runtime.GOMAXPROCS(max(2, prev))
			par := tok.Encode(tc.text, false)

			if !equalIDs(seq, par) {
				t.Fatalf("encode mismatch between sequential and parallel paths: seq=%d par=%d", len(seq), len(par))
			}
		})
	}
}

func FuzzAddedTokenMatcher(f *testing.F) {
	var matcher addedTokenMatcher
	for _, token := range []string{"<tag>", "<tag>x", "éé", "\x00"} {
		matcher.add(token, token)
	}
	for _, input := range []string{"", "plain", "a<tag>xb", "<tag>x<tag>", "<<tag", "aééa", "\xff<tag>\x00"} {
		f.Add(input)
	}
	f.Fuzz(func(t *testing.T, input string) {
		remaining := input
		for _, part := range matcher.split(input) {
			if part.text == "" || !strings.HasPrefix(remaining, part.text) {
				t.Fatalf("split(%q) changed or inserted text: %q", input, part.text)
			}
			remaining = remaining[len(part.text):]
		}
		if remaining != "" {
			t.Fatalf("split(%q) dropped suffix %q", input, remaining)
		}
	})
}
