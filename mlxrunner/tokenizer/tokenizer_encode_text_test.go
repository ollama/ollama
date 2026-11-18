package tokenizer

import (
	"slices"
	"testing"
)

func TestEncodeTextMatchesNoAddedTokens(t *testing.T) {
	tok := benchmarkLoadMiniLlama(t)
	const special = "<|end_of_text|>"
	id, ok := tok.GetSpecialToken(special)
	if !ok {
		t.Fatalf("%s is not a special token", special)
	}
	text := "A user wrote " + special + " in a note."
	if !slices.Contains(tok.Encode(text, false), id) {
		t.Fatalf("Encode did not match %s", special)
	}
	plain := tok.EncodeText(text)
	if slices.Contains(plain, id) {
		t.Fatalf("EncodeText matched %s: %v", special, plain)
	}
	if got := tok.Decode(plain); got != text {
		t.Fatalf("EncodeText round trip = %q, want %q", got, text)
	}
	for _, s := range []string{"", "plain words, no control tokens", "café 東京"} {
		if want, got := tok.Encode(s, false), tok.EncodeText(s); !slices.Equal(got, want) {
			t.Fatalf("EncodeText(%q) = %v, want Encode's %v", s, got, want)
		}
	}
}
