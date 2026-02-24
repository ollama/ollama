//go:build mlx

package tokenizer

import (
	"runtime"
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
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	input := "a<tag>xb"
	want := []string{"a", "<tag>x", "b"}

	got := tok.splitBySpecialTokens(input)
	if len(got) != len(want) {
		t.Fatalf("split length mismatch: got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("split mismatch at %d: got %v want %v", i, got, want)
		}
	}
}

func TestSplitBySpecialTokensFallbackWithoutCache(t *testing.T) {
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
		t.Fatalf("failed to load tokenizer: %v", err)
	}

	input := "a<tag>xb"
	want := []string{"a", "<tag>x", "b"}

	// Simulate construction outside loader path where cache is not set.
	tok.sortedSpecialTokens = nil

	got := tok.splitBySpecialTokens(input)
	if len(got) != len(want) {
		t.Fatalf("split length mismatch: got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("split mismatch at %d: got %v want %v", i, got, want)
		}
	}
}

func TestEncodeDeterministicAcrossGOMAXPROCS(t *testing.T) {
	tok := benchmarkLoadMiniLlama(t)

	input := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 640)

	prev := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(prev)

	runtime.GOMAXPROCS(1)
	seq := tok.Encode(input, false)

	if prev < 2 {
		runtime.GOMAXPROCS(2)
	} else {
		runtime.GOMAXPROCS(prev)
	}
	par := tok.Encode(input, false)

	if !equalIDs(seq, par) {
		t.Fatalf("encode mismatch between sequential and parallel paths: seq=%d par=%d", len(seq), len(par))
	}
}
