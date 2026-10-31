package mlxrunner

import (
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

// TestStopMatcherRandomized streams random tokenizations of random text through
// matchers with random stop sets and checks the output against a reference
// computed on the whole text: everything before the earliest stop is sent in
// order, nothing from the stop on is, and every response's logprobs go out
// exactly once, attached to a prefix of that response's text.
func TestStopMatcherRandomized(t *testing.T) {
	alphabet := []string{"a", "b", "c", "\n", " ", "あ", "い", "【", "】", "S", "T", "O", "P"}
	rng := rand.New(rand.NewPCG(1, 2))

	for iter := 0; iter < 20000; iter++ {
		// Build the text and cut it into pieces (tokens) at random byte
		// positions on rune boundaries.
		var sb strings.Builder
		for n := rng.IntN(40); n > 0; n-- {
			sb.WriteString(alphabet[rng.IntN(len(alphabet))])
		}
		text := sb.String()
		var pieces []string
		for rest := text; rest != ""; {
			runes := []rune(rest)
			k := 1 + rng.IntN(min(len(runes), 6))
			pieces = append(pieces, string(runes[:k]))
			rest = string(runes[k:])
		}

		var stops []string
		for n := rng.IntN(4); n > 0; n-- {
			var s strings.Builder
			for m := rng.IntN(5); m > 0; m-- {
				s.WriteString(alphabet[rng.IntN(len(alphabet))])
			}
			stops = append(stops, s.String())
		}

		// Reference: earliest occurrence of any non-empty stop.
		wantMatched := false
		want := text
		for _, stop := range stops {
			if stop == "" {
				continue
			}
			if i := strings.Index(text, stop); i >= 0 && (!wantMatched || i < len(want)) {
				want = text[:i]
				wantMatched = true
			}
		}

		s := newStopMatcher(stops)
		var out []CompletionResponse
		matched := false
		for i, piece := range pieces {
			lp := []llm.Logprob{{TokenLogprob: llm.TokenLogprob{Token: piece, Logprob: float64(i)}}}
			ready, ok := s.push(CompletionResponse{Content: piece, Logprobs: lp})
			out = append(out, ready...)
			if ok {
				matched = true
				break
			}
		}
		if !matched {
			out = append(out, s.flush()...)
		}

		var got strings.Builder
		seen := map[int]bool{}
		for _, r := range out {
			got.WriteString(r.Content)
			if len(r.Logprobs) == 0 {
				t.Fatalf("iter %d: response %q has no logprobs\npieces=%q stops=%q", iter, r.Content, pieces, stops)
			}
			idx := int(r.Logprobs[0].Logprob)
			if seen[idx] {
				t.Fatalf("iter %d: logprobs of piece %d emitted twice\npieces=%q stops=%q", iter, idx, pieces, stops)
			}
			seen[idx] = true
			if !strings.HasPrefix(pieces[idx], r.Content) {
				t.Fatalf("iter %d: response %q is not a prefix of its piece %q\npieces=%q stops=%q", iter, r.Content, pieces[idx], pieces, stops)
			}
		}
		if got.String() != want || matched != wantMatched {
			t.Fatalf("iter %d: got %q matched=%v, want %q matched=%v\npieces=%q stops=%q",
				iter, got.String(), matched, want, wantMatched, pieces, stops)
		}
		// Without a match every piece's logprobs must have gone out.
		if !matched && len(seen) != len(pieces) {
			t.Fatalf("iter %d: %d of %d pieces' logprobs emitted\npieces=%q stops=%q", iter, len(seen), len(pieces), pieces, stops)
		}
	}
}
