package mlxrunner

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/llm"
)

// streamThrough pushes each piece through a matcher built from stops and
// returns the contents it released, whether a stop matched, and what flush
// released afterwards.
func streamThrough(stops []string, pieces ...string) (sent []string, matched bool, flushed []string) {
	s := newStopMatcher(stops)
	for _, piece := range pieces {
		ready, ok := s.push(CompletionResponse{Content: piece})
		for _, r := range ready {
			sent = append(sent, r.Content)
		}
		if ok {
			matched = true
			break
		}
	}
	for _, r := range s.flush() {
		flushed = append(flushed, r.Content)
	}
	return sent, matched, flushed
}

func TestStopMatcher(t *testing.T) {
	cases := []struct {
		name    string
		stops   []string
		pieces  []string
		sent    []string
		matched bool
		flushed []string
	}{
		{name: "no stops passes through", pieces: []string{"a", "b"}, sent: []string{"a", "b"}},
		{name: "empty stop is ignored", stops: []string{""}, pieces: []string{"a", "b"}, sent: []string{"a", "b"}},
		{name: "stop inside one piece", stops: []string{"STOP"}, pieces: []string{"Hello STOP world"}, sent: []string{"Hello "}, matched: true},
		{name: "stop split across pieces", stops: []string{"STOP"}, pieces: []string{"Hel", "lo ST", "OP more"}, sent: []string{"Hel", "lo "}, matched: true},
		{name: "stop at the very start", stops: []string{"STOP"}, pieces: []string{"ST", "OP"}, matched: true},
		{name: "false start is released", stops: []string{"STOP"}, pieces: []string{"S", "Sun"}, sent: []string{"S", "Sun"}},
		{name: "false start is released whole", stops: []string{"STOP"}, pieces: []string{"go ST", "ART"}, sent: []string{"go ST", "ART"}},
		{name: "partial match is flushed whole at the end", stops: []string{"STOP"}, pieces: []string{"abc ST"}, flushed: []string{"abc ST"}},
		{name: "earliest of several stops wins", stops: []string{"world", "lo"}, pieces: []string{"hello world"}, sent: []string{"hel"}, matched: true},
		{name: "multibyte stop split across pieces", stops: []string{"【入力確定】"}, pieces: []string{"考え【入", "力確定】続き"}, sent: []string{"考え"}, matched: true},
		{name: "text after the stop is dropped", stops: []string{"\n\n"}, pieces: []string{"one\n", "\ntwo", "three"}, sent: []string{"one"}, matched: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sent, matched, flushed := streamThrough(tc.stops, tc.pieces...)
			if !slices.Equal(sent, tc.sent) || matched != tc.matched || !slices.Equal(flushed, tc.flushed) {
				t.Fatalf("got sent=%q matched=%v flushed=%q, want sent=%q matched=%v flushed=%q",
					sent, matched, flushed, tc.sent, tc.matched, tc.flushed)
			}
		})
	}
}

func TestStopMatcherKeepsLogprobsWithTheirText(t *testing.T) {
	logprobs := []llm.Logprob{{TokenLogprob: llm.TokenLogprob{Token: "go ST"}}}

	t.Run("cut at a match", func(t *testing.T) {
		s := newStopMatcher([]string{"STOP"})
		if ready, matched := s.push(CompletionResponse{Content: "go ST", Logprobs: logprobs}); matched || len(ready) != 0 {
			t.Fatalf("first push = %+v matched=%v, want \"go ST\" held", ready, matched)
		}
		ready, matched := s.push(CompletionResponse{Content: "OP"})
		if !matched || len(ready) != 1 || ready[0].Content != "go " || len(ready[0].Logprobs) != 1 {
			t.Fatalf("second push = %+v matched=%v, want \"go \" carrying the logprobs", ready, matched)
		}
	})

	t.Run("flushed without a match", func(t *testing.T) {
		s := newStopMatcher([]string{"STOP"})
		s.push(CompletionResponse{Content: "go ST", Logprobs: logprobs})
		flushed := s.flush()
		if len(flushed) != 1 || flushed[0].Content != "go ST" || len(flushed[0].Logprobs) != 1 {
			t.Fatalf("flush = %+v, want \"go ST\" whole with its logprobs", flushed)
		}
	})
}
