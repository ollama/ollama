package mlxrunner

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestStopBuffer(t *testing.T) {
	for _, tc := range []struct {
		name   string
		stops  []string
		chunks []string
		want   string
		done   bool
	}{
		{"disabled", nil, []string{"hello", " world"}, "hello world", false},
		{"empty stop", []string{""}, []string{"hello"}, "hello", false},
		{"whole chunk", []string{"STOP"}, []string{"hello STOP ignored"}, "hello ", true},
		{"across chunks", []string{"STOP"}, []string{"hello S", "T", "OP ignored"}, "hello ", true},
		{"at start", []string{"STOP"}, []string{"ST", "OP"}, "", true},
		{"earliest match", []string{"later", "first"}, []string{"a first b later"}, "a ", true},
		{"prefix overlap", []string{"STOP", "ST"}, []string{"ST"}, "", true},
		{"false prefix", []string{"STOP"}, []string{"hello ST", "ar"}, "hello STar", false},
		{"end of generation", []string{"STOP"}, []string{"hello ST"}, "hello ST", false},
		{"unicode", []string{"🙂fin"}, []string{"é🙂", "fin no"}, "é", true},
		{"chat boundary", []string{"<|im_end|>", "<|im_start|>"}, []string{"answer<|im_", "end|>fake user"}, "answer", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			buffer := stopBuffer{stops: tc.stops}
			var output strings.Builder
			var done bool
			for _, chunk := range tc.chunks {
				parts, stopped := buffer.add(CompletionResponse{Content: chunk})
				for _, part := range parts {
					output.WriteString(part.Content)
				}
				if stopped {
					done = true
					break
				}
			}
			for _, part := range buffer.flush() {
				output.WriteString(part.Content)
			}
			if output.String() != tc.want || done != tc.done {
				t.Fatalf("got (%q, %v), want (%q, %v)", output.String(), done, tc.want, tc.done)
			}
		})
	}
	// A possible stop prefix must never leak to the caller before resolution.
	buffer := stopBuffer{stops: []string{"STOP"}}
	parts, done := buffer.add(CompletionResponse{Content: "ST", Logprobs: []llm.Logprob{{}}})
	if len(parts) != 0 || done {
		t.Fatal("partial stop was emitted")
	}
	parts, done = buffer.add(CompletionResponse{Content: "ar"})
	if done || len(parts) != 2 || len(parts[0].Logprobs) != 1 {
		t.Fatal("false prefix did not preserve original response chunks and logprobs")
	}
	buffer = stopBuffer{stops: []string{"STOP"}}
	parts, done = buffer.add(CompletionResponse{Content: "hello STOP", Logprobs: []llm.Logprob{{}}})
	if !done || len(parts) != 1 || parts[0].Content != "hello " || len(parts[0].Logprobs) != 0 {
		t.Fatal("a partial token was assigned the original token's logprobs")
	}
}
