package server

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
)

func TestThinkingCloseWhitespaceForCompletion(t *testing.T) {
	parser := parsers.ParserForName("harmony")
	parser.Init(nil, nil, nil)
	want := []string{"<|start|>", "assistant", "final", "commentary", "<|constrain|>", "json"}
	if got := thinkingCloseWhitespaceForCompletion(parser, false); !slices.Equal(got, want) {
		t.Fatalf("whitespace boundaries = %q, want %q", got, want)
	}
	if got := thinkingCloseWhitespaceForCompletion(parser, true); got != nil {
		t.Fatalf("raw request has whitespace boundaries: %q", got)
	}
	parser.Init(nil, &api.Message{Role: "assistant", Content: "prefilled content"}, nil)
	if got := thinkingCloseWhitespaceForCompletion(parser, false); got != nil {
		t.Fatalf("content prefill has whitespace boundaries: %q", got)
	}
	if got := thinkingCloseWhitespaceForCompletion(parsers.ParserForName("qwen3"), false); got != nil {
		t.Fatalf("literal thinking parser has whitespace boundaries: %q", got)
	}
	if got := thinkingCloseWhitespaceForCompletion(nil, false); got != nil {
		t.Fatalf("absent parser has whitespace boundaries: %q", got)
	}
}
