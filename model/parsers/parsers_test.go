package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

type mockParser struct {
	name string
}

func (m *mockParser) Init(tools []api.Tool, lastMessage *api.Message) []api.Tool {
	return tools
}

func (m *mockParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	return "mock:" + s, "", nil, nil
}

func (m *mockParser) HasToolSupport() bool {
	return false
}

func (m *mockParser) HasThinkingSupport() bool {
	return false
}

func TestRegisterCustomParser(t *testing.T) {
	// Register a custom parser
	Register("custom-parser", func() Parser {
		return &mockParser{name: "custom"}
	})

	// Retrieve it
	parser := ParserForName("custom-parser")
	if parser == nil {
		t.Fatal("expected parser to be registered")
	}

	// Test it works
	content, _, _, err := parser.Add("test", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "mock:test" {
		t.Errorf("expected 'mock:test', got %q", content)
	}
}

func TestBuiltInParsersStillWork(t *testing.T) {
	tests := []struct {
		name string
	}{
		{"passthrough"},
		{"qwen3-coder"},
		{"harmony"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := ParserForName(tt.name)
			if parser == nil {
				t.Fatalf("expected built-in parser %q to exist", tt.name)
			}
		})
	}
}

func TestOverrideBuiltInParser(t *testing.T) {
	// Override a built-in parser
	Register("passthrough", func() Parser {
		return &mockParser{name: "override"}
	})

	// Should get the override
	parser := ParserForName("passthrough")
	if parser == nil {
		t.Fatal("expected parser to exist")
	}

	// Test it's the override
	content, _, _, err := parser.Add("test", false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "mock:test" {
		t.Errorf("expected 'mock:test' from override, got %q", content)
	}
}

func TestUnknownParserReturnsNil(t *testing.T) {
	parser := ParserForName("nonexistent-parser")
	if parser != nil {
		t.Error("expected nil for unknown parser")
	}
}
