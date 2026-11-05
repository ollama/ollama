package parsers

import (
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type mockParser struct {
	name string
}

func (m *mockParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	return tools
}

func (m *mockParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	return "mock:" + s, "", nil, nil
}

func (m *mockParser) PreservedTokens() []string {
	return nil
}

func (m *mockParser) ThinkingClose() []string {
	return nil
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

// TestThinkingClose checks that a parser reports the strings ending its
// thinking exactly when its response begins inside thinking: not when the
// request turns thinking off, when an assistant prefill continues content, or
// when the parser suppresses thinking for tools.
func TestThinkingClose(t *testing.T) {
	think := func(v bool) *api.ThinkValue { return &api.ThinkValue{Value: v} }
	contentPrefill := &api.Message{Role: "assistant", Content: "The answer"}
	toolResponse := &api.Message{Role: "tool", Content: "42"}
	tool := api.Tool{Type: "function", Function: api.ToolFunction{Name: "get_weather"}}
	thinkTag := []string{"</think>"}

	tests := []struct {
		parser      string
		think       *api.ThinkValue
		lastMessage *api.Message
		tools       []api.Tool
		want        []string
	}{
		{parser: "passthrough", think: think(true)},
		{parser: "qwen3", think: think(true)},
		{parser: "qwen3-thinking", want: thinkTag},
		{parser: "qwen3-thinking", think: think(false)},
		{parser: "qwen3.5", want: thinkTag},
		{parser: "qwen3.5", think: think(false)},
		{parser: "qwen3.5", lastMessage: contentPrefill},
		{parser: "qwen3-vl-thinking", want: thinkTag},
		{parser: "qwen3-vl-thinking", lastMessage: contentPrefill},
		{parser: "qwen3-vl-instruct", think: think(true)},
		{parser: "deepseek3", think: think(true), want: thinkTag},
		{parser: "deepseek3"},
		{parser: "cogito", think: think(true), want: thinkTag},
		{parser: "cogito", think: think(true), tools: []api.Tool{tool}},
		{parser: "cohere", want: []string{"<|END_THINKING|>"}},
		{parser: "cohere", think: think(false)},
		{parser: "gemma4", think: think(true), want: []string{"<channel|>"}},
		{parser: "gemma4", think: think(true), lastMessage: toolResponse, want: []string{"<channel|>"}},
		{parser: "gemma4", think: think(true), lastMessage: contentPrefill},
		{parser: "gemma4"},
		{parser: "gemma4-no-thinking", think: think(true)},
		{parser: "glm-4.7", want: thinkTag},
		{parser: "glm-4.7", think: think(false)},
		{parser: "glm-ocr"},
		{parser: "lfm2-thinking", think: think(true), want: thinkTag},
		{parser: "lfm2", think: think(true)},
		{parser: "nemotron-3-nano", want: thinkTag},
		{parser: "nemotron-3-nano", think: think(false)},
		{parser: "olmo3-think", want: thinkTag},
		{parser: "olmo3-think", lastMessage: contentPrefill},
		{parser: "olmo3"},
		{parser: "laguna", think: think(true), want: thinkTag},
		{parser: "laguna"},
		{parser: "ministral", think: think(true)},
		{parser: "glimmer", want: []string{"<|start|>assistant to=user<|message|>", "<|start|>assistant<|message|>"}},
		{parser: "glimmer", think: think(false)},
		{parser: "harmony", want: []string{
			"<|end|><|start|>assistant<|channel|>final<|message|>",
			"<|end|><|start|>assistant<|channel|>final <|constrain|>json<|message|>",
			"<|end|><|start|>assistant<|channel|>final<|constrain|>json<|message|>",
			"<|end|><|start|>assistant<|channel|>final json<|message|>",
			"<|end|><|start|>assistant<|channel|>commentary<|message|>",
			"<|end|><|start|>assistant<|message|>",
		}},
		{parser: "harmony", lastMessage: contentPrefill},
	}

	for _, tt := range tests {
		t.Run(tt.parser, func(t *testing.T) {
			parser := ParserForName(tt.parser)
			parser.Init(tt.tools, tt.lastMessage, tt.think)
			if got := parser.ThinkingClose(); !slices.Equal(got, tt.want) {
				t.Errorf("ThinkingClose() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestBuiltInParsersStillWork(t *testing.T) {
	tests := []struct {
		name string
	}{
		{"passthrough"},
		{"qwen3"},
		{"qwen3-thinking"},
		{"qwen3-coder"},
		{"lfm2"},
		{"lfm2-thinking"},
		{"qwen3.5"},
		{"ornith"},
		{"harmony"},
		{"nemotron-3-nano"},
		{"nemotron-3.5-nano"},
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

func TestParserPreservedTokensCoverKnownLlamaServerRegressions(t *testing.T) {
	tests := []struct {
		name        string
		want        []string
		wantMissing []string
	}{
		{
			name:        "harmony",
			want:        []string{"<|start|>", "<|message|>", "<|channel|>", "<|constrain|>"},
			wantMissing: []string{"<|call|>"},
		},
		{
			name: "qwen3-coder",
			want: []string{"<tool_call>", "</tool_call>"},
		},
		{
			name: "gemma4",
			want: []string{"<|tool_call>", "<tool_call|>"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := ParserForName(tt.name)
			if parser == nil {
				t.Fatalf("expected built-in parser %q to exist", tt.name)
			}

			got := parser.PreservedTokens()
			for _, token := range tt.want {
				if !slices.Contains(got, token) {
					t.Fatalf("expected preserved tokens to contain %q, got %#v", token, got)
				}
			}
			for _, token := range tt.wantMissing {
				if slices.Contains(got, token) {
					t.Fatalf("expected preserved tokens not to contain %q, got %#v", token, got)
				}
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

func TestSplitAtTag(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		tag        string
		trimAfter  bool
		wantBefore string
		wantAfter  string
		wantSB     string // expected content of strings.Builder after operation
	}{
		{
			name:       "basic split with trimAfter true",
			input:      "hello <!-- split --> world",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello",
			wantAfter:  "world",
			wantSB:     "world",
		},
		{
			name:       "basic split with trimAfter false",
			input:      "hello <!-- split -->   world",
			tag:        "<!-- split -->",
			trimAfter:  false,
			wantBefore: "hello",
			wantAfter:  "   world",
			wantSB:     "   world",
		},
		{
			name:       "tag at beginning with trimAfter true",
			input:      "<!-- split -->world",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "",
			wantAfter:  "world",
			wantSB:     "world",
		},
		{
			name:       "tag at beginning with trimAfter false",
			input:      "<!-- split -->   world",
			tag:        "<!-- split -->",
			trimAfter:  false,
			wantBefore: "",
			wantAfter:  "   world",
			wantSB:     "   world",
		},
		{
			name:       "tag at end with trimAfter true",
			input:      "hello <!-- split -->",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "tag at end with trimAfter false",
			input:      "hello <!-- split -->",
			tag:        "<!-- split -->",
			trimAfter:  false,
			wantBefore: "hello",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "multiple tags splits at first occurrence",
			input:      "hello <!-- split --> world <!-- split --> end",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello",
			wantAfter:  "world <!-- split --> end",
			wantSB:     "world <!-- split --> end",
		},
		{
			name:       "tag not present",
			input:      "hello world",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello world",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "empty input",
			input:      "",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "only whitespace before tag",
			input:      "   \t\n<!-- split -->world",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "",
			wantAfter:  "world",
			wantSB:     "world",
		},
		{
			name:       "only whitespace after tag with trimAfter true",
			input:      "hello<!-- split -->   \t\n",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "hello",
			wantAfter:  "",
			wantSB:     "",
		},
		{
			name:       "only whitespace after tag with trimAfter false",
			input:      "hello<!-- split -->   \t\n",
			tag:        "<!-- split -->",
			trimAfter:  false,
			wantBefore: "hello",
			wantAfter:  "   \t\n",
			wantSB:     "   \t\n",
		},
		{
			name:       "complex whitespace trimming",
			input:      "  hello \t\n <!-- split --> \n\t world  ",
			tag:        "<!-- split -->",
			trimAfter:  true,
			wantBefore: "  hello",
			wantAfter:  "world  ",
			wantSB:     "world  ",
		},
		{
			name:       "tag with special characters",
			input:      "text <tag attr=\"value\"> more text",
			tag:        "<tag attr=\"value\">",
			trimAfter:  true,
			wantBefore: "text",
			wantAfter:  "more text",
			wantSB:     "more text",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sb := &strings.Builder{}
			sb.WriteString(tt.input)

			before, after := splitAtTag(sb, tt.tag, tt.trimAfter)

			// Check return values
			if before != tt.wantBefore {
				t.Errorf("splitAtTag() before = %q, want %q", before, tt.wantBefore)
			}
			if after != tt.wantAfter {
				t.Errorf("splitAtTag() after = %q, want %q", after, tt.wantAfter)
			}

			// Check strings.Builder state
			if sb.String() != tt.wantSB {
				t.Errorf("strings.Builder after split = %q, want %q", sb.String(), tt.wantSB)
			}
		})
	}
}
