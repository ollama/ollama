package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestMinistralParserStreaming(t *testing.T) {
	type step struct {
		input      string
		wantEvents []ministralEvent
	}

	cases := []struct {
		desc  string
		tools []api.Tool
		steps []step
		think bool // whether to enable thinking support
	}{
		// Content streaming
		{
			desc: "simple content",
			steps: []step{
				{input: "Hello, how can I help you?", wantEvents: []ministralEvent{
					ministralEventContent{content: "Hello, how can I help you?"},
				}},
			},
		},
		{
			desc: "streaming content word by word",
			steps: []step{
				{input: "Hello,", wantEvents: []ministralEvent{ministralEventContent{content: "Hello,"}}},
				{input: " how", wantEvents: []ministralEvent{ministralEventContent{content: " how"}}},
				{input: " can I help?", wantEvents: []ministralEvent{ministralEventContent{content: " can I help?"}}},
			},
		},

		// Simple tool calls
		{
			desc:  "simple tool call",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "get_weather"}}},
			steps: []step{
				{input: `[TOOL_CALLS]get_weather[ARGS]{"location": "San Francisco"}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "get_weather", args: `{"location": "San Francisco"}`},
				}},
			},
		},
		{
			desc:  "tool call with nested object",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "create_entities"}}},
			steps: []step{
				{input: `[TOOL_CALLS]create_entities[ARGS]{"entities": [{"entityType": "Person", "name": "Jack", "observations": ["Works as a baker"]}]}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "create_entities", args: `{"entities": [{"entityType": "Person", "name": "Jack", "observations": ["Works as a baker"]}]}`},
				}},
			},
		},
		{
			desc:  "tool call with deeply nested objects",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "update_config"}}},
			steps: []step{
				{input: `[TOOL_CALLS]update_config[ARGS]{"settings": {"user": {"profile": {"name": "John", "age": 30}}, "theme": "dark"}}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "update_config", args: `{"settings": {"user": {"profile": {"name": "John", "age": 30}}, "theme": "dark"}}`},
				}},
			},
		},
		{
			desc:  "tool call with array of objects",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "process_items"}}},
			steps: []step{
				{input: `[TOOL_CALLS]process_items[ARGS]{"items": [{"id": 1}, {"id": 2}, {"id": 3}]}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "process_items", args: `{"items": [{"id": 1}, {"id": 2}, {"id": 3}]}`},
				}},
			},
		},
		{
			desc:  "tool call with escaped quotes in string",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "search"}}},
			steps: []step{
				{input: `[TOOL_CALLS]search[ARGS]{"query": "say \"hello\""}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "search", args: `{"query": "say \"hello\""}`},
				}},
			},
		},
		{
			desc:  "tool call with braces inside string",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "format"}}},
			steps: []step{
				{input: `[TOOL_CALLS]format[ARGS]{"template": "Hello {name}!"}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "format", args: `{"template": "Hello {name}!"}`},
				}},
			},
		},
		{
			desc:  "empty JSON object",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "no_args"}}},
			steps: []step{
				{input: `[TOOL_CALLS]no_args[ARGS]{}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "no_args", args: `{}`},
				}},
			},
		},
		{
			desc:  "JSON with newlines in string",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "write"}}},
			steps: []step{
				{input: `[TOOL_CALLS]write[ARGS]{"content": "line1\nline2\nline3"}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "write", args: `{"content": "line1\nline2\nline3"}`},
				}},
			},
		},
		{
			desc:  "backslash in string value",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "path"}}},
			steps: []step{
				{input: `[TOOL_CALLS]path[ARGS]{"dir": "C:\\Users\\test"}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "path", args: `{"dir": "C:\\Users\\test"}`},
				}},
			},
		},

		// Content after tool call
		{
			desc:  "content after tool call",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "test"}}},
			steps: []step{
				// NOTE: It's unclear if this is valid Ministral output, but the parser
				// currently treats text after a tool call as regular content. This test
				// documents that behavior so we notice if it changes.
				{input: `[TOOL_CALLS]test[ARGS]{"a": 1}some content after`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "test", args: `{"a": 1}`},
					ministralEventContent{content: "some content after"},
				}},
			},
		},

		// Multiple tool calls
		{
			desc: "multiple tool calls in sequence",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "get_weather"}},
				{Function: api.ToolFunction{Name: "get_time"}},
			},
			steps: []step{
				{input: `[TOOL_CALLS]get_weather[ARGS]{"location": "NYC"}[TOOL_CALLS]get_time[ARGS]{"timezone": "EST"}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "get_weather", args: `{"location": "NYC"}`},
					ministralEventToolCall{name: "get_time", args: `{"timezone": "EST"}`},
				}},
			},
		},
		{
			desc: "multiple tool calls streamed separately",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "tool_a"}},
				{Function: api.ToolFunction{Name: "tool_b"}},
			},
			steps: []step{
				{input: `[TOOL_CALLS]tool_a[ARGS]{"x": 1}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "tool_a", args: `{"x": 1}`},
				}},
				{input: `[TOOL_CALLS]tool_b[ARGS]{"y": 2}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "tool_b", args: `{"y": 2}`},
				}},
			},
		},

		// Streaming tool calls
		{
			desc:  "streaming tool call with nested objects",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "create_entities"}}},
			steps: []step{
				{input: "[TOOL_CALLS]create_entities[ARGS]", wantEvents: []ministralEvent{}},
				{input: `{"entities": [{"entityType": "Person",`, wantEvents: []ministralEvent{}},
				{input: ` "name": "Jack",`, wantEvents: []ministralEvent{}},
				{input: ` "observations": ["Works`, wantEvents: []ministralEvent{}},
				{input: ` as a baker"]}`, wantEvents: []ministralEvent{}},
				{input: `]}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "create_entities", args: `{"entities": [{"entityType": "Person", "name": "Jack", "observations": ["Works as a baker"]}]}`},
				}},
			},
		},
		{
			desc:  "streaming with incomplete JSON waits for completion",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "test"}}},
			steps: []step{
				{input: "[TOOL_CALLS]test[ARGS]{", wantEvents: []ministralEvent{}},
				{input: `"a": {`, wantEvents: []ministralEvent{}},
				{input: `"b": 1`, wantEvents: []ministralEvent{}},
				{input: `}`, wantEvents: []ministralEvent{}},
				{input: `}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "test", args: `{"a": {"b": 1}}`},
				}},
			},
		},

		// Partial tag handling
		{
			desc: "partial tool tag fakeout",
			steps: []step{
				{input: "abc[TOOL", wantEvents: []ministralEvent{ministralEventContent{content: "abc"}}},
				{input: " not a tag", wantEvents: []ministralEvent{ministralEventContent{content: "[TOOL not a tag"}}},
			},
		},
		{
			desc:  "tool call tag split across chunks",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "test"}}},
			steps: []step{
				{input: "[TOOL_", wantEvents: []ministralEvent{}},
				{input: "CALLS]test[ARGS]{}", wantEvents: []ministralEvent{
					ministralEventToolCall{name: "test", args: `{}`},
				}},
			},
		},
		{
			desc:  "content before tool call",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "get_weather"}}},
			steps: []step{
				{input: "hello [TOOL_CALLS]get_weather[ARGS]{}", wantEvents: []ministralEvent{
					ministralEventContent{content: "hello"},
					ministralEventToolCall{name: "get_weather", args: `{}`},
				}},
			},
		},
		{
			desc:  "whitespace between content and tool call is trimmed",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "test"}}},
			steps: []step{
				{input: "content \n [TOOL_CALLS]test[ARGS]{}", wantEvents: []ministralEvent{
					ministralEventContent{content: "content"},
					ministralEventToolCall{name: "test", args: `{}`},
				}},
			},
		},
		{
			desc:  "tabs and newlines before tool call are trimmed",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "test"}}},
			steps: []step{
				{input: "content\t\n\t[TOOL_CALLS]test[ARGS]{}", wantEvents: []ministralEvent{
					ministralEventContent{content: "content"},
					ministralEventToolCall{name: "test", args: `{}`},
				}},
			},
		},
		{
			desc:  "non-breaking space before tool call is trimmed",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "test"}}},
			steps: []step{
				// \u00a0 is non-breaking space, which unicode.IsSpace considers whitespace
				{input: "content\u00a0[TOOL_CALLS]test[ARGS]{}", wantEvents: []ministralEvent{
					ministralEventContent{content: "content"},
					ministralEventToolCall{name: "test", args: `{}`},
				}},
			},
		},
		{
			desc: "whitespace before THINK tag is trimmed",
			steps: []step{
				{input: "content \n [THINK]thinking[/THINK]after", wantEvents: []ministralEvent{
					ministralEventContent{content: "content"},
					ministralEventThinking{thinking: "thinking"},
					ministralEventContent{content: "after"},
				}},
			},
		},
		{
			desc: "trailing whitespace withheld then emitted",
			steps: []step{
				{input: "Hello ", wantEvents: []ministralEvent{ministralEventContent{content: "Hello"}}},
				{input: "world", wantEvents: []ministralEvent{ministralEventContent{content: " world"}}},
			},
		},
		{
			desc: "trailing newline withheld then emitted",
			steps: []step{
				{input: "Hello\n", wantEvents: []ministralEvent{ministralEventContent{content: "Hello"}}},
				{input: "world", wantEvents: []ministralEvent{ministralEventContent{content: "\nworld"}}},
			},
		},

		// Thinking support
		{
			desc:  "thinking content",
			think: true,
			steps: []step{
				{input: "thinking here[/THINK]", wantEvents: []ministralEvent{
					ministralEventThinking{thinking: "thinking here"},
				}},
				{input: "content after", wantEvents: []ministralEvent{
					ministralEventContent{content: "content after"},
				}},
			},
		},
		{
			desc:  "thinking with whitespace after end tag",
			think: true,
			steps: []step{
				{input: "my thoughts[/THINK]  \n  response", wantEvents: []ministralEvent{
					ministralEventThinking{thinking: "my thoughts"},
					ministralEventContent{content: "response"},
				}},
			},
		},
		{
			desc:  "non-breaking space after think end tag is trimmed",
			think: true,
			steps: []step{
				// \u00a0 is non-breaking space
				{input: "thinking[/THINK]\u00a0response", wantEvents: []ministralEvent{
					ministralEventThinking{thinking: "thinking"},
					ministralEventContent{content: "response"},
				}},
			},
		},
		{
			desc:  "partial think end tag",
			think: true,
			steps: []step{
				{input: "thinking[/THI", wantEvents: []ministralEvent{ministralEventThinking{thinking: "thinking"}}},
				{input: "NK]after", wantEvents: []ministralEvent{ministralEventContent{content: "after"}}},
			},
		},
		{
			desc:  "think tag fakeout",
			think: true,
			steps: []step{
				{input: "thinking[/THI", wantEvents: []ministralEvent{ministralEventThinking{thinking: "thinking"}}},
				{input: "not end tag", wantEvents: []ministralEvent{ministralEventThinking{thinking: "[/THInot end tag"}}},
			},
		},
		{
			desc:  "thinking then tool call",
			think: true,
			tools: []api.Tool{{Function: api.ToolFunction{Name: "test"}}},
			steps: []step{
				{input: "let me think[/THINK][TOOL_CALLS]test[ARGS]{}", wantEvents: []ministralEvent{
					ministralEventThinking{thinking: "let me think"},
					ministralEventToolCall{name: "test", args: `{}`},
				}},
			},
		},

		// Content then THINK tag transition
		{
			desc: "content then think tag",
			steps: []step{
				{input: "content[THINK]thinking[/THINK]more", wantEvents: []ministralEvent{
					ministralEventContent{content: "content"},
					ministralEventThinking{thinking: "thinking"},
					ministralEventContent{content: "more"},
				}},
			},
		},

		// Unicode handling
		{
			desc: "unicode content",
			steps: []step{
				{input: "你好 🌍 مرحبا", wantEvents: []ministralEvent{
					ministralEventContent{content: "你好 🌍 مرحبا"},
				}},
			},
		},
		{
			desc:  "unicode in tool args",
			tools: []api.Tool{{Function: api.ToolFunction{Name: "greet"}}},
			steps: []step{
				{input: `[TOOL_CALLS]greet[ARGS]{"message": "你好 🌍"}`, wantEvents: []ministralEvent{
					ministralEventToolCall{name: "greet", args: `{"message": "你好 🌍"}`},
				}},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			parser := MinistralParser{}
			parser.hasThinkingSupport = tc.think
			parser.Init(tc.tools, nil, nil)

			for i, step := range tc.steps {
				parser.buffer.WriteString(step.input)
				gotEvents := parser.parseEvents()

				if len(gotEvents) == 0 && len(step.wantEvents) == 0 {
					// avoid deep equal on empty vs. nil slices
					continue
				}

				if !reflect.DeepEqual(gotEvents, step.wantEvents) {
					t.Errorf("step %d: input %q: got events %#v, want %#v", i, step.input, gotEvents, step.wantEvents)
				}
			}
		})
	}
}

func TestMinistralParser_Errors(t *testing.T) {
	t.Run("unknown tool returns error", func(t *testing.T) {
		p := &MinistralParser{}
		p.Init([]api.Tool{{Function: api.ToolFunction{Name: "known_tool"}}}, nil, nil)

		_, _, _, err := p.Add(`[TOOL_CALLS]unknown_tool[ARGS]{"a": 1}`, true)
		if err == nil {
			t.Fatal("expected error for unknown tool")
		}
	})

	t.Run("invalid JSON returns error", func(t *testing.T) {
		p := &MinistralParser{}
		p.Init([]api.Tool{{Function: api.ToolFunction{Name: "test"}}}, nil, nil)

		_, _, _, err := p.Add(`[TOOL_CALLS]test[ARGS]{invalid json}`, true)
		if err == nil {
			t.Fatal("expected error for invalid JSON")
		}
	})
}

func TestFindJSONEnd(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected int
	}{
		{
			name:     "simple object",
			input:    `{"a": 1}`,
			expected: 7,
		},
		{
			name:     "nested object",
			input:    `{"a": {"b": 2}}`,
			expected: 14,
		},
		{
			name:     "array inside object",
			input:    `{"items": [1, 2, 3]}`,
			expected: 19,
		},
		{
			name:     "braces in string",
			input:    `{"template": "Hello {name}!"}`,
			expected: 28,
		},
		{
			name:     "escaped quotes",
			input:    `{"msg": "say \"hi\""}`,
			expected: 20,
		},
		{
			name:     "incomplete object",
			input:    `{"a": {"b": 1}`,
			expected: -1,
		},
		{
			name:     "deeply nested",
			input:    `{"a": {"b": {"c": {"d": 1}}}}`,
			expected: 28,
		},
		{
			name:     "object with trailing content",
			input:    `{"a": 1} extra`,
			expected: 7,
		},
		{
			name:     "array",
			input:    `[{"a": 1}, {"b": 2}]`,
			expected: 19,
		},
		{
			name:     "escaped backslash before quote",
			input:    `{"path": "C:\\"}`,
			expected: 15,
		},
		{
			name:     "empty string",
			input:    "",
			expected: -1,
		},
		{
			name:     "no opening brace",
			input:    "hello world",
			expected: -1,
		},
		{
			name:     "only opening brace",
			input:    "{",
			expected: -1,
		},
		{
			name:     "unclosed string",
			input:    `{"key": "unclosed`,
			expected: -1,
		},
		{
			name:     "double escaped backslash then quote",
			input:    `{"path": "C:\\\\"}`,
			expected: 17,
		},
		{
			name:     "unicode in key and value",
			input:    `{"키": "값"}`,
			expected: 13,
		},
		{
			name:     "nested arrays",
			input:    `{"matrix": [[1, 2], [3, 4]]}`,
			expected: 27,
		},
		{
			name:     "mixed nesting",
			input:    `{"a": [{"b": {"c": [1, 2, 3]}}]}`,
			expected: 31,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := findJSONEnd(tt.input)
			if result != tt.expected {
				t.Errorf("findJSONEnd(%q) = %d, want %d", tt.input, result, tt.expected)
			}
		})
	}
}

func TestMinistralParser_HasToolSupport(t *testing.T) {
	p := &MinistralParser{}
	if !p.HasToolSupport() {
		t.Error("expected HasToolSupport to return true")
	}
}

func TestMinistralParser_HasThinkingSupport(t *testing.T) {
	p := &MinistralParser{hasThinkingSupport: false}
	if p.HasThinkingSupport() {
		t.Error("expected HasThinkingSupport to return false")
	}

	p = &MinistralParser{hasThinkingSupport: true}
	if !p.HasThinkingSupport() {
		t.Error("expected HasThinkingSupport to return true")
	}
}
