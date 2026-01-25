package parsers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

// processAll keeps calling Add until no more progress is made.
// This simulates the behavior of feeding all input at once and letting
// the parser process through multiple states.
func processAll(p *MinistralParser, input string) (content, thinking string, calls []api.ToolCall, err error) {
	// Feed the input
	c, th, cl, err := p.Add(input, false)
	if err != nil {
		return c, th, cl, err
	}
	content += c
	thinking += th
	calls = append(calls, cl...)

	// Keep calling Add with empty string until parser settles
	// This allows state transitions to complete
	for i := 0; i < 10; i++ { // max 10 iterations to prevent infinite loop
		c, th, cl, err := p.Add("", false)
		if err != nil {
			return content + c, thinking + th, append(calls, cl...), err
		}
		if c == "" && th == "" && len(cl) == 0 {
			break
		}
		content += c
		thinking += th
		calls = append(calls, cl...)
	}

	// Final drain
	c, th, cl, err = p.Add("", true)
	if err != nil {
		return content + c, thinking + th, append(calls, cl...), err
	}
	content += c
	thinking += th
	calls = append(calls, cl...)

	return content, thinking, calls, nil
}

func TestMinistralParser(t *testing.T) {
	tests := []struct {
		name             string
		tools            []api.Tool
		input            string
		expectedContent  string
		expectedThinking string
		expectedCalls    []api.ToolCall
	}{
		{
			name:            "simple content",
			input:           "Hello, how can I help you?",
			expectedContent: "Hello, how can I help you?",
		},
		{
			name: "simple tool call",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "get_weather"}},
			},
			input:           `[TOOL_CALLS]get_weather[ARGS]{"location": "San Francisco"}`,
			expectedContent: `get_weather[ARGS]{"location": "San Francisco"}`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "San Francisco"}),
					},
				},
			},
		},
		{
			name: "tool call with nested object",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "create_entities"}},
			},
			input:           `[TOOL_CALLS]create_entities[ARGS]{"entities": [{"entityType": "Person", "name": "Jack", "observations": ["Works as a baker at Big Baker Co."]}]}`,
			expectedContent: `create_entities[ARGS]{"entities": [{"entityType": "Person", "name": "Jack", "observations": ["Works as a baker at Big Baker Co."]}]}`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "create_entities",
						Arguments: testArgs(map[string]any{
							"entities": []any{
								map[string]any{
									"entityType":   "Person",
									"name":         "Jack",
									"observations": []any{"Works as a baker at Big Baker Co."},
								},
							},
						}),
					},
				},
			},
		},
		{
			name: "tool call with deeply nested objects",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "update_config"}},
			},
			input:           `[TOOL_CALLS]update_config[ARGS]{"settings": {"user": {"profile": {"name": "John", "age": 30}}, "theme": "dark"}}`,
			expectedContent: `update_config[ARGS]{"settings": {"user": {"profile": {"name": "John", "age": 30}}, "theme": "dark"}}`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "update_config",
						Arguments: testArgs(map[string]any{
							"settings": map[string]any{
								"user": map[string]any{
									"profile": map[string]any{
										"name": "John",
										"age":  float64(30),
									},
								},
								"theme": "dark",
							},
						}),
					},
				},
			},
		},
		{
			name: "tool call with array of objects",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "process_items"}},
			},
			input:           `[TOOL_CALLS]process_items[ARGS]{"items": [{"id": 1}, {"id": 2}, {"id": 3}]}`,
			expectedContent: `process_items[ARGS]{"items": [{"id": 1}, {"id": 2}, {"id": 3}]}`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "process_items",
						Arguments: testArgs(map[string]any{
							"items": []any{
								map[string]any{"id": float64(1)},
								map[string]any{"id": float64(2)},
								map[string]any{"id": float64(3)},
							},
						}),
					},
				},
			},
		},
		{
			name: "tool call with escaped quotes in string",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "search"}},
			},
			input:           `[TOOL_CALLS]search[ARGS]{"query": "say \"hello\""}`,
			expectedContent: `search[ARGS]{"query": "say \"hello\""}`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "search",
						Arguments: testArgs(map[string]any{"query": `say "hello"`}),
					},
				},
			},
		},
		{
			name: "tool call with braces inside string",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "format"}},
			},
			input:           `[TOOL_CALLS]format[ARGS]{"template": "Hello {name}!"}`,
			expectedContent: `format[ARGS]{"template": "Hello {name}!"}`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "format",
						Arguments: testArgs(map[string]any{"template": "Hello {name}!"}),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &MinistralParser{}
			p.Init(tt.tools, nil, nil)

			content, thinking, calls, err := processAll(p, tt.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if diff := cmp.Diff(content, tt.expectedContent); diff != "" {
				t.Errorf("content mismatch (-got +want):\n%s", diff)
			}
			if diff := cmp.Diff(thinking, tt.expectedThinking); diff != "" {
				t.Errorf("thinking mismatch (-got +want):\n%s", diff)
			}
			if diff := cmp.Diff(calls, tt.expectedCalls, argsComparer); diff != "" {
				t.Errorf("calls mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestMinistralParser_Streaming(t *testing.T) {
	tests := []struct {
		name            string
		tools           []api.Tool
		chunks          []string
		expectedContent string
		expectedCalls   []api.ToolCall
	}{
		{
			name:            "streaming content",
			chunks:          []string{"Hello, ", "how ", "can I help?"},
			expectedContent: "Hello, how can I help?",
		},
		{
			name: "streaming tool call with nested objects",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "create_entities"}},
			},
			chunks: []string{
				"[TOOL_CALLS]create_entities[ARGS]",
				`{"entities": [{"entityType": "Person",`,
				` "name": "Jack",`,
				` "observations": ["Works`,
				` as a baker at Big Baker Co."]}`,
				`]}`,
			},
			expectedContent: "create_entities[ARGS]",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "create_entities",
						Arguments: testArgs(map[string]any{
							"entities": []any{
								map[string]any{
									"entityType":   "Person",
									"name":         "Jack",
									"observations": []any{"Works as a baker at Big Baker Co."},
								},
							},
						}),
					},
				},
			},
		},
		{
			name: "streaming with incomplete JSON waits for completion",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "test"}},
			},
			chunks: []string{
				"[TOOL_CALLS]test[ARGS]{",
				`"a": {`,
				`"b": 1`,
				`}`,
				`}`,
			},
			expectedContent: "test[ARGS]{",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "test",
						Arguments: testArgs(map[string]any{
							"a": map[string]any{
								"b": float64(1),
							},
						}),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &MinistralParser{}
			p.Init(tt.tools, nil, nil)

			var allContent string
			var allCalls []api.ToolCall

			for _, chunk := range tt.chunks {
				content, _, calls, err := p.Add(chunk, false)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				allContent += content
				allCalls = append(allCalls, calls...)
			}

			// Drain
			content, _, calls, err := p.Add("", true)
			if err != nil {
				t.Fatalf("unexpected error on done: %v", err)
			}
			allContent += content
			allCalls = append(allCalls, calls...)

			if diff := cmp.Diff(allContent, tt.expectedContent); diff != "" {
				t.Errorf("content mismatch (-got +want):\n%s", diff)
			}
			if diff := cmp.Diff(allCalls, tt.expectedCalls, argsComparer); diff != "" {
				t.Errorf("calls mismatch (-got +want):\n%s", diff)
			}
		})
	}
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

func TestMinistralParser_EdgeCases(t *testing.T) {
	tests := []struct {
		name             string
		tools            []api.Tool
		input            string
		expectedContent  string
		expectedThinking string
		expectedCalls    []api.ToolCall
		expectError      bool
	}{
		{
			name: "unknown tool returns error",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "known_tool"}},
			},
			input:       `[TOOL_CALLS]unknown_tool[ARGS]{"a": 1}`,
			expectError: true,
		},
		{
			name: "invalid JSON returns error",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "test"}},
			},
			input:       `[TOOL_CALLS]test[ARGS]{invalid json}`,
			expectError: true,
		},
		{
			name: "empty JSON object",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "no_args"}},
			},
			input:           `[TOOL_CALLS]no_args[ARGS]{}`,
			expectedContent: "no_args[ARGS]{}",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "no_args",
						Arguments: testArgs(map[string]any{}),
					},
				},
			},
		},
		{
			name: "JSON with newlines in string",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "write"}},
			},
			input:           `[TOOL_CALLS]write[ARGS]{"content": "line1\nline2\nline3"}`,
			expectedContent: "write[ARGS]{\"content\": \"line1\\nline2\\nline3\"}",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "write",
						Arguments: testArgs(map[string]any{"content": "line1\nline2\nline3"}),
					},
				},
			},
		},
		{
			name: "backslash in string value",
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "path"}},
			},
			input:           `[TOOL_CALLS]path[ARGS]{"dir": "C:\\Users\\test"}`,
			expectedContent: "path[ARGS]{\"dir\": \"C:\\\\Users\\\\test\"}",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "path",
						Arguments: testArgs(map[string]any{"dir": "C:\\Users\\test"}),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &MinistralParser{}
			p.state = ministralCollectingContent
			p.Init(tt.tools, nil, nil)
			p.state = ministralCollectingContent

			content, thinking, calls, err := processAll(p, tt.input)

			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if diff := cmp.Diff(content, tt.expectedContent); diff != "" {
				t.Errorf("content mismatch (-got +want):\n%s", diff)
			}
			if diff := cmp.Diff(thinking, tt.expectedThinking); diff != "" {
				t.Errorf("thinking mismatch (-got +want):\n%s", diff)
			}
			if diff := cmp.Diff(calls, tt.expectedCalls, argsComparer); diff != "" {
				t.Errorf("calls mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
