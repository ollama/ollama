package parsers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestOlmo3Parser(t *testing.T) {
	tests := []struct {
		name             string
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
			name:  "simple tool call",
			input: `<function_calls>get_weather(location="San Francisco")</function_calls>`,
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
			name:            "content then tool call",
			input:           `Let me check the weather.<function_calls>get_weather(location="NYC")</function_calls>`,
			expectedContent: "Let me check the weather.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "NYC"}),
					},
				},
			},
		},
		{
			name:  "tool call with multiple arguments",
			input: `<function_calls>book_flight(from="SFO", to="NYC", date="2024-01-15")</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "book_flight",
						Arguments: testArgs(map[string]any{
							"from": "SFO",
							"to":   "NYC",
							"date": "2024-01-15",
						}),
					},
				},
			},
		},
		{
			name: "multiple tool calls",
			input: `<function_calls>get_weather(location="San Francisco")
get_weather(location="New York")</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "San Francisco"}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "New York"}),
					},
				},
			},
		},
		{
			name:  "tool call with numeric argument",
			input: `<function_calls>set_temperature(value=72)</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "set_temperature",
						Arguments: testArgs(map[string]any{"value": int64(72)}),
					},
				},
			},
		},
		{
			name:  "tool call with float argument",
			input: `<function_calls>set_price(amount=19.99)</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "set_price",
						Arguments: testArgs(map[string]any{"amount": 19.99}),
					},
				},
			},
		},
		{
			name:  "tool call with boolean argument",
			input: `<function_calls>toggle_setting(enabled=true)</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "toggle_setting",
						Arguments: testArgs(map[string]any{"enabled": true}),
					},
				},
			},
		},
		{
			name:  "tool call with null argument",
			input: `<function_calls>clear_value(field=null)</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "clear_value",
						Arguments: testArgs(map[string]any{"field": nil}),
					},
				},
			},
		},
		{
			name:  "tool call with array argument",
			input: `<function_calls>process_items(items=["apple", "banana", "cherry"])</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "process_items",
						Arguments: testArgs(map[string]any{"items": []any{"apple", "banana", "cherry"}}),
					},
				},
			},
		},
		{
			name:  "tool call with dict argument",
			input: `<function_calls>update_config(settings={"theme": "dark", "fontSize": 14})</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "update_config",
						Arguments: testArgs(map[string]any{
							"settings": map[string]any{
								"theme":    "dark",
								"fontSize": int64(14),
							},
						}),
					},
				},
			},
		},
		{
			name:  "tool call with nested dict",
			input: `<function_calls>create_request(data={"user": {"name": "John", "age": 30}, "active": true})</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "create_request",
						Arguments: testArgs(map[string]any{
							"data": map[string]any{
								"user": map[string]any{
									"name": "John",
									"age":  int64(30),
								},
								"active": true,
							},
						}),
					},
				},
			},
		},
		{
			name:  "tool call with no arguments",
			input: `<function_calls>get_current_time()</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_current_time",
						Arguments: testArgs(map[string]any{}),
					},
				},
			},
		},
		{
			name:  "tool call with single quotes",
			input: `<function_calls>search(query='hello world')</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "search",
						Arguments: testArgs(map[string]any{"query": "hello world"}),
					},
				},
			},
		},
		{
			name:  "tool call with escaped quotes",
			input: `<function_calls>search(query="say \"hello\"")</function_calls>`,
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
			name:  "tool call with mixed argument types",
			input: `<function_calls>create_user(name="John", age=30, active=true)</function_calls>`,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "create_user",
						Arguments: testArgs(map[string]any{
							"name":   "John",
							"age":    int64(30),
							"active": true,
						}),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Olmo3Parser{}
			p.Init(nil, nil, nil)

			content, thinking, calls, err := p.Add(tt.input, false)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Drain remaining content
			finalContent, finalThinking, finalCalls, err := p.Add("", true)
			if err != nil {
				t.Fatalf("unexpected error on done: %v", err)
			}
			content += finalContent
			thinking += finalThinking
			calls = append(calls, finalCalls...)

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

func TestOlmo3Parser_Streaming(t *testing.T) {
	tests := []struct {
		name            string
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
			name:   "streaming tool call",
			chunks: []string{"<function_", "calls>get_weather", "(location=\"SF\")", "</function_calls>"},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "SF"}),
					},
				},
			},
		},
		{
			name:            "streaming content then tool call",
			chunks:          []string{"Let me check.", "<function_calls>", "get_weather(location=\"NYC\")", "</function_calls>"},
			expectedContent: "Let me check.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "NYC"}),
					},
				},
			},
		},
		{
			name:   "tool call tag split across chunks",
			chunks: []string{"<func", "tion_calls>test()</function_calls>"},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "test",
						Arguments: testArgs(map[string]any{}),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Olmo3Parser{}
			p.Init(nil, nil, nil)

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

func TestOlmo3Parser_HasToolSupport(t *testing.T) {
	p := &Olmo3Parser{}
	if !p.HasToolSupport() {
		t.Error("expected HasToolSupport to return true")
	}
}

func TestOlmo3Parser_HasThinkingSupport(t *testing.T) {
	p := &Olmo3Parser{}
	if p.HasThinkingSupport() {
		t.Error("expected HasThinkingSupport to return false")
	}
}

func TestParseOlmo3FunctionCalls(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []api.ToolCall
		wantErr  bool
	}{
		{
			name:  "simple call",
			input: `get_weather(location="SF")`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "SF"}),
					},
				},
			},
		},
		{
			name:  "multiple args",
			input: `send_email(to="user@example.com", subject="Hello", body="Test message")`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "send_email",
						Arguments: testArgs(map[string]any{
							"to":      "user@example.com",
							"subject": "Hello",
							"body":    "Test message",
						}),
					},
				},
			},
		},
		{
			name: "multiple calls with newlines",
			input: `get_weather(location="SF")
get_time(timezone="PST")`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "SF"}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name:      "get_time",
						Arguments: testArgs(map[string]any{"timezone": "PST"}),
					},
				},
			},
		},
		{
			name:     "empty input",
			input:    "",
			expected: nil,
		},
		{
			name:     "whitespace only",
			input:    "   \n   ",
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls, err := parseOlmo3FunctionCalls(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseOlmo3FunctionCalls() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(calls, tt.expected, argsComparer); diff != "" {
				t.Errorf("calls mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestParseOlmo3Value(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected any
	}{
		{"string double quotes", `"hello"`, "hello"},
		{"string single quotes", `'hello'`, "hello"},
		{"integer", "42", int64(42)},
		{"negative integer", "-10", int64(-10)},
		{"float", "3.14", 3.14},
		{"boolean true", "true", true},
		{"boolean True", "True", true},
		{"boolean false", "false", false},
		{"null", "null", nil},
		{"None", "None", nil},
		{"empty array", "[]", []any{}},
		{"array with strings", `["a", "b"]`, []any{"a", "b"}},
		{"array with numbers", "[1, 2, 3]", []any{int64(1), int64(2), int64(3)}},
		{"empty object", "{}", map[string]any{}},
		{"simple object", `{"name": "John"}`, map[string]any{"name": "John"}},
		{"object with number", `{"age": 30}`, map[string]any{"age": int64(30)}},
		{"object with multiple keys", `{"a": 1, "b": 2}`, map[string]any{"a": int64(1), "b": int64(2)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseOlmo3Value(tt.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(result, tt.expected); diff != "" {
				t.Errorf("value mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
