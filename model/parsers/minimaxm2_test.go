package parsers

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestMiniMaxM2Parser(t *testing.T) {
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"location": {
							Type: api.PropertyType{"string"},
						},
						"unit": {
							Type: api.PropertyType{"string"},
						},
					},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "search_web",
				Description: "Search the web",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"query": {
							Type: api.PropertyType{"string"},
						},
						"limit": {
							Type: api.PropertyType{"integer"},
						},
					},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "calculate",
				Description: "Perform a calculation",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"expression": {
							Type: api.PropertyType{"string"},
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name     string
		input    string
		wantContent  string
		wantThinking string
		wantCalls    []api.ToolCall
		wantError    bool
	}{
		// Basic functionality
		{
			name:         "simple content",
			input:        "Hello world",
			wantContent:  "Hello world",
			wantThinking: "",
			wantCalls:    nil,
		},
		{
			name: "single invoke with single parameter",
			input: `<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
			},
		},
		{
			name: "single invoke with multiple parameters",
			input: `<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">San Francisco</parameter>
<parameter name="unit">celsius</parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "San Francisco",
							"unit":     "celsius",
						},
					},
				},
			},
		},
		{
			name: "single invoke with no parameters",
			input: `<minimax:tool_call>
<invoke name="calculate">
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "calculate",
						Arguments: map[string]any{},
					},
				},
			},
		},
		{
			name: "multiple invokes in one tool_call block",
			input: `<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
</invoke>
<invoke name="get_weather">
<parameter name="location">London</parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "London",
						},
					},
				},
			},
		},
		{
			name: "parameter with JSON object value",
			input: `<minimax:tool_call>
<invoke name="search_web">
<parameter name="query">{"keywords": ["AI", "news"]}</parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search_web",
						Arguments: map[string]any{
							"query": map[string]any{
								"keywords": []any{"AI", "news"},
							},
						},
					},
				},
			},
		},
		{
			name: "parameter with JSON array value",
			input: `<minimax:tool_call>
<invoke name="search_web">
<parameter name="query">["AI", "machine learning"]</parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search_web",
						Arguments: map[string]any{
							"query": []any{"AI", "machine learning"},
						},
					},
				},
			},
		},
		{
			name: "parameter with integer value",
			input: `<minimax:tool_call>
<invoke name="search_web">
<parameter name="query">AI news</parameter>
<parameter name="limit">10</parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search_web",
						Arguments: map[string]any{
							"query": "AI news",
							"limit": float64(10), // JSON unmarshals numbers as float64
						},
					},
				},
			},
		},

		// Content mixing
		{
			name:         "content before tool call",
			input:        "Let me check the weather. <minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">Tokyo</parameter>\n</invoke>\n</minimax:tool_call>",
			wantContent:  "Let me check the weather. ",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
			},
		},
		{
			name:         "content after tool call",
			input:        "<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">Tokyo</parameter>\n</invoke>\n</minimax:tool_call>\nHere is the weather.",
			wantContent:  "Here is the weather.",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
			},
		},
		{
			name:         "content before and after tool call",
			input:        "Let me check. <minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">Tokyo</parameter>\n</invoke>\n</minimax:tool_call>\nDone checking.",
			wantContent:  "Let me check. Done checking.",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
			},
		},
		{
			name:         "thinking content only",
			input:        "<think>I need to analyze this problem first.</think>",
			wantContent:  "",
			wantThinking: "I need to analyze this problem first.",
			wantCalls:    nil,
		},
		{
			name:         "thinking then tool call",
			input:        "<think>I should check the weather.</think>\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">Tokyo</parameter>\n</invoke>\n</minimax:tool_call>",
			wantContent:  "",
			wantThinking: "I should check the weather.",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
			},
		},
		{
			name:         "thinking with content",
			input:        "Let me think. <think>This is complex.</think> Okay, done.",
			wantContent:  "Let me think. Okay, done.",
			wantThinking: "This is complex.",
			wantCalls:    nil,
		},

		// Edge cases
		{
			name: "whitespace in parameter values",
			input: `<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">  San Francisco  </parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "San Francisco",
						},
					},
				},
			},
		},
		{
			name: "empty parameter value",
			input: `<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location"></parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "",
						},
					},
				},
			},
		},
		{
			name: "special characters in values",
			input: `<minimax:tool_call>
<invoke name="search_web">
<parameter name="query">AI &amp; Machine Learning</parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search_web",
						Arguments: map[string]any{
							"query": "AI &amp; Machine Learning",
						},
					},
				},
			},
		},
		{
			name:         "tool call inside think tag (should treat as thinking)",
			input:        "<think>Maybe I should call <minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Tokyo</parameter></invoke></minimax:tool_call></think>",
			wantContent:  "",
			wantThinking: "Maybe I should call <minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Tokyo</parameter></invoke></minimax:tool_call>",
			wantCalls:    nil,
		},
		{
			name:         "multiple thinking blocks",
			input:        "<think>First thought</think> Some content. <think>Second thought</think>",
			wantContent:  "Some content. ",
			wantThinking: "First thoughtSecond thought",
			wantCalls:    nil,
		},
		{
			name: "multiple tool call blocks",
			input: `<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
</invoke>
</minimax:tool_call>
Some text.
<minimax:tool_call>
<invoke name="search_web">
<parameter name="query">news</parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "Some text.\n",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "search_web",
						Arguments: map[string]any{
							"query": "news",
						},
					},
				},
			},
		},

		// Recovery scenarios
		{
			name: "missing closing parameter tag",
			input: `<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Tokyo</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
			},
		},
		{
			name: "missing closing invoke tag",
			input: `<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Tokyo</parameter>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
			},
		},
		{
			name: "parameter with newlines (should trim)",
			input: `<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">
Tokyo
</parameter>
</invoke>
</minimax:tool_call>`,
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &MiniMaxM2Parser{}
			parser.Init(tools, nil, nil)

			gotContent, gotThinking, gotCalls, err := parser.Add(tt.input, true)

			if (err != nil) != tt.wantError {
				t.Errorf("Add() error = %v, wantError %v", err, tt.wantError)
				return
			}

			if gotContent != tt.wantContent {
				t.Errorf("Add() content = %q, want %q", gotContent, tt.wantContent)
			}

			if gotThinking != tt.wantThinking {
				t.Errorf("Add() thinking = %q, want %q", gotThinking, tt.wantThinking)
			}

			if diff := cmp.Diff(tt.wantCalls, gotCalls); diff != "" {
				t.Errorf("Add() calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMiniMaxM2ParserStreaming(t *testing.T) {
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"location": {Type: api.PropertyType{"string"}},
					},
				},
			},
		},
	}

	tests := []struct {
		name         string
		chunks       []string
		wantContent  string
		wantThinking string
		wantCalls    []api.ToolCall
	}{
		{
			name: "tool call tag split across chunks",
			chunks: []string{
				"Let me check. <minimax:tool",
				"_call>\n<invoke name=\"get_weather\">\n",
				"<parameter name=\"location\">Tokyo</parameter>\n",
				"</invoke>\n</minimax:tool_call>",
			},
			wantContent:  "Let me check. ",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"location": "Tokyo"},
					},
				},
			},
		},
		{
			name: "invoke tag split across chunks",
			chunks: []string{
				"<minimax:tool_call>\n<inv",
				"oke name=\"get_weather\">\n<parameter name=\"location\">",
				"Tokyo</parameter>\n</invoke>\n</minimax:tool_call>",
			},
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"location": "Tokyo"},
					},
				},
			},
		},
		{
			name: "parameter value split across chunks",
			chunks: []string{
				"<minimax:tool_call>\n<invoke name=\"get_weather\">\n",
				"<parameter name=\"location\">San Fran",
				"cisco</parameter>\n</invoke>\n</minimax:tool_call>",
			},
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"location": "San Francisco"},
					},
				},
			},
		},
		{
			name: "closing tag split across chunks",
			chunks: []string{
				"<minimax:tool_call>\n<invoke name=\"get_weather\">\n",
				"<parameter name=\"location\">Tokyo</parameter>\n</inv",
				"oke>\n</minimax:tool_call>",
			},
			wantContent:  "",
			wantThinking: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"location": "Tokyo"},
					},
				},
			},
		},
		{
			name: "thinking tag split across chunks",
			chunks: []string{
				"Let me think. <thi",
				"nk>This is complex.</th",
				"ink> Done.",
			},
			wantContent:  "Let me think. Done.",
			wantThinking: "This is complex.",
			wantCalls:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &MiniMaxM2Parser{}
			parser.Init(tools, nil, nil)

			var allContent, allThinking string
			var allCalls []api.ToolCall

			for i, chunk := range tt.chunks {
				isDone := i == len(tt.chunks)-1
				content, thinking, calls, err := parser.Add(chunk, isDone)
				if err != nil {
					t.Fatalf("Add() error = %v", err)
				}
				allContent += content
				allThinking += thinking
				allCalls = append(allCalls, calls...)
			}

			if allContent != tt.wantContent {
				t.Errorf("Streaming content = %q, want %q", allContent, tt.wantContent)
			}

			if allThinking != tt.wantThinking {
				t.Errorf("Streaming thinking = %q, want %q", allThinking, tt.wantThinking)
			}

			if diff := cmp.Diff(tt.wantCalls, allCalls); diff != "" {
				t.Errorf("Streaming calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMiniMaxM2ParserErrors(t *testing.T) {
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Type:       "object",
					Properties: map[string]api.ToolProperty{},
				},
			},
		},
	}

	tests := []struct {
		name      string
		input     string
		wantError bool
	}{
		{
			name: "unknown tool",
			input: `<minimax:tool_call>
<invoke name="unknown_function">
<parameter name="param">value</parameter>
</invoke>
</minimax:tool_call>`,
			wantError: true,
		},
		{
			name:      "no tools provided but model makes tool call",
			input:     `<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">Tokyo</parameter>\n</invoke>\n</minimax:tool_call>`,
			wantError: true, // Should error when tool is not in registry
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &MiniMaxM2Parser{}
			if tt.name == "no tools provided but model makes tool call" {
				parser.Init(nil, nil, nil) // No tools
			} else {
				parser.Init(tools, nil, nil)
			}

			_, _, _, err := parser.Add(tt.input, true)

			if (err != nil) != tt.wantError {
				t.Errorf("Add() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

func TestMiniMaxM2ParserAttributeExtraction(t *testing.T) {
	tests := []struct {
		name string
		tag  string
		want string
	}{
		{
			name: "double quotes",
			tag:  `<invoke name="get_weather">`,
			want: "get_weather",
		},
		{
			name: "single quotes",
			tag:  `<invoke name='get_weather'>`,
			want: "get_weather",
		},
		{
			name: "with extra attributes",
			tag:  `<invoke name="get_weather" id="123">`,
			want: "get_weather",
		},
		{
			name: "parameter tag",
			tag:  `<parameter name="location">`,
			want: "location",
		},
		{
			name: "no name attribute",
			tag:  `<invoke>`,
			want: "",
		},
		{
			name: "malformed quote",
			tag:  `<invoke name="get_weather>`,
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractNameAttribute(tt.tag)
			if got != tt.want {
				t.Errorf("extractNameAttribute() = %q, want %q", got, tt.want)
			}
		})
	}
}
