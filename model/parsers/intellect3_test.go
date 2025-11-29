package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestIntellect3ParserThinkingOnly(t *testing.T) {
	cases := []struct {
		desc      string
		chunks    []string
		wantText  string
		wantThink string
	}{
		{
			desc:      "simple thinking content",
			chunks:    []string{"<think>I need to analyze this</think>Here is my response"},
			wantText:  "Here is my response",
			wantThink: "I need to analyze this",
		},
		{
			desc:      "thinking with whitespace",
			chunks:    []string{"<think>\n  Some thoughts  \n</think>\n\nContent"},
			wantText:  "Content",
			wantThink: "Some thoughts  \n", // Thinking parser preserves internal whitespace
		},
		{
			desc:      "thinking only",
			chunks:    []string{"<think>Just thinking</think>"},
			wantText:  "",
			wantThink: "Just thinking",
		},
		{
			desc:      "no thinking tags",
			chunks:    []string{"Just regular content"},
			wantText:  "Just regular content",
			wantThink: "",
		},
		{
			desc:      "streaming thinking content",
			chunks:    []string{"<think>Fir", "st part", " second part</think>Content"},
			wantText:  "Content",
			wantThink: "First part second part",
		},
		{
			desc:      "partial opening tag",
			chunks:    []string{"<thi", "nk>Thinking</think>Content"},
			wantText:  "Content",
			wantThink: "Thinking",
		},
		{
			desc:      "partial closing tag",
			chunks:    []string{"<think>Thinking</thi", "nk>Content"},
			wantText:  "Content",
			wantThink: "Thinking",
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			parser := Intellect3Parser{}
			parser.Init(nil, nil, nil)

			var gotText, gotThink string
			for i, chunk := range tc.chunks {
				isLast := i == len(tc.chunks)-1
				text, think, calls, err := parser.Add(chunk, isLast)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				gotText += text
				gotThink += think
				if len(calls) > 0 {
					t.Fatalf("expected no tool calls, got %v", calls)
				}
			}

			if gotText != tc.wantText {
				t.Errorf("content: got %q, want %q", gotText, tc.wantText)
			}
			if gotThink != tc.wantThink {
				t.Errorf("thinking: got %q, want %q", gotThink, tc.wantThink)
			}
		})
	}
}

func TestIntellect3ParserToolCallsOnly(t *testing.T) {
	tools := []api.Tool{
		tool("get_weather", map[string]api.ToolProperty{
			"location": {Type: api.PropertyType{"string"}},
			"unit":     {Type: api.PropertyType{"string"}},
		}),
	}

	cases := []struct {
		desc      string
		chunks    []string
		wantText  string
		wantCalls []api.ToolCall
	}{
		{
			desc: "simple tool call",
			chunks: []string{
				"Let me check the weather<tool_call><function=get_weather>\n<parameter=location>\nSan Francisco\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function></tool_call>",
			},
			wantText: "Let me check the weather",
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
			desc: "tool call streaming",
			chunks: []string{
				"Checking<tool_call><function=get_wea",
				"ther>\n<parameter=location>\nNew York\n</param", //nolint:all
				"eter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function></tool_call>Done",
			},
			wantText: "CheckingDone",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "New York",
							"unit":     "fahrenheit",
						},
					},
				},
			},
		},
		{
			desc: "multiple tool calls",
			chunks: []string{
				"<tool_call><function=get_weather>\n<parameter=location>\nBoston\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function></tool_call>",
				"<tool_call><function=get_weather>\n<parameter=location>\nSeattle\n</parameter>\n<parameter=unit>\nfahrenheit\n</parameter>\n</function></tool_call>",
			},
			wantText: "",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Boston",
							"unit":     "celsius",
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Seattle",
							"unit":     "fahrenheit",
						},
					},
				},
			},
		},
		{
			desc:      "no tool calls",
			chunks:    []string{"Just regular content"},
			wantText:  "Just regular content",
			wantCalls: nil,
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			parser := Intellect3Parser{}
			parser.Init(tools, nil, nil)

			var gotText string
			var gotCalls []api.ToolCall
			for i, chunk := range tc.chunks {
				isLast := i == len(tc.chunks)-1
				text, think, calls, err := parser.Add(chunk, isLast)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				gotText += text
				gotCalls = append(gotCalls, calls...)
				if think != "" {
					t.Fatalf("expected no thinking, got %q", think)
				}
			}

			if gotText != tc.wantText {
				t.Errorf("content: got %q, want %q", gotText, tc.wantText)
			}
			if !reflect.DeepEqual(gotCalls, tc.wantCalls) {
				t.Errorf("tool calls: got %#v, want %#v", gotCalls, tc.wantCalls)
			}
		})
	}
}

func TestIntellect3ParserCombined(t *testing.T) {
	tools := []api.Tool{
		tool("get_weather", map[string]api.ToolProperty{
			"location": {Type: api.PropertyType{"string"}},
			"unit":     {Type: api.PropertyType{"string"}},
		}),
	}

	cases := []struct {
		desc      string
		chunks    []string
		wantText  string
		wantThink string
		wantCalls []api.ToolCall
	}{
		{
			desc: "thinking then tool call",
			chunks: []string{
				"<think>Need to get weather data</think>Let me check<tool_call><function=get_weather>\n<parameter=location>\nParis\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function></tool_call>",
			},
			wantText:  "Let me check",
			wantThink: "Need to get weather data",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Paris",
							"unit":     "celsius",
						},
					},
				},
			},
		},
		{
			desc: "thinking, tool call, and final content",
			chunks: []string{
				"<think>User wants weather info</think>Checking weather<tool_call><function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function></tool_call>Done!",
			},
			wantText:  "Checking weatherDone!",
			wantThink: "User wants weather info",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Tokyo",
							"unit":     "celsius",
						},
					},
				},
			},
		},
		{
			desc: "streaming combined content",
			chunks: []string{
				"<think>Analyzing",
				" the request</think>",
				"Let me help<tool_call>",
				"<function=get_weather>\n<parameter=location>\nLondon",
				"\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>",
				"</tool_call>There you go!",
			},
			wantText:  "Let me helpThere you go!",
			wantThink: "Analyzing the request",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "London",
							"unit":     "celsius",
						},
					},
				},
			},
		},
		{
			desc: "multiple tool calls with thinking",
			chunks: []string{
				"<think>Need multiple locations</think>",
				"<tool_call><function=get_weather>\n<parameter=location>\nBoston\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function></tool_call>",
				"and<tool_call><function=get_weather>\n<parameter=location>\nBerlin\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function></tool_call>",
			},
			wantText:  "and",
			wantThink: "Need multiple locations",
			wantCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Boston",
							"unit":     "celsius",
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]any{
							"location": "Berlin",
							"unit":     "celsius",
						},
					},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			parser := Intellect3Parser{}
			parser.Init(tools, nil, nil)

			var gotText, gotThink string
			var gotCalls []api.ToolCall
			for i, chunk := range tc.chunks {
				isLast := i == len(tc.chunks)-1
				text, think, calls, err := parser.Add(chunk, isLast)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				gotText += text
				gotThink += think
				gotCalls = append(gotCalls, calls...)
			}

			if gotText != tc.wantText {
				t.Errorf("content: got %q, want %q", gotText, tc.wantText)
			}
			if gotThink != tc.wantThink {
				t.Errorf("thinking: got %q, want %q", gotThink, tc.wantThink)
			}
			if !reflect.DeepEqual(gotCalls, tc.wantCalls) {
				t.Errorf("tool calls: got %#v, want %#v", gotCalls, tc.wantCalls)
			}
		})
	}
}

func TestIntellect3ParserEdgeCases(t *testing.T) {
	tools := []api.Tool{
		tool("test_func", map[string]api.ToolProperty{
			"param": {Type: api.PropertyType{"string"}},
		}),
	}

	cases := []struct {
		desc      string
		chunks    []string
		wantText  string
		wantThink string
		wantCalls int
	}{
		{
			desc:      "empty input",
			chunks:    []string{""},
			wantText:  "",
			wantThink: "",
			wantCalls: 0,
		},
		{
			desc:      "only whitespace",
			chunks:    []string{"   \n  \t  "},
			wantText:  "",
			wantThink: "",
			wantCalls: 0,
		},
		{
			desc:      "unclosed thinking tag",
			chunks:    []string{"<think>Never closes"},
			wantText:  "",
			wantThink: "Never closes",
			wantCalls: 0,
		},
		{
			desc:      "unclosed tool call tag",
			chunks:    []string{"<tool_call><function=test_func>\n<parameter=param>\nvalue\n</parameter>\n</function>"},
			wantText:  "", // Qwen3CoderParser waits for closing tag, doesn't emit partial tool calls
			wantThink: "",
			wantCalls: 0, // Won't be parsed until </tool_call> is seen
		},
		{
			desc:      "unicode in thinking",
			chunks:    []string{"<think>思考中 🤔</think>答案是 42"},
			wantText:  "答案是 42",
			wantThink: "思考中 🤔",
			wantCalls: 0,
		},
		{
			desc:      "fake thinking tag",
			chunks:    []string{"<thinking>This is not the right tag</thinking>Content"},
			wantText:  "<thinking>This is not the right tag</thinking>Content",
			wantThink: "",
			wantCalls: 0,
		},
		{
			desc:      "fake tool call tag",
			chunks:    []string{"<tool>Not a tool call</tool>"},
			wantText:  "<tool>Not a tool call</tool>",
			wantThink: "",
			wantCalls: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			parser := Intellect3Parser{}
			parser.Init(tools, nil, nil)

			var gotText, gotThink string
			var gotCalls []api.ToolCall
			for i, chunk := range tc.chunks {
				isLast := i == len(tc.chunks)-1
				text, think, calls, err := parser.Add(chunk, isLast)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				gotText += text
				gotThink += think
				gotCalls = append(gotCalls, calls...)
			}

			if gotText != tc.wantText {
				t.Errorf("content: got %q, want %q", gotText, tc.wantText)
			}
			if gotThink != tc.wantThink {
				t.Errorf("thinking: got %q, want %q", gotThink, tc.wantThink)
			}
			if len(gotCalls) != tc.wantCalls {
				t.Errorf("tool calls count: got %d, want %d", len(gotCalls), tc.wantCalls)
			}
		})
	}
}

func TestIntellect3ParserCapabilities(t *testing.T) {
	parser := Intellect3Parser{}

	if !parser.HasToolSupport() {
		t.Error("Intellect3Parser should have tool support")
	}

	if !parser.HasThinkingSupport() {
		t.Error("Intellect3Parser should have thinking support")
	}
}

func TestIntellect3ParserInit(t *testing.T) {
	parser := Intellect3Parser{}

	tools := []api.Tool{
		tool("test", map[string]api.ToolProperty{
			"param": {Type: api.PropertyType{"string"}},
		}),
	}

	returnedTools := parser.Init(tools, nil, nil)

	// Should return tools unchanged (delegated to Qwen3CoderParser)
	if !reflect.DeepEqual(returnedTools, tools) {
		t.Errorf("Init should return tools unchanged")
	}
}

func TestIntellect3ParserWhitespaceHandling(t *testing.T) {
	tools := []api.Tool{
		tool("test", map[string]api.ToolProperty{
			"param": {Type: api.PropertyType{"string"}},
		}),
	}

	cases := []struct {
		desc      string
		chunks    []string
		wantText  string
		wantThink string
	}{
		{
			desc:      "whitespace between thinking and content",
			chunks:    []string{"<think>Thinking</think>\n\n\nContent"},
			wantText:  "Content",
			wantThink: "Thinking",
		},
		{
			desc:      "whitespace inside thinking tags",
			chunks:    []string{"<think>  \n  Thinking  \n  </think>Content"},
			wantText:  "Content",
			wantThink: "Thinking  \n  ", // Thinking parser preserves internal whitespace
		},
		{
			desc:      "leading whitespace before thinking",
			chunks:    []string{"   <think>Thinking</think>Content"},
			wantText:  "Content",
			wantThink: "Thinking",
		},
		{
			desc:      "whitespace before tool call",
			chunks:    []string{"Text   <tool_call><function=test>\n<parameter=param>\nvalue\n</parameter>\n</function></tool_call>"},
			wantText:  "Text",
			wantThink: "",
		},
		{
			desc:      "whitespace after tool call",
			chunks:    []string{"<tool_call><function=test>\n<parameter=param>\nvalue\n</parameter>\n</function></tool_call>   Text"},
			wantText:  "Text",
			wantThink: "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			parser := Intellect3Parser{}
			parser.Init(tools, nil, nil)

			var gotText, gotThink string
			for i, chunk := range tc.chunks {
				isLast := i == len(tc.chunks)-1
				text, think, _, err := parser.Add(chunk, isLast)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				gotText += text
				gotThink += think
			}

			if gotText != tc.wantText {
				t.Errorf("content: got %q, want %q", gotText, tc.wantText)
			}
			if gotThink != tc.wantThink {
				t.Errorf("thinking: got %q, want %q", gotThink, tc.wantThink)
			}
		})
	}
}
