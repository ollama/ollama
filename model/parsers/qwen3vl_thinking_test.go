package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen3VLThinkingParserStreaming(t *testing.T) {
	type step struct {
		input      string
		wantEvents []qwenEvent
	}

	cases := []struct {
		desc  string
		steps []step
		only  bool
	}{
		{
			desc: "simple thinking",
			steps: []step{
				{input: "abc</think>", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc"}}},
			},
		},
		{
			desc: "simple trip thinking",
			steps: []step{
				{input: "<think>abc</think>", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "<think>abc"}}},
			},
		},
		{
			desc: "thinking with split tags",
			steps: []step{
				{input: "abc", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc"}}},
				{input: "</think>", wantEvents: []qwenEvent{}},
			},
		},
		{
			desc: "multiple think tags",
			steps: []step{
				{input: "abc<think>actually, is not thinking</think>", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc<think>actually, is not thinking"}}},
			},
		},
		{
			desc: "thinking and tool call",
			steps: []step{
				{
					input: "I'm thinking</think><tool_call>I'm tool calling</tool_call>",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "I'm thinking"},
						qwenEventRawToolCall{raw: "I'm tool calling"},
					},
				},
			},
		},
		{
			desc: "thinking and content",
			steps: []step{
				{
					input: "I'm thinking</think>I'm content",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "I'm thinking"},
						qwenEventContent{content: "I'm content"},
					},
				},
			},
		},
		{
			desc: "thinking and tool call and content",
		},
		{
			desc: "nested thinking (outside thinking, inside thinking)",
			steps: []step{
				{
					input: "I'm thinking<think>I'm nested thinking</think></think>",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "I'm thinking<think>I'm nested thinking"},
						qwenEventContent{content: "</think>"},
					},
				},
			},
		},
		{
			desc: "interleaved thinking",
			steps: []step{
				{
					input: "<think>I'm thinking</think>I'm actually content</think>",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "<think>I'm thinking"},
						qwenEventContent{content: "I'm actually content</think>"},
					},
				},
			},
		},
		{
			desc: "nested thinking and tool call (outside thinking, inside tool call)",
			steps: []step{
				{
					input:      "I'm thinking<tool_call>I'm nested tool call</tool_call></think>",
					wantEvents: []qwenEvent{qwenEventThinkingContent{content: "I'm thinking<tool_call>I'm nested tool call</tool_call>"}},
				},
			},
		},
		{
			desc: "nested thinking and tool call (outside tool call, inside thinking)",
			steps: []step{
				{
					input: "<tool_call>I'm nested tool call<think>I'm thinking</think></tool_call>",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "<tool_call>I'm nested tool call<think>I'm thinking"},
						qwenEventContent{content: "</tool_call>"},
					},
				},
			},
		},
		{
			desc: "interleaved thinking and tool call",
			steps: []step{
				{
					input: "I'm thinking<tool_call>I'm NOT a nested tool call</think></tool_call><tool_call>I'm nested tool call 2<think></tool_call></think>",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "I'm thinking<tool_call>I'm NOT a nested tool call"},
						qwenEventContent{content: "</tool_call>"},
						qwenEventRawToolCall{raw: "I'm nested tool call 2<think>"},
						qwenEventContent{content: "</think>"},
					},
				},
			},
		},
		{
			desc: "partial thinking tag fakeout",
			steps: []step{
				{
					input:      "abc</think",
					wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc"}},
				},
				{
					input:      " fakeout",
					wantEvents: []qwenEvent{qwenEventThinkingContent{content: "</think fakeout"}},
				},
			},
		},
		{
			desc: "partial thinking incomplete",
			steps: []step{
				{
					input:      "abc<think>unfinished</think", // when something is ambiguious, we dont emit anything
					wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc<think>unfinished"}},
				},
			},
		},
		{
			desc: "test with split thinking and content",
			steps: []step{
				{
					input:      "abc<think>unfinished</th", // when something is ambiguious, we dont emit anything
					wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc<think>unfinished"}},
				},
				{
					input: "ink> def",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "def"},
					},
				},
			},
		},
		{
			desc: "thinking with no tags",
			steps: []step{
				{
					input: "Hello I am thinking",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "Hello I am thinking"},
					},
				},
				{
					input: "Hello I am thinking some more",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "Hello I am thinking some more"},
					},
				},
				{
					input: "Hello I am think</think>     NOT",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "Hello I am think"},
						qwenEventContent{content: "NOT"},
					},
				},
			},
		},
	}
	anyOnlies := false
	for _, tc := range cases {
		if tc.only {
			anyOnlies = true
		}
	}

	for _, tc := range cases {
		if anyOnlies && !tc.only {
			continue
		}

		t.Run(tc.desc, func(t *testing.T) {
			parser := Qwen3VLParser{hasThinkingSupport: true}
			parser.Init([]api.Tool{}, nil)
			// parser.state = CollectingThinkingContent

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

func TestQwen3VLThinkingToolParser(t *testing.T) {
	type step struct {
		name         string
		rawToolCall  string
		tools        []api.Tool
		wantToolCall api.ToolCall
	}

	steps := []step{
		{
			name:        "simple tool call",
			tools:       []api.Tool{},
			rawToolCall: `{"name": "get-current-weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "get-current-weather",
					Arguments: map[string]any{
						"location": "San Francisco, CA",
						"unit":     "fahrenheit",
					},
				},
			},
		},
		{
			name:        "names with spaces",
			tools:       []api.Tool{},
			rawToolCall: `{"name": "get current temperature", "arguments": {"location with spaces": "San Francisco", "unit with spaces": "celsius"}}`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "get current temperature",
					Arguments: map[string]any{
						"location with spaces": "San Francisco",
						"unit with spaces":     "celsius",
					},
				},
			},
		},
		{
			name:        "names with quotes",
			tools:       []api.Tool{},
			rawToolCall: `{"name": "\"get current temperature\"", "arguments": {"\"location with spaces\"": "San Francisco", "\"unit with spaces\"": "\"celsius\""}}`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "\"get current temperature\"",
					Arguments: map[string]any{
						"\"location with spaces\"": "San Francisco",
						"\"unit with spaces\"":     "\"celsius\"",
					},
				},
			},
		},
		{
			name:        "tool call with typed parameters (json types)",
			tools:       []api.Tool{},
			rawToolCall: `{"name": "calculate", "arguments": {"x": 3.14, "y": 42, "enabled": true, "items": ["a", "b", "c"]}}`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "calculate",
					Arguments: map[string]any{
						"x":       3.14,
						"y":       float64(42),
						"enabled": true,
						"items":   []any{"a", "b", "c"},
					},
				},
			},
		},
		{
			name:        "ampersands in parameter values",
			tools:       []api.Tool{},
			rawToolCall: `{"name": "exec", "arguments": {"command": "ls && echo \"done\""}}`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "exec",
					Arguments: map[string]any{
						"command": "ls && echo \"done\"",
					},
				},
			},
		},
		{
			name:        "angle brackets in parameter values",
			tools:       []api.Tool{},
			rawToolCall: `{"name": "exec", "arguments": {"command": "ls && echo \"a > b and a < b\""}}`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "exec",
					Arguments: map[string]any{
						"command": "ls && echo \"a > b and a < b\"",
					},
				},
			},
		},
		{
			name:        "unicode in function names and parameters",
			tools:       []api.Tool{},
			rawToolCall: `{"name": "获取天气", "arguments": {"城市": "北京", "message": "Hello! 你好! 🌟 مرحبا"}}`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "获取天气",
					Arguments: map[string]any{
						"城市":      "北京",
						"message": "Hello! 你好! 🌟 مرحبا",
					},
				},
			},
		},
	}

	for i, step := range steps {
		gotToolCall, err := parseJSONToolCall(qwenEventRawToolCall{raw: step.rawToolCall}, step.tools)
		if err != nil {
			t.Errorf("step %d (%s): %v", i, step.name, err)
		}
		if !reflect.DeepEqual(gotToolCall, step.wantToolCall) {
			t.Errorf("step %d (%s): got tool call %#v, want %#v", i, step.name, gotToolCall, step.wantToolCall)
		}
	}
}

func TestQwen3VLParserState(t *testing.T) {
	cases := []struct {
		desc        string
		hasThinking bool
		last        *api.Message
		wantState   qwenParserState
	}{
		{
			desc:        "no thinking support => CollectingContent",
			hasThinking: false,
			last:        nil,
			wantState:   CollectingContent,
		},
		{
			desc:        "thinking support, no last message => CollectingThinkingContent",
			hasThinking: true,
			last:        nil,
			wantState:   CollectingThinkingContent,
		},
		{
			desc:        "thinking support, last assistant with empty content => CollectingThinkingContent",
			hasThinking: true,
			last:        &api.Message{Role: "assistant", Content: ""},
			wantState:   CollectingThinkingContent,
		},
		{
			desc:        "thinking support, last assistant with content => CollectingContent",
			hasThinking: true,
			last:        &api.Message{Role: "assistant", Content: "hello"},
			wantState:   CollectingContent,
		},
		{
			desc:        "thinking support, last is user => CollectingThinkingContent",
			hasThinking: true,
			last:        &api.Message{Role: "user", Content: "hi"},
			wantState:   CollectingThinkingContent,
		},
	}

	for _, tc := range cases {
		parser := Qwen3VLParser{hasThinkingSupport: tc.hasThinking}
		parser.Init(nil, tc.last)
		if parser.state != tc.wantState {
			t.Errorf("%s: got state %v, want %v", tc.desc, parser.state, tc.wantState)
		}
	}
}

func TestQwen3VLThinkingParserWithThinkingPrefill(t *testing.T) {
	type step struct {
		input      string
		wantEvents []qwenEvent
	}

	cases := []struct {
		desc  string
		steps []step
		only  bool
	}{
		{
			desc: "thinking prefill",
			steps: []step{
				{input: "abc</think>", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc"}}},
			},
		},
		{
			desc: "thinking prefill with content",
			steps: []step{
				{input: "abc</th", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc"}}},
				{input: "ink> def", wantEvents: []qwenEvent{qwenEventContent{content: "def"}}},
			},
		},
		{
			desc: "thinking prefill with fakeout",
			steps: []step{
				{input: "abc</think", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc"}}},
				{input: " fakeout </think", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "</think fakeout"}}},
				{input: ">", wantEvents: []qwenEvent{}},
			},
		},
		{
			desc: "thinking prefill with spaces",
			steps: []step{
				{input: "        </think> starting content", wantEvents: []qwenEvent{qwenEventContent{content: "starting content"}}},
			},
		},
	}
	last := &api.Message{Role: "assistant", Thinking: "i am thinking"} // so if there is thinking the test is still thinking

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			parser := Qwen3VLParser{hasThinkingSupport: true}
			parser.Init([]api.Tool{}, last)

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

func TestQwen3VLThinkingParserWithNonThinkingPrefill(t *testing.T) {
	type step struct {
		input      string
		wantEvents []qwenEvent
	}

	cases := []struct {
		desc  string
		steps []step
		only  bool
	}{
		{
			desc: "thinking prefill",
			steps: []step{
				{input: "abc</think>", wantEvents: []qwenEvent{qwenEventContent{content: "abc</think>"}}},
			},
		},
		{
			desc: "thinking prefill with content",
			steps: []step{
				{input: "abc</th", wantEvents: []qwenEvent{qwenEventContent{content: "abc</th"}}},
				{input: "ink> def", wantEvents: []qwenEvent{qwenEventContent{content: "ink> def"}}},
			},
		},
		{
			desc: "thinking prefill with fakeout",
			steps: []step{
				{input: "abc</think", wantEvents: []qwenEvent{qwenEventContent{content: "abc</think"}}},
				{input: " fakeout </think", wantEvents: []qwenEvent{qwenEventContent{content: " fakeout </think"}}},
				{input: ">", wantEvents: []qwenEvent{qwenEventContent{content: ">"}}},
			},
		},
		{
			desc: "thinking prefill with spaces",
			steps: []step{
				{input: "        </think> starting content", wantEvents: []qwenEvent{qwenEventContent{content: "        </think> starting content"}}},
			},
		},
	}
	last := &api.Message{Role: "assistant", Thinking: "i am thinking", Content: "i am content"} // so if there is thinking the test is still thinking

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			parser := Qwen3VLParser{hasThinkingSupport: true}
			parser.Init([]api.Tool{}, last)

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

func TestQwen3VLThinkingParserStreamingAssistantPrefillContent(t *testing.T) {
	// last message is assistant with content ⇒ start in CollectingContent
	last := &api.Message{Role: "assistant", Content: "has content"}
	parser := Qwen3VLParser{hasThinkingSupport: true}
	parser.Init([]api.Tool{}, last)

	type step struct {
		input      string
		wantEvents []qwenEvent
	}

	steps := []step{
		{input: "abc</think>", wantEvents: []qwenEvent{qwenEventContent{content: "abc</think>"}}},
		{input: "<tool_call>{\"name\": \"x\", \"arguments\": {}}</tool_call>", wantEvents: []qwenEvent{qwenEventRawToolCall{raw: "{\"name\": \"x\", \"arguments\": {}}"}}},
	}

	for i, s := range steps {
		parser.buffer.WriteString(s.input)
		gotEvents := parser.parseEvents()
		if len(gotEvents) == 0 && len(s.wantEvents) == 0 {
			continue
		}
		if !reflect.DeepEqual(gotEvents, s.wantEvents) {
			t.Fatalf("step %d: input %q: got %#v, want %#v", i, s.input, gotEvents, s.wantEvents)
		}
	}
}
