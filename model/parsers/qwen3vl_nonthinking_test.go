package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen3VLNonThinkingParserStreaming(t *testing.T) {
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
				{input: "abc</think>", wantEvents: []qwenEvent{qwenEventContent{content: "abc</think>"}}},
			},
		},
		{
			desc: "simple trip thinking",
			steps: []step{
				{input: "<think>abc</think>", wantEvents: []qwenEvent{qwenEventContent{content: "<think>abc</think>"}}},
			},
		},
		{
			desc: "thinking with split tags",
			steps: []step{
				{input: "abc", wantEvents: []qwenEvent{qwenEventContent{content: "abc"}}},
				{input: "</think>", wantEvents: []qwenEvent{qwenEventContent{content: "</think>"}}},
			},
		},
		{
			desc: "multiple think tags",
			steps: []step{
				{input: "abc<think>actually, is not thinking</think>", wantEvents: []qwenEvent{qwenEventContent{content: "abc<think>actually, is not thinking</think>"}}},
			},
		},
		{
			desc: "thinking and tool call",
			steps: []step{
				{
					input: "I'm thinking</think><tool_call>I'm tool calling</tool_call>",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "I'm thinking</think>"},
						qwenEventRawToolCall{raw: "I'm tool calling"},
					},
				},
			},
		},
		{
			desc: "nested thinking (outside thinking, inside thinking)",
			steps: []step{
				{
					input: "I'm thinking<think>I'm nested thinking</think></think>",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "I'm thinking<think>I'm nested thinking</think></think>"},
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
						qwenEventContent{content: "<think>I'm thinking</think>I'm actually content</think>"},
					},
				},
			},
		},
		{
			desc: "nested thinking and tool call (outside thinking, inside tool call)",
			steps: []step{
				{
					input: "I'm thinking<tool_call>I'm nested tool call</tool_call></think>",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "I'm thinking"},
						qwenEventRawToolCall{raw: "I'm nested tool call"},
						qwenEventContent{content: "</think>"},
					},
				},
			},
		},
		{
			desc: "nested thinking and tool call (outside tool call, inside thinking)",
			steps: []step{
				{
					input: "<tool_call>I'm nested tool call<think>I'm thinking</think></tool_call>",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "I'm nested tool call<think>I'm thinking</think>"},
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
						qwenEventContent{content: "I'm thinking"},
						qwenEventRawToolCall{raw: "I'm NOT a nested tool call</think>"},
						qwenEventRawToolCall{raw: "I'm nested tool call 2<think>"},
						qwenEventContent{content: "</think>"},
					},
				},
			},
		},
		{
			desc: "emit unambiguous before partial tool open (trailing ws)",
			steps: []step{
				{
					input:      "abc\u00a0\n<tool_call",
					wantEvents: []qwenEvent{qwenEventContent{content: "abc"}},
				},
				{
					input:      " fakeout",
					wantEvents: []qwenEvent{qwenEventContent{content: "\u00a0\n<tool_call fakeout"}},
				},
			},
		},
		{
			desc: "unambiguous empty: partial tool open at buffer start",
			steps: []step{
				{
					input:      "<tool_ca",
					wantEvents: []qwenEvent{},
				},
				{
					input: "ll>abc</tool_call>",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "abc"},
					},
				},
			},
		},
		{
			desc: "partial thinking tag fakeout",
			steps: []step{
				{
					input:      "abc</think",
					wantEvents: []qwenEvent{qwenEventContent{content: "abc</think"}},
				},
				{
					input:      " fakeout",
					wantEvents: []qwenEvent{qwenEventContent{content: " fakeout"}},
				},
			},
		},
		{
			desc: "partial thinking incomplete",
			steps: []step{
				{
					input:      "abc<think>unfinished<", // when something is ambiguious, we dont emit anything
					wantEvents: []qwenEvent{qwenEventContent{content: "abc<think>unfinished"}},
				},
			},
		},
		{
			desc: "test with split tool and content",
			steps: []step{
				{
					input: "abc<tool_call>unfinished</", // when something is ambiguious, we dont emit anything
					wantEvents: []qwenEvent{
						qwenEventContent{content: "abc"},
					},
				},
				{
					input: "tool_call> def",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "unfinished"},
						qwenEventContent{content: "def"},
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
			parser := Qwen3VLParser{hasThinkingSupport: false}
			parser.Init([]api.Tool{}, nil)

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

func TestQwenOldParserStreaming(t *testing.T) {
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
			desc: "simple message streamed word by word",
			steps: []step{
				{
					input:      "hi",
					wantEvents: []qwenEvent{qwenEventContent{content: "hi"}},
				},
				{
					input:      " there",
					wantEvents: []qwenEvent{qwenEventContent{content: " there"}},
				},
			},
		},
		{
			desc: "content before tool call",
			steps: []step{
				{
					input:      "hi there<tool_call>",
					wantEvents: []qwenEvent{qwenEventContent{content: "hi there"}},
				},
			},
		},
		{
			desc: "multiple tool calls in one message",
			steps: []step{
				{
					input: "before1<tool_call>in tool call</tool_call>after1<tool_call>in tool call 2</tool_call>after2",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "before1"},
						qwenEventRawToolCall{raw: "in tool call"},
						qwenEventContent{content: "after1"},
						qwenEventRawToolCall{raw: "in tool call 2"},
						qwenEventContent{content: "after2"},
					},
				},
			},
		},
		{
			desc: "tool calls with split tags",
			steps: []step{
				{
					input: "before<tool",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "before"},
					},
				},
				{
					input:      "_call>in tool call</tool",
					wantEvents: []qwenEvent{},
				},
				{
					input: "_call>af",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "in tool call"},
						qwenEventContent{content: "af"},
					},
				},
				{
					input: "ter",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "ter"},
					},
				},
			},
		},
		{
			desc: "trailing whitespace between content and tool call",
			steps: []step{
				{
					input: "abc\n<tool_call>def</tool_call>",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "abc"},
						qwenEventRawToolCall{raw: "def"},
					},
				},
			},
		},
		{
			desc: "trailing whitespace between tool call and content",
			steps: []step{
				{
					input: "<tool_call>abc</tool_call>\ndef",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "abc"},
						qwenEventContent{content: "def"},
					},
				},
			},
		},
		{
			desc: "empty content before tool call",
			steps: []step{
				{
					input: "\n<tool_call>abc</tool_call>",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "abc"},
					},
				},
			},
		},
		{
			desc: "partial tool open tag fakeout",
			steps: []step{
				{
					input: "abc\n<tool_call",
					wantEvents: []qwenEvent{
						// \n should not be emitted yet because `<tool_call` might be a tool
						// open tag, in which case the whitespace should be trimmed
						qwenEventContent{content: "abc"},
					},
				},
				{
					input: " fakeout",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "\n<tool_call fakeout"},
					},
				},
			},
		},
		{
			desc: "token-by-token whitespace handling",
			steps: []step{
				{
					input: "a",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "a"},
					},
				},
				{
					input:      "\n",
					wantEvents: []qwenEvent{},
				},
				{
					input: "b",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "\nb"},
					},
				},
			},
		},
		{
			desc: "unicode content",
			steps: []step{
				{
					input: "你好 🌍<tool_call>test</tool_call>مرحبا",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "你好 🌍"},
						qwenEventRawToolCall{raw: "test"},
						qwenEventContent{content: "مرحبا"},
					},
				},
			},
		},
		{
			desc: "arabic text handling",
			steps: []step{
				{
					input:      "مرحبا بالعالم",
					wantEvents: []qwenEvent{qwenEventContent{content: "مرحبا بالعالم"}},
				},
			},
		},
		{
			desc: "emoji passthrough",
			steps: []step{
				{
					input:      "✅",
					wantEvents: []qwenEvent{qwenEventContent{content: "✅"}},
				},
			},
		},
		{
			desc: "emoji after tool call",
			steps: []step{
				{
					input: "<tool_call>test</tool_call>完成 ✅",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "test"},
						qwenEventContent{content: "完成 ✅"},
					},
				},
			},
		},
		{
			desc: "unicode streaming with whitespace handling",
			steps: []step{
				{
					input: "مرحبا",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "مرحبا"},
					},
				},
				{
					input:      " \n",
					wantEvents: []qwenEvent{},
				},
				{
					input: "世界",
					wantEvents: []qwenEvent{
						qwenEventContent{content: " \n世界"},
					},
				},
			},
		},
		{
			desc: "non-breaking space withheld across chunks",
			steps: []step{
				{
					input: "Hello\u00a0",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "Hello"},
					},
				},
				{
					input: "world",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "\u00a0world"},
					},
				},
			},
		},
		{
			desc: "ideographic space before partial tool",
			steps: []step{
				{
					input: "Hello\u3000<tool",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "Hello"},
					},
				},
				{
					input:      "_call>abc",
					wantEvents: []qwenEvent{},
				},
				{
					input: "</tool_call>def",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "abc"},
						qwenEventContent{content: "def"},
					},
				},
			},
		},
		{
			desc: "ideographic space before partial tool fakeout",
			steps: []step{
				{
					input: "Hello\u3000<tool",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "Hello"},
					},
				},
				{
					input: "fakeout>abc",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "\u3000<toolfakeout>abc"},
					},
				},
			},
		},
		{
			desc: "unicode with partial tool tag",
			steps: []step{
				{
					input: "测试🎯 <to",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "测试🎯"},
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
			parser := Qwen3VLParser{hasThinkingSupport: false}
			parser.Init([]api.Tool{}, nil)

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

func TestQwen3VLNonThinkingToolParser(t *testing.T) {
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
