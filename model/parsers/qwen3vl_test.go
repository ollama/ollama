package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

// tool creates a test tool with the given name and properties
// func tool(name string, props map[string]api.ToolProperty) api.Tool {
// 	t := api.Tool{Type: "function", Function: api.ToolFunction{Name: name}}
// 	t.Function.Parameters.Type = "object"
// 	t.Function.Parameters.Properties = props
// 	return t
// }

func TestQwen3VLParserStreaming(t *testing.T) {
	type step struct {
		input      string
		wantEvents []qwenEvent
	}

	cases := []struct {
		desc  string
		steps []step
		only  bool
	}{
		// all of this is just thinking tests
		{
			desc: "simple thinking",
			steps: []step{
				{input: "<thinking>abc</thinking>", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc"}}},
			},
		},
		{
			desc: "thinking with split tags",
			steps: []step{
				{input: "<thinking>abc", wantEvents: []qwenEvent{}},
				{input: "</thinking>", wantEvents: []qwenEvent{qwenEventThinkingContent{content: "abc"}}},
			},
		},
		{
			desc: "thinking and tool call",
			steps: []step{
				{
					input: "<thinking>I'm thinking</thinking><tool_call>I'm tool calling</tool_call>",
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
					input: "<thinking>I'm thinking</thinking>I'm content",
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
					input: "<thinking>I'm thinking<thinking>I'm nested thinking</thinking></thinking>",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "I'm thinking<thinking>I'm nested thinking"},
						qwenEventContent{content: "</thinking>"},
					},
				},
			},
		},
		{
			desc: "interleaved thinking",
			steps: []step{
				{
					input: "<thinking>I'm thinking<thinking></thinking>I'm actually content</thinking>",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "I'm thinking<thinking>"},
						qwenEventContent{content: "I'm actually content</thinking>"},
					},
				},
			},
		},
		{
			desc: "nested thinking and tool call (outside thinking, inside tool call)",
			steps: []step{
				{
					input:      "<thinking>I'm thinking<tool_call>I'm nested tool call</tool_call></thinking>",
					wantEvents: []qwenEvent{qwenEventThinkingContent{content: "I'm thinking<tool_call>I'm nested tool call</tool_call>"}},
				},
			},
		},
		{
			desc: "nested thinking and tool call (outside tool call, inside thinking)",
			steps: []step{
				{
					input:      "<tool_call>I'm nested tool call<thinking>I'm thinking</thinking></tool_call>",
					wantEvents: []qwenEvent{qwenEventRawToolCall{raw: "I'm nested tool call<thinking>I'm thinking</thinking>"}},
				},
			},
		},
		{
			desc: "interleaved thinking and tool call",
			steps: []step{
				{
					input: "<thinking>I'm thinking<tool_call>I'm NOT a nested tool call</thinking></tool_call><tool_call>I'm nested tool call 2<thinking></tool_call></thinking>",
					wantEvents: []qwenEvent{
						qwenEventThinkingContent{content: "I'm thinking<tool_call>I'm NOT a nested tool call"},
						qwenEventContent{content: "</tool_call>"},
						qwenEventRawToolCall{raw: "I'm nested tool call 2<thinking>"},
						qwenEventContent{content: "</thinking>"},
					},
				},
			},
		},
		{
			desc: "partial thinking tag fakeout",
			steps: []step{
				{
					input: "abc<thinking",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "abc"},
					},
				},
				{
					input: " fakeout",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "<thinking fakeout"},
					},
				},
			},
		},
		{
			desc: "partial thinking incomplete",
			steps: []step{
				{
					input: "abc<thinking>unfinished</thinking", // when something is ambiguious, we dont emit anything
					wantEvents: []qwenEvent{
						qwenEventContent{content: "abc"},
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
			parser := Qwen3VLParser{}

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

func TestQwen3VLComplex(t *testing.T) {
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
			desc: "simple tool call",
			steps: []step{
				{
					input:      "Here are 30 distinct and popular emojis for you! 😊\n\n1. 😂  \n2. ❤️  \n3. 🌟  \n4. 🐶  \n5. 🍕  \n6. ✨  \n7. 🌈  \n8. 🎉  \n9. 🌎  \n10. 🦁  \n11. 💯  \n12. 🥰  \n13. 🌸  \n14. 🚀  \n15. 🌊  \n16. 🍦  \n17. 🌙  \n18. 🌞  \n19. 🌻  \n20. 🦋  \n21. 🍃  \n22. 🏆  \n23. 🌮  \n24. 🧸  \n25. 🎮  \n26. 📚  \n27. ✈️  \n28. 🌟 (sparkles)  \n29. 🌈 (rainbow)  \n30. 🥳  \n\n*Bonus fun fact:* The 😂 (Face with Tears of Joy) was Oxford Dictionaries' Word of the Year in 2015! 🎉  \nLet me know if you'd like themed emojis (e.g., animals, food, or emotions)! 🐱🍕📚",
					wantEvents: []qwenEvent{qwenEventContent{content: "bruh"}},
				},
			},
		},
	}
	for _, tc := range cases {
		for i, step := range tc.steps {
			parser := Qwen3VLParser{}
			parser.buffer.WriteString(step.input)
			gotEvents := parser.parseEvents()
			if !reflect.DeepEqual(gotEvents, step.wantEvents) {
				t.Errorf("step %d: input %q: got events %#v, want %#v", i, step.input, gotEvents, step.wantEvents)
			}
		}
	}
}

// TODO: devin was saying something about json cant figure out types?
// do we need to test for
func TestQwen3VLToolParser(t *testing.T) {
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
			rawToolCall: `{"function": {"name": "get-current-weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}}`,
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
			rawToolCall: `{"function": {"name": "get current temperature", "arguments": {"location with spaces": "San Francisco", "unit with spaces": "celsius"}}}`,
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
			rawToolCall: `{"function": {"name": "\"get current temperature\"", "arguments": {"\"location with spaces\"": "San Francisco", "\"unit with spaces\"": "\"celsius\""}}}`,
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
			rawToolCall: `{"function": {"name": "calculate", "arguments": {"x": 3.14, "y": 42, "enabled": true, "items": ["a", "b", "c"]}}}`,
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
			rawToolCall: `{"function": {"name": "exec", "arguments": {"command": "ls && echo \"done\""}}}`,
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
			rawToolCall: `{"function": {"name": "exec", "arguments": {"command": "ls && echo \"a > b and a < b\""}}}`,
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
			rawToolCall: `{"function": {"name": "获取天气", "arguments": {"城市": "北京", "message": "Hello! 你好! 🌟 مرحبا"}}}`,
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
