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
		{
			desc: "with thinking",
		},
		{
			desc: "thinking and tool call",
		},
		{
			desc: "thinking and content",
		},
		{
			desc: "thinking and tool call and content",
		},
		{
			desc: "nested thinking (outside thinking, inside thinking)",
		},
		{
			desc: "interleaved thinking",
		},
		{
			desc: "nested thinking and tool call (outside thinking, inside tool call)",
		},
		{
			desc: "nested thinking and tool call (inside tool call, outside thinking)",
		},
		{
			desc: "interleaved thinking and tool call",
		},
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
