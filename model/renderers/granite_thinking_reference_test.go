package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

// Reference strings captured by rendering ibm-granite/granite-4.2-3b's actual
// chat_template.jinja with jinja2 3.1.5, to keep the Go implementation in
// sync with the real template rather than a re-derivation of it.
func TestGraniteThinkingRenderReference(t *testing.T) {
	boolThink := func(b bool) *api.ThinkValue { return &api.ThinkValue{Value: b} }

	weatherTool := api.Tool{Type: "function", Function: api.ToolFunction{
		Name:        "get_weather",
		Description: "Get the weather",
	}}
	weatherTool.Function.Parameters.Type = "object"
	weatherTool.Function.Parameters.Properties = api.NewToolPropertiesMap()
	weatherTool.Function.Parameters.Properties.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "City name"})
	weatherTool.Function.Parameters.Required = []string{"location"}

	toolCall := api.ToolCall{Function: api.ToolCallFunction{Name: "get_weather", Arguments: api.NewToolCallFunctionArguments()}}
	toolCall.Function.Arguments.Set("location", "Paris")

	toolsPreamble := "# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
		`{"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "City name"}}}}` +
		"\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

	cases := []struct {
		name  string
		msgs  []api.Message
		tools []api.Tool
		think *api.ThinkValue
		want  string
	}{
		{
			name: "system_and_user",
			msgs: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hi"},
			},
			think: boolThink(true),
			want:  "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "thinking_disabled",
			msgs: []api.Message{
				{Role: "user", Content: "Hi"},
			},
			think: boolThink(false),
			want:  "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>",
		},
		{
			name: "tools_no_system",
			msgs: []api.Message{
				{Role: "user", Content: "Weather in Paris?"},
			},
			tools: []api.Tool{weatherTool},
			think: boolThink(true),
			want: "<|im_start|>system\n" + toolsPreamble + "<|im_end|>\n" +
				"<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "history_tool_call_truncates_reasoning",
			msgs: []api.Message{
				{Role: "user", Content: "Weather in Paris?"},
				{Role: "assistant", Thinking: "I should call the weather tool.", ToolCalls: []api.ToolCall{toolCall}},
				{Role: "tool", Content: `{"temp": 15}`},
				{Role: "user", Content: "And London?"},
			},
			tools: []api.Tool{weatherTool},
			think: boolThink(true),
			want: "<|im_start|>system\n" + toolsPreamble + "<|im_end|>\n" +
				"<|im_start|>user\nWeather in Paris?<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n" +
				"<|im_start|>user\n<tool_response>\n{\"temp\": 15}\n</tool_response>\n<|im_end|>\n" +
				"<|im_start|>user\nAnd London?<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
		{
			name: "history_plain_content",
			msgs: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello there!"},
				{Role: "user", Content: "How are you?"},
			},
			think: boolThink(true),
			want: "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>Hello there!<|im_end|>\n" +
				"<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n<think>\n",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RenderWithRenderer("granite-thinking", tc.msgs, tc.tools, tc.think)
			if err != nil {
				t.Fatalf("Render error: %v", err)
			}
			if got != tc.want {
				t.Errorf("Render mismatch\n got:  %q\nwant: %q", got, tc.want)
			}
		})
	}
}
