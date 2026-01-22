package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestGLM46Renderer(t *testing.T) {
	tests := []struct {
		name       string
		messages   []api.Message
		tools      []api.Tool
		thinkValue *api.ThinkValue
		expected   string
		skip       string
	}{
		{
			name: "basic",
			messages: []api.Message{
				{Role: "user", Content: "Hello, how are you?"},
			},
			expected: `[gMASK]<sop><|user|>
Hello, how are you?<|assistant|>`,
		},
		{
			name: "basic with system message",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello, how are you?"},
			},
			expected: `[gMASK]<sop><|system|>
You are a helpful assistant.<|user|>
Hello, how are you?<|assistant|>`,
		},
		{
			name: "basic with user assistant user",
			messages: []api.Message{
				{Role: "user", Content: "What is the capital of France?"},
				{Role: "assistant", Thinking: "Let me analyze the request...", Content: "The capital of France is Paris."},
				{Role: "user", Content: "Fantastic!"},
			},
			expected: `[gMASK]<sop><|user|>
What is the capital of France?<|assistant|>
The capital of France is Paris.<|user|>
Fantastic!<|assistant|>`,
		},
		{
			skip: "tool call ordering not guaranteed yet",
			name: "tools",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant with access to tools."},
				{Role: "user", Content: "What is the weather like in Tokyo?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather in a given location",
						Parameters: api.ToolFunctionParameters{
							Type:       "object",
							Required:   []string{"location"},
							Properties: propsMap(`{"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}`),
						},
					},
				},
			},
			expected: `[gMASK]<sop><|system|>
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"get_weather","description":"Get the current weather in a given location","parameters":{"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","description":"","enum":["celsius","fahrenheit"]}}}}}
</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}
<arg_key>{arg-key-1}</arg_key>
<arg_value>{arg-value-1}</arg_value>
<arg_key>{arg-key-2}</arg_key>
<arg_value>{arg-value-2}</arg_value>
...
</tool_call><|system|>
You are a helpful assistant with access to tools.<|user|>
What is the weather like in Tokyo?<|assistant|>`,
		},
		{
			skip: "tool call ordering not guaranteed yet",
			name: "tool calls",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant with access to tools."},
				{Role: "user", Content: "What is the weather like in Tokyo?"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: args(`{"location": "Tokyo, Japan", "unit": "celsius"}`),
							},
						},
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: args(`{"location": "Japan", "unit": "fahrenheit"}`),
							},
						},
					},
				},
				{
					Role:     "tool",
					Content:  "{\"temperature\": 22, \"weather\": \"partly cloudy\", \"humidity\": 65}",
					ToolName: "get_weather",
				},
				{
					Role:     "tool",
					Content:  "{\"temperature\": 68, \"weather\": \"sunny\", \"humidity\": 75}",
					ToolName: "get_weather",
				},
				{
					Role:    "assistant",
					Content: "The weather in Tokyo is currently partly cloudy with a temperature of 22°C and 65% humidity. It's a pleasant day with moderate temperatures.",
				},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather in a given location",
						Parameters: api.ToolFunctionParameters{
							Type:       "object",
							Required:   []string{"location"},
							Properties: propsMap(`{"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}`),
						},
					},
				},
			},
			expected: `[gMASK]<sop><|system|>
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"get_weather","description":"Get the current weather in a given location","parameters":{"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","description":"","enum":["celsius","fahrenheit"]}}}}}
</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}
<arg_key>{arg-key-1}</arg_key>
<arg_value>{arg-value-1}</arg_value>
<arg_key>{arg-key-2}</arg_key>
<arg_value>{arg-value-2}</arg_value>
...
</tool_call><|system|>
You are a helpful assistant with access to tools.<|user|>
What is the weather like in Tokyo?<|assistant|>
<think></think>
<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>Tokyo, Japan</arg_value>
<arg_key>unit</arg_key>
<arg_value>celsius</arg_value>
</tool_call>
<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>Japan</arg_value>
<arg_key>unit</arg_key>
<arg_value>fahrenheit</arg_value>
</tool_call><|observation|>
<tool_response>
{"temperature": 22, "weather": "partly cloudy", "humidity": 65}
</tool_response>
<tool_response>
{"temperature": 68, "weather": "sunny", "humidity": 75}
</tool_response><|assistant|>
<think></think>
The weather in Tokyo is currently partly cloudy with a temperature of 22°C and 65% humidity. It's a pleasant day with moderate temperatures.<|assistant|>`,
		},
		{
			name: "think true",
			messages: []api.Message{
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `[gMASK]<sop><|user|>
Hello, how are you?<|assistant|>`,
		},
		{
			name: "think false",
			messages: []api.Message{
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `[gMASK]<sop><|user|>
Hello, how are you?/nothink<|assistant|>
<think></think>
`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.skip != "" {
				t.Skip(tt.skip)
			}
			renderer := &GLM46Renderer{}
			rendered, err := renderer.Render(tt.messages, tt.tools, tt.thinkValue)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
				t.Logf("Got:\n%s", rendered)
				t.Logf("Expected:\n%s", tt.expected)
			}
		})
	}
}
