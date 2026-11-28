package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// TestStreamingSplitInvariance verifies that parser output does not depend on
// how the response is chunked: feeding a response in every possible two-way
// split must yield the same content, thinking, and tool calls as feeding it
// whole. See https://github.com/ollama/ollama/issues/18681.
func TestStreamingSplitInvariance(t *testing.T) {
	tool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: "get_weather",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: testPropsMap(map[string]api.ToolProperty{"city": {Type: []string{"string"}}}),
			},
		},
	}

	tests := []struct {
		name     string
		parser   Parser
		tools    []api.Tool
		text     string
	}{
		{
			name:   "deepseek3 tool call",
			parser: &DeepSeek3Parser{hasThinkingSupport: true},
			tools:  []api.Tool{tool},
			text:   "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\":\"Oslo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
		},
		{
			name:   "deepseek3 content before tool call",
			parser: &DeepSeek3Parser{hasThinkingSupport: true},
			tools:  []api.Tool{tool},
			text:   "Checking the weather.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\":\"Oslo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜> Done.",
		},
		{
			name:   "cogito tool call",
			parser: &CogitoParser{},
			tools:  []api.Tool{tool},
			text:   "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\":\"Oslo\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
		},
		{
			name:   "lfm2 tool call",
			parser: &LFM2Parser{hasThinkingSupport: false},
			tools:  []api.Tool{tool},
			text:   "<|tool_call_start|>[get_weather(city='Oslo')]<|tool_call_end|>",
		},
		{
			name:   "lfm2 content before tool call",
			parser: &LFM2Parser{hasThinkingSupport: false},
			tools:  []api.Tool{tool},
			text:   "Sure, let me check.<|tool_call_start|>[get_weather(city='Oslo')]<|tool_call_end|>",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			refParser := tc.parser
			refParser.Init(tc.tools, nil, nil)
			refContent, refThinking, refCalls, err := refParser.Add(tc.text, true)
			if err != nil {
				t.Fatalf("whole parse failed: %v", err)
			}

			for i := 1; i < len(tc.text); i++ {
				p := tc.parser
				p.Init(tc.tools, nil, nil)
				var content, thinking strings.Builder
				var calls []api.ToolCall
				c1, t1, c1alls, err1 := p.Add(tc.text[:i], false)
				c2, t2, c2alls, err2 := p.Add(tc.text[i:], true)
				if err1 != nil || err2 != nil {
					t.Fatalf("split %d: chunk add failed: %v %v", i, err1, err2)
				}
				calls = append(calls, c1alls...)
				calls = append(calls, c2alls...)
				content.WriteString(c1)
				content.WriteString(c2)
				thinking.WriteString(t1)
				thinking.WriteString(t2)

				if content.String() != refContent || thinking.String() != refThinking || !toolCallsEqual(calls, refCalls) {
					t.Fatalf("split %d mismatch:\nwhole: content=%q thinking=%q calls=%v\nsplit: content=%q thinking=%q calls=%v",
						i, refContent, refThinking, refCalls, content.String(), thinking.String(), calls)
				}
			}
		})
	}
}
