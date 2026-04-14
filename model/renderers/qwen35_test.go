package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen35RendererUsesXMLToolCallingFormat(t *testing.T) {
	renderer := &Qwen35Renderer{isThinking: true}
	msgs := []api.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What's the weather in Paris?"},
		{
			Role:    "assistant",
			Content: "I'll check.",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgsOrdered([]orderedArg{
							{Key: "location", Value: "Paris"},
						}),
					},
				},
			},
		},
		{Role: "tool", Content: "22C"},
		{Role: "user", Content: "Thanks"},
	}
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsOrdered([]orderedProp{
						{
							Key: "location",
							Value: api.ToolProperty{
								Type: api.PropertyType{"string"},
							},
						},
					}),
					Required: []string{"location"},
				},
			},
		},
	}

	got, err := renderer.Render(msgs, tools, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	if !strings.Contains(got, "<tools>") {
		t.Fatalf("expected tools section in prompt, got:\n%s", got)
	}
	if !strings.Contains(got, "<function=example_function_name>") {
		t.Fatalf("expected xml-style tool call instructions, got:\n%s", got)
	}

	wantToolCall := "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>"
	if !strings.Contains(got, wantToolCall) {
		t.Fatalf("expected xml tool call payload, got:\n%s", got)
	}

	toolsIdx := strings.Index(got, "# Tools")
	systemIdx := strings.Index(got, "You are a helpful assistant.")
	if toolsIdx == -1 || systemIdx == -1 || systemIdx < toolsIdx {
		t.Fatalf("expected system prompt appended after tool instructions, got:\n%s", got)
	}
}

func TestQwen35RendererNoThinkPrefill(t *testing.T) {
	renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
	msgs := []api.Message{
		{Role: "user", Content: "hello"},
	}

	got, err := renderer.Render(msgs, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	if !strings.HasSuffix(got, "<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Fatalf("expected explicit no-think prefill, got:\n%s", got)
	}
}

func TestQwen35RendererBackToBackToolCallsAndResponses(t *testing.T) {
	renderer := &Qwen35Renderer{isThinking: true}

	msgs := []api.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Run add and multiply."},
		{
			Role:     "assistant",
			Content:  "I'll run both now.",
			Thinking: "Need to call add and multiply.",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "add",
						Arguments: testArgsOrdered([]orderedArg{
							{Key: "a", Value: 2},
							{Key: "b", Value: 3},
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "multiply",
						Arguments: testArgsOrdered([]orderedArg{
							{Key: "x", Value: 4},
							{Key: "y", Value: 5},
						}),
					},
				},
			},
		},
		{Role: "tool", Content: "5"},
		{Role: "tool", Content: "20"},
		{Role: "user", Content: "Summarize the results."},
	}

	got, err := renderer.Render(msgs, qwen35MathTools(), nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	if strings.Contains(got, "Need to call add and multiply.") {
		t.Fatalf("did not expect historical reasoning block in this sequence, got:\n%s", got)
	}

	wantToolCalls := `<tool_call>
<function=add>
<parameter=a>
2
</parameter>
<parameter=b>
3
</parameter>
</function>
</tool_call>
<tool_call>
<function=multiply>
<parameter=x>
4
</parameter>
<parameter=y>
5
</parameter>
</function>
</tool_call>`
	if !strings.Contains(got, wantToolCalls) {
		t.Fatalf("expected back-to-back tool calls, got:\n%s", got)
	}

	wantToolResponses := `<|im_start|>user
<tool_response>
5
</tool_response>
<tool_response>
20
</tool_response><|im_end|>`
	if !strings.Contains(got, wantToolResponses) {
		t.Fatalf("expected grouped back-to-back tool responses, got:\n%s", got)
	}

	if !strings.HasSuffix(got, "<|im_start|>assistant\n<think>\n") {
		t.Fatalf("expected assistant thinking prefill at end, got:\n%s", got)
	}
}

func TestQwen35RendererInterleavedThinkingAndTools(t *testing.T) {
	renderer := &Qwen35Renderer{isThinking: true}

	msgs := []api.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Plan a picnic in Paris."},
		{
			Role:     "assistant",
			Content:  "Checking weather first.",
			Thinking: "Need weather before giving advice.",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgsOrdered([]orderedArg{
							{Key: "location", Value: "Paris"},
						}),
					},
				},
			},
		},
		{Role: "tool", Content: "22C"},
		{
			Role:     "assistant",
			Content:  "Checking UV too.",
			Thinking: "Need UV index for sunscreen advice.",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_uv",
						Arguments: testArgsOrdered([]orderedArg{
							{Key: "location", Value: "Paris"},
						}),
					},
				},
			},
		},
		{Role: "tool", Content: "5"},
	}

	got, err := renderer.Render(msgs, qwen35WeatherUVTools(), nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	wantFirstTurn := `<|im_start|>assistant
<think>
Need weather before giving advice.
</think>

Checking weather first.

<tool_call>
<function=get_weather>
<parameter=location>
Paris
</parameter>
</function>
</tool_call><|im_end|>`
	if !strings.Contains(got, wantFirstTurn) {
		t.Fatalf("expected first assistant thinking/tool sequence, got:\n%s", got)
	}

	wantSecondTurn := `<|im_start|>assistant
<think>
Need UV index for sunscreen advice.
</think>

Checking UV too.

<tool_call>
<function=get_uv>
<parameter=location>
Paris
</parameter>
</function>
</tool_call><|im_end|>`
	if !strings.Contains(got, wantSecondTurn) {
		t.Fatalf("expected second assistant thinking/tool sequence, got:\n%s", got)
	}

	if !strings.HasSuffix(got, "<|im_start|>assistant\n<think>\n") {
		t.Fatalf("expected assistant thinking prefill at end, got:\n%s", got)
	}
}

func TestQwen35RendererAssistantPrefillWithThinking(t *testing.T) {
	renderer := &Qwen35Renderer{isThinking: true}
	msgs := []api.Message{
		{Role: "user", Content: "Write two words."},
		{
			Role:     "assistant",
			Thinking: "Keep it short.",
			Content:  "Hello world",
		},
	}

	got, err := renderer.Render(msgs, nil, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	want := `<|im_start|>user
Write two words.<|im_end|>
<|im_start|>assistant
<think>
Keep it short.
</think>

Hello world`
	if got != want {
		t.Fatalf("unexpected prefill output\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

func qwen35MathTools() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "add",
				Description: "Add two numbers",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsOrdered([]orderedProp{
						{
							Key: "a",
							Value: api.ToolProperty{
								Type: api.PropertyType{"integer"},
							},
						},
						{
							Key: "b",
							Value: api.ToolProperty{
								Type: api.PropertyType{"integer"},
							},
						},
					}),
					Required: []string{"a", "b"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "multiply",
				Description: "Multiply two numbers",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsOrdered([]orderedProp{
						{
							Key: "x",
							Value: api.ToolProperty{
								Type: api.PropertyType{"integer"},
							},
						},
						{
							Key: "y",
							Value: api.ToolProperty{
								Type: api.PropertyType{"integer"},
							},
						},
					}),
					Required: []string{"x", "y"},
				},
			},
		},
	}
}

func qwen35WeatherUVTools() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get weather for a location",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsOrdered([]orderedProp{
						{
							Key: "location",
							Value: api.ToolProperty{
								Type: api.PropertyType{"string"},
							},
						},
					}),
					Required: []string{"location"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_uv",
				Description: "Get UV index for a location",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsOrdered([]orderedProp{
						{
							Key: "location",
							Value: api.ToolProperty{
								Type: api.PropertyType{"string"},
							},
						},
					}),
					Required: []string{"location"},
				},
			},
		},
	}
}
