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

func TestQwen35RendererStructuredToolArgumentsUseSpacedJSON(t *testing.T) {
	renderer := &Qwen35Renderer{isThinking: true}

	msgs := []api.Message{
		{Role: "user", Content: "call tool"},
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "echo",
						Arguments: testArgsOrdered([]orderedArg{
							{
								Key: "payload",
								Value: map[string]any{
									"content": "if (x < 5 && y > 3) {}",
								},
							},
						}),
					},
				},
			},
		},
	}

	got, err := renderer.Render(msgs, nil, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	want := "<parameter=payload>\n{\"content\": \"if (x < 5 && y > 3) {}\"}\n</parameter>"
	if !strings.Contains(got, want) {
		t.Fatalf("expected spaced, non-escaped JSON tool argument, got:\n%s", got)
	}
}

func TestQwen35RendererToolDefinitionsUseLiteralHTMLChars(t *testing.T) {
	renderer := &Qwen35Renderer{isThinking: true}

	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Returns temperature in <fahrenheit> & <celsius>",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsOrdered([]orderedProp{
						{
							Key: "location",
							Value: api.ToolProperty{
								Type:        api.PropertyType{"string"},
								Description: "City name with <tag> & symbol",
							},
						},
						{
							Key: "filters",
							Value: api.ToolProperty{
								Type: api.PropertyType{"array"},
								Items: map[string]any{
									"type":        "string",
									"description": "Use < and > literally & keep order",
								},
							},
						},
					}),
					Required: []string{"location"},
				},
			},
		},
	}

	got, err := renderer.Render([]api.Message{{Role: "user", Content: "call tool"}}, tools, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	if strings.Contains(got, "\\u003c") || strings.Contains(got, "\\u003e") || strings.Contains(got, "\\u0026") {
		t.Fatalf("expected literal <, >, and & in tool definitions, got:\n%s", got)
	}

	want := `{"type": "function", "function": {"name": "get_weather", "description": "Returns temperature in <fahrenheit> & <celsius>", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "City name with <tag> & symbol"}, "filters": {"type": "array", "items": {"description": "Use < and > literally & keep order", "type": "string"}}}}}}`
	if !strings.Contains(got, want) {
		t.Fatalf("expected literal nested tool definition JSON, got:\n%s", got)
	}
}

// TestQwen35RendererInterleavedThinkingAndTools verifies that historical
// assistant messages with thinking content and tool calls are rendered
// correctly in both think=true and think=false modes.
//
// The think=false subtest is the primary regression detector for the fork's
// unconditional thinking block fix (commit 4044b63f). The official Qwen 3.5
// Jinja2 template (Qwen/Qwen3.5-27B on HuggingFace) checks enable_thinking
// in exactly one place — the generation prompt at the end. Historical
// assistant messages receive <think> block wrapping based solely on their
// position relative to last_query_index, with no enable_thinking check.
//
// The fork's fix has two parts that work together:
//   - splitQwen35ReasoningContent (line 65): removed the isThinking parameter
//     so reasoning extraction is unconditional — the upstream version gates on
//     isThinking, silently discarding stored reasoning when think=false.
//   - Rendering condition (line 143): removed the isThinking && prefix so
//     <think> wrapping is unconditional for messages after lastQueryIndex.
//
// If either part regresses, the think=false subtest fails:
//   - Re-adding isThinking to splitQwen35ReasoningContent causes the reasoning
//     TEXT to vanish from inside the <think> tags (extraction suppressed).
//   - Re-adding isThinking && to line 143 causes the <think> TAGS THEMSELVES
//     to vanish (wrapping suppressed).
//
// The think=true subtest would pass in both regression scenarios because
// isThinking is true, satisfying any re-added gate. This is why the pair
// of subtests is necessary: think=true alone provides zero protection.
func TestQwen35RendererInterleavedThinkingAndTools(t *testing.T) {
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

	// Both assistant messages are at indices 2 and 4, both after
	// lastQueryIndex=1 (the user message "Plan a picnic in Paris.").
	// Both have non-empty Thinking fields. The official template wraps
	// both in <think> blocks unconditionally.

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

	t.Run("think=true", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true}
		got, err := renderer.Render(msgs, qwen35WeatherUVTools(), nil)
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		if !strings.Contains(got, wantFirstTurn) {
			t.Fatalf("expected first assistant thinking/tool sequence, got:\n%s", got)
		}

		if !strings.Contains(got, wantSecondTurn) {
			t.Fatalf("expected second assistant thinking/tool sequence, got:\n%s", got)
		}

		if !strings.HasSuffix(got, "<|im_start|>assistant\n<think>\n") {
			t.Fatalf("expected assistant thinking prefill at end, got:\n%s", got)
		}
	})

	// think=false: The critical regression test for the fork's unconditional
	// thinking block fix. The renderer is constructed with isThinking: true
	// (the struct default), then Render() is called with ThinkValue{Value: false}
	// to override it at runtime — exactly as happens when a client sends
	// think: false on a request after previously using think: true.
	//
	// The historical assistant messages MUST still contain their <think> blocks
	// with the full reasoning text. The official Qwen 3.5 template never checks
	// enable_thinking when rendering historical messages. Stripping <think>
	// blocks from history when the client switches to think=false produces a
	// prompt the model was never trained on: every subsequent token ID shifts,
	// the KV cache is invalidated from the first affected message onward, and
	// the model's attention patterns break down.
	//
	// Only the generation prompt at the end should differ: think=false produces
	// <think>\n\n</think>\n\n (empty thinking block, 6 tokens matching the
	// official template) instead of think=true's <think>\n (open block for
	// the model to fill).
	t.Run("think=false", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		got, err := renderer.Render(msgs, qwen35WeatherUVTools(), &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// Verify that historical thinking blocks are preserved despite think=false.
		// This is the core assertion: the official template renders <think> blocks
		// unconditionally for messages after last_query_index.
		if !strings.Contains(got, wantFirstTurn) {
			t.Fatalf(
				"historical thinking block stripped from first assistant turn when think=false.\n\n"+
					"The official Qwen 3.5 Jinja2 template renders <think> blocks for historical "+
					"assistant messages unconditionally — enable_thinking is checked in exactly one "+
					"place (the generation prompt at the end), never in the message rendering loop. "+
					"Historical messages after last_query_index always get <think> wrapping, even "+
					"when enable_thinking is false.\n\n"+
					"If this fails, the isThinking gate was likely re-added to either "+
					"splitQwen35ReasoningContent (line 65, suppresses reasoning extraction) or "+
					"the rendering condition (line 143, suppresses <think> tag wrapping).\n\n"+
					"got:\n%s", got,
			)
		}

		if !strings.Contains(got, wantSecondTurn) {
			t.Fatalf(
				"historical thinking block stripped from second assistant turn when think=false.\n\n"+
					"Same cause as above — the isThinking gate was likely re-added. Both historical "+
					"assistant messages must retain their <think> blocks regardless of the current "+
					"turn's thinking mode.\n\n"+
					"got:\n%s", got,
			)
		}

		// The generation prompt must use the non-thinking form: an empty <think>
		// block that signals the model to skip reasoning and generate content
		// directly. This is the ONLY part of the output that should differ between
		// think=true and think=false.
		wantSuffix := "<|im_start|>assistant\n<think>\n\n</think>\n\n"
		if !strings.HasSuffix(got, wantSuffix) {
			tail := got
			if len(tail) > 200 {
				tail = tail[len(tail)-200:]
			}
			t.Fatalf(
				"wrong generation prompt suffix for think=false.\n\n"+
					"When thinking is disabled, the generation prompt must include an empty thinking "+
					"block (<think>\\n\\n</think>\\n\\n — 6 tokens matching the official template's "+
					"add_generation_prompt block for enable_thinking=false). Without this, the model "+
					"may attempt reasoning when the caller explicitly requested no thinking.\n\n"+
					"expected suffix: %q\ngot tail: %q", wantSuffix, tail,
			)
		}
	})
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

// TestQwen35RendererAssistantToolCallIsNotPrefill verifies that when the last
// message in a conversation is an assistant message containing tool calls, the
// renderer properly closes the turn with <|im_end|> and appends a generation
// prompt — never treating it as a prefill continuation.
//
// This is the counterpart to TestQwen35RendererAssistantPrefillWithThinking,
// which verifies the correct prefill case (last message is assistant WITHOUT
// tool calls — the model should continue the partial text).
//
// The official Qwen 3.5 Jinja2 template (Qwen/Qwen3.5-27B on HuggingFace)
// emits <|im_end|> unconditionally after every assistant message. The
// generation prompt is a separate concern, emitted after all messages. Ollama's
// Go renderer infers prefill from the message structure: if the last message is
// an assistant message, it assumes the client is providing partial text for
// continuation. The critical guard is that this heuristic must NOT apply to
// assistant messages with tool calls — those are complete turns, not partial
// prefills. Without the guard, the model sees an unclosed assistant turn ending
// in </tool_call> with no generation prompt: a token sequence absent from all
// training data, causing undefined model behavior in exactly the agentic
// tool-calling loops where Qwen 3.5 is most valuable.
func TestQwen35RendererAssistantToolCallIsNotPrefill(t *testing.T) {
	weatherToolCall := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_weather",
			Arguments: testArgsOrdered([]orderedArg{
				{Key: "location", Value: "Paris"},
			}),
		},
	}

	t.Run("think=true", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true}
		msgs := []api.Message{
			{Role: "user", Content: "What's the weather in Paris?"},
			{
				Role:      "assistant",
				Content:   "Let me check the weather.",
				Thinking:  "I should use the weather tool.",
				ToolCalls: []api.ToolCall{weatherToolCall},
			},
		}

		got, err := renderer.Render(msgs, nil, nil)
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// The assistant turn must be closed. The official Qwen 3.5 Jinja2 template
		// emits <|im_end|> unconditionally after every assistant message. Without it,
		// the model sees an unclosed turn ending in </tool_call> — a prompt shape
		// absent from all training data. This typically means the prefill condition
		// is incorrectly treating assistant messages with tool calls as partial
		// continuations.
		//
		// This scenario triggers when agentic frameworks send conversation history
		// where the last message is the assistant's tool-calling turn, with tool
		// results arriving in a separate subsequent request.
		if !strings.Contains(got, "</tool_call><|im_end|>") {
			t.Errorf(
				"missing <|im_end|> after the assistant's tool call.\n\n"+
					"The official Qwen 3.5 Jinja2 template emits <|im_end|> unconditionally after every "+
					"assistant message. Without it, the model sees an unclosed assistant turn ending in "+
					"</tool_call> — a prompt shape absent from all training data. This typically means "+
					"the prefill condition is incorrectly treating assistant messages with tool calls as "+
					"partial continuations.\n\n"+
					"This scenario triggers when agentic frameworks send conversation history where the "+
					"last message is the assistant's tool-calling turn, with tool results arriving in a "+
					"separate subsequent request.\n\ngot:\n%s", got,
			)
		}

		// A completed assistant turn with tool calls must be followed by a generation
		// prompt so the model knows to begin a new response. Without it, the model
		// has no <|im_start|>assistant token to start generating, and inference
		// behavior is undefined.
		wantSuffix := "<|im_start|>assistant\n<think>\n"
		if !strings.HasSuffix(got, wantSuffix) {
			tail := got
			if len(tail) > 120 {
				tail = tail[len(tail)-120:]
			}
			t.Errorf(
				"missing generation prompt after the closed assistant turn.\n\n"+
					"When the last message is a completed assistant turn with tool calls, the renderer must "+
					"append a generation prompt (<|im_start|>assistant\\n<think>\\n) so the model begins a "+
					"new response. Without it, the model has no signal to start generating and inference "+
					"behavior is undefined.\n\n"+
					"expected suffix: %q\ngot tail: %q", wantSuffix, tail,
			)
		}

		want := `<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<think>
I should use the weather tool.
</think>

Let me check the weather.

<tool_call>
<function=get_weather>
<parameter=location>
Paris
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>assistant
<think>
`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch.\n\n"+
					"Any byte-level deviation changes the token IDs the model sees, pushing the input out "+
					"of the training distribution. In multi-turn agentic conversations, deviations also "+
					"invalidate the KV cache from the divergence point, forcing expensive recomputation "+
					"of all subsequent tokens.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})

	t.Run("think=false", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		msgs := []api.Message{
			{Role: "user", Content: "What's the weather in Paris?"},
			{
				Role:      "assistant",
				Content:   "I'll check the weather.",
				ToolCalls: []api.ToolCall{weatherToolCall},
			},
		}

		got, err := renderer.Render(msgs, nil, &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// Same <|im_end|> check as think=true — the official template closes every
		// assistant message unconditionally regardless of thinking mode.
		if !strings.Contains(got, "</tool_call><|im_end|>") {
			t.Errorf(
				"missing <|im_end|> after the assistant's tool call in think=false mode.\n\n"+
					"The official Qwen 3.5 template closes every assistant message unconditionally, "+
					"regardless of whether thinking is enabled. Without <|im_end|>, the model sees an "+
					"unclosed turn — a prompt shape it was never trained on.\n\ngot:\n%s", got,
			)
		}

		// When thinking is disabled, the generation prompt includes an empty thinking
		// block so the model skips reasoning and proceeds directly to content. Without
		// this prompt, the model may attempt reasoning when the caller explicitly
		// requested no thinking, or may fail to generate altogether.
		wantSuffix := "<|im_start|>assistant\n<think>\n\n</think>\n\n"
		if !strings.HasSuffix(got, wantSuffix) {
			tail := got
			if len(tail) > 120 {
				tail = tail[len(tail)-120:]
			}
			t.Errorf(
				"missing non-thinking generation prompt after the closed assistant turn.\n\n"+
					"When thinking is disabled, the generation prompt includes an empty thinking block "+
					"(<think>\\n\\n</think>\\n\\n) so the model skips reasoning and proceeds directly to "+
					"content generation. Without this prompt, the model may attempt reasoning when the "+
					"caller explicitly requested no thinking, or may fail to generate altogether.\n\n"+
					"expected suffix: %q\ngot tail: %q", wantSuffix, tail,
			)
		}

		want := `<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<think>

</think>

I'll check the weather.

<tool_call>
<function=get_weather>
<parameter=location>
Paris
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>assistant
<think>

</think>

`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch (think=false).\n\n"+
					"Any byte-level deviation changes the token IDs the model sees, pushing the input out "+
					"of the training distribution. In multi-turn agentic conversations, deviations also "+
					"invalidate the KV cache from the divergence point, forcing expensive recomputation "+
					"of all subsequent tokens.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})
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
