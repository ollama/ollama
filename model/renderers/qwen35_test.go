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

// TestSplitQwen35ReasoningContent tests the function that extracts reasoning
// and remaining content from an assistant message. This function's output
// determines the exact bytes the model sees inside <think>...\n</think> and
// after the \n\n that follows, for every historical assistant message. The
// renderer wraps the output as:
//
//	<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{remaining}
//
// The official Qwen 3.5 Jinja2 template (Qwen/Qwen3.5-27B on HuggingFace)
// has the same two-path extraction structure:
//
//	Path 1: if message.reasoning_content is string → use it directly
//	Path 2: elif '</think>' in content →
//	           reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
//	           content   = content.split('</think>')[-1].lstrip('\n')
//	Final:  reasoning_content = reasoning_content|trim
//
// The fork's Go equivalent uses strings.Index (first </think>),
// strings.LastIndex (last <think> before it), strings.TrimLeft("\n") on
// remaining content, and strings.TrimSpace on reasoning. These are equivalent
// to the template's operations for all realistic inputs (single </think> tag).
//
// The most important property tested here is path equivalence: different
// clients encode the same model turn differently (explicit Thinking field vs
// inline <think> tags in Content), and all encodings must produce identical
// (reasoning, remaining) tuples. If they don't, switching clients
// mid-conversation — or a client upgrading how it stores thinking — changes
// the rendered prompt, which means the model sees different token IDs for the
// same conversation history. This causes silent quality degradation (the model
// receives a prompt shape it was not trained on) and KV cache invalidation
// (every byte difference in the historical portion forces expensive
// recomputation of all subsequent tokens).
func TestSplitQwen35ReasoningContent(t *testing.T) {
	tests := []struct {
		name            string
		content         string
		messageThinking string
		wantReasoning   string
		wantRemaining   string
	}{
		// Test 1a-c: Path equivalence. Three different clients encode the same
		// model turn (reasoning="Let me check the weather", visible="Paris is
		// 18°C.") in three different ways. All three must produce identical
		// output so the renderer emits the same bytes regardless of which
		// client sent the history.
		{
			// Client A: well-behaved client that stores the Thinking field
			// separately from Content, as Ollama's parser delivers them.
			// This is Path 1 (explicit field).
			name:            "equivalence/explicit_thinking_field",
			content:         "Paris is 18°C.",
			messageThinking: "Let me check the weather",
			wantReasoning:   "Let me check the weather",
			wantRemaining:   "Paris is 18°C.",
		},
		{
			// Client B: third-party client that stores the full model output
			// including the prefilled <think> open tag as a single content
			// string, with the Thinking field absent (empty in Go). The \n
			// after <think> and \n before </think> reflect how the model
			// actually generates: the renderer prefills "<think>\n", the model
			// outputs "reasoning\n</think>\n\ncontent", so the full captured
			// string is "<think>\nreasoning\n</think>\ncontent".
			// This is Path 2a (both tags present).
			name:            "equivalence/inline_tags_with_open_and_close",
			content:         "<think>\nLet me check the weather\n</think>\nParis is 18°C.",
			messageThinking: "",
			wantReasoning:   "Let me check the weather",
			wantRemaining:   "Paris is 18°C.",
		},
		{
			// Client C: client that captures only the model's generated tokens
			// after the renderer's "<think>\n" prefill, not the prefill itself.
			// The stored content starts with the reasoning text, then
			// "</think>", then the visible answer. There is no <think> open tag
			// because it was part of the prefill, not the model's output.
			// This is Path 2b (close tag without open tag).
			name:            "equivalence/inline_close_tag_only_after_prefill",
			content:         "Let me check the weather\n</think>\nParis is 18°C.",
			messageThinking: "",
			wantReasoning:   "Let me check the weather",
			wantRemaining:   "Paris is 18°C.",
		},

		// Test 2: Multi-line reasoning with internal newlines. The model often
		// produces multi-step reasoning like "Step 1: ...\nStep 2: ...". The
		// \n immediately after <think> and immediately before </think> must be
		// stripped (they are formatting around the tags, not part of the
		// reasoning), but internal \n between reasoning steps must survive.
		// The official template does this via rstrip('\n') and lstrip('\n')
		// which strip only newlines, then |trim which strips all surrounding
		// whitespace — but internal newlines are preserved because trim only
		// operates on the ends. The fork's TrimSpace is equivalent.
		{
			name:            "inline_tags_multiline_reasoning",
			content:         "<think>\nStep 1: look up weather\nStep 2: format answer\n</think>\nParis is 18°C.",
			messageThinking: "",
			wantReasoning:   "Step 1: look up weather\nStep 2: format answer",
			wantRemaining:   "Paris is 18°C.",
		},

		// Test 3: Close tag without open tag — the model's actual raw output
		// format after the renderer's prefill. When thinking is enabled, the
		// renderer writes "<|im_start|>assistant\n<think>\n" before the model
		// generates. The model then produces "reasoning\n</think>\n\nvisible".
		// The <think> tag is part of the prefill, not the model's output. The
		// official template handles this identically: split('</think>')[0]
		// does not require a preceding <think>, and the subsequent
		// split('<think>')[-1] is a no-op when no <think> exists (Python's
		// "text".split('<think>') returns ["text"], and [-1] returns "text").
		{
			name:            "close_tag_only_model_raw_output",
			content:         "Let me check\n</think>\nParis is 18°C.",
			messageThinking: "",
			wantReasoning:   "Let me check",
			wantRemaining:   "Paris is 18°C.",
		},

		// Test 4: Explicit Thinking field takes priority over inline tags.
		// If a client sends both a non-empty Thinking field AND content with
		// <think> tags (a malformed but possible input), the function must NOT
		// double-extract. Path 1 fires at line 65 and returns immediately.
		// The content is returned exactly as received — the inline tags become
		// literal characters in the visible portion of the prompt. The renderer
		// wraps only the explicit field's reasoning in <think>...</think> tags.
		{
			name:            "explicit_field_wins_over_inline_tags",
			content:         "<think>\ninline reasoning\n</think>\nvisible content",
			messageThinking: "explicit reasoning from Thinking field",
			wantReasoning:   "explicit reasoning from Thinking field",
			wantRemaining:   "<think>\ninline reasoning\n</think>\nvisible content",
		},

		// Test 5: No thinking at all — neither explicit field nor inline tags.
		// The function returns empty reasoning. The renderer wraps this as
		// <think>\n\n</think>\n\n (an empty thinking block), which matches the
		// official template's output when reasoning_content is empty after trim.
		{
			name:            "no_thinking_baseline",
			content:         "just content, no thinking",
			messageThinking: "",
			wantReasoning:   "",
			wantRemaining:   "just content, no thinking",
		},

		// Test 6: Whitespace-only Thinking field. The string "   \n  " is not
		// equal to "", so Path 1's guard (messageThinking != "") evaluates to
		// true and Path 1 fires. strings.TrimSpace returns "". The content is
		// NOT parsed for inline tags — Path 2 is never reached. The official
		// template also takes its primary path here ("   \n  " is a string, so
		// `message.reasoning_content is string` is true), and |trim returns "".
		// This documents that whitespace-only fields do not fall through to
		// content parsing.
		{
			name:            "whitespace_only_thinking_field",
			content:         "content that should not be parsed for tags",
			messageThinking: "   \n  ",
			wantReasoning:   "",
			wantRemaining:   "content that should not be parsed for tags",
		},

		// Test 7: Empty content with explicit thinking. The model produced
		// reasoning but no visible text — its response is entirely tool calls.
		// This is a common pattern in agentic use: the model reasons about
		// which tool to call, then the visible response is just the tool call
		// XML (which is rendered separately by the tool call loop at lines
		// 149-166, not by splitQwen35ReasoningContent).
		{
			name:            "empty_content_explicit_thinking_tool_call_only",
			content:         "",
			messageThinking: "Let me call the function",
			wantReasoning:   "Let me call the function",
			wantRemaining:   "",
		},

		// Test 8: Both empty. The assistant message has no thinking and no
		// visible content — the turn consists entirely of tool calls with no
		// reasoning. Path 3: neither messageThinking != "" nor
		// strings.Index("", "</think>") != -1.
		{
			name:            "both_empty_tool_calls_only_no_reasoning",
			content:         "",
			messageThinking: "",
			wantReasoning:   "",
			wantRemaining:   "",
		},

		// Test 9: Multiple </think> tags. The official Jinja2 template uses
		// content.split('</think>')[-1] to extract remaining content, which
		// takes everything after the LAST </think> tag. Grammar constraints
		// prevent the model from generating multiple </think> tags when tools
		// are active, but plain chat without tools has no grammar — the model
		// generates freely and could produce multiple </think> tags. A client
		// that stored such output and sent it back as history must not cause
		// the renderer to include a literal </think> tag in the visible
		// content portion of the prompt — the model was never trained on
		// </think> appearing outside the thinking block wrapper.
		//
		// The fork currently uses strings.Index (first </think>) at line 69,
		// which produces remaining="middle\n</think>\nend" — wrong. The fix
		// is to use strings.LastIndex, matching the official template's [-1]
		// behavior.
		{
			name:            "multiple_close_tags_takes_after_last",
			content:         "<think>\nfirst\n</think>\nmiddle\n</think>\nend",
			messageThinking: "",
			wantReasoning:   "first",
			wantRemaining:   "end",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotReasoning, gotRemaining := splitQwen35ReasoningContent(tt.content, tt.messageThinking)

			if gotReasoning != tt.wantReasoning {
				t.Errorf(
					"reasoning mismatch.\n\n"+
						"The reasoning value is placed inside <think>\\n{reasoning}\\n</think> by the "+
						"renderer at line 144. A wrong value means the model sees different tokens inside "+
						"the thinking block than what it was trained on.\n\n"+
						"got:  %q\nwant: %q", gotReasoning, tt.wantReasoning,
				)
			}

			if gotRemaining != tt.wantRemaining {
				t.Errorf(
					"remaining content mismatch.\n\n"+
						"The remaining value is placed after </think>\\n\\n by the renderer at line 144. "+
						"A wrong value means the model sees different tokens in the visible content "+
						"portion of the assistant message than what it was trained on.\n\n"+
						"got:  %q\nwant: %q", gotRemaining, tt.wantRemaining,
				)
			}
		})
	}

	// Verify the equivalence property explicitly: the first three test cases
	// encode the same model turn three different ways. Assert that all three
	// produce the exact same (reasoning, remaining) tuple. This is a separate
	// assertion from the individual test cases because the property it tests —
	// "the model sees the same prompt regardless of which client sent the
	// history" — is more important than any single extraction path being correct.
	t.Run("equivalence_across_all_three_client_encodings", func(t *testing.T) {
		r1, c1 := splitQwen35ReasoningContent(tests[0].content, tests[0].messageThinking) // explicit field
		r2, c2 := splitQwen35ReasoningContent(tests[1].content, tests[1].messageThinking) // inline both tags
		r3, c3 := splitQwen35ReasoningContent(tests[2].content, tests[2].messageThinking) // inline close only

		if r1 != r2 || r2 != r3 {
			t.Errorf(
				"reasoning differs across client encodings of the same model turn.\n\n"+
					"Three clients stored the same assistant response differently:\n"+
					"  Client A (explicit Thinking field):  reasoning=%q\n"+
					"  Client B (inline <think>...</think>): reasoning=%q\n"+
					"  Client C (inline </think> only):      reasoning=%q\n\n"+
					"All three must produce identical reasoning so the renderer emits the same "+
					"bytes inside <think>...\\n</think> regardless of which client sent the history. "+
					"Different bytes mean different token IDs, which means the model receives a "+
					"prompt it was not trained on and the KV cache is invalidated.",
				r1, r2, r3,
			)
		}

		if c1 != c2 || c2 != c3 {
			t.Errorf(
				"remaining content differs across client encodings of the same model turn.\n\n"+
					"Three clients stored the same assistant response differently:\n"+
					"  Client A (explicit Thinking field):  remaining=%q\n"+
					"  Client B (inline <think>...</think>): remaining=%q\n"+
					"  Client C (inline </think> only):      remaining=%q\n\n"+
					"All three must produce identical remaining content so the renderer emits the "+
					"same bytes after </think>\\n\\n regardless of which client sent the history.",
				c1, c2, c3,
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
