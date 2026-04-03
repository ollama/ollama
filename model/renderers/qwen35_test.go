package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// TestQwen35RendererUsesXMLToolCallingFormat verifies byte-exact rendering of a
// single-tool-call agentic conversation: system message with one tool definition,
// a user query, an assistant response with visible content and one tool call, a
// single tool response, and a follow-up user query.
//
// Ground truth: the expected strings were derived by running the official
// Qwen/Qwen3.5-27B Jinja2 chat template (from tokenizer_config.json on
// HuggingFace, verified identical between Qwen3.5-27B and Qwen3.5-35B-A3B)
// with the same message array and tools. HuggingFace Transformers' tojson
// override (json.dumps with ensure_ascii=False, sort_keys=False, default
// separators) was applied. The Python output was saved as the expected
// strings. Tool definition JSON, message structure, and string arguments
// match Go byte-for-byte. The boolean argument diverges: the template
// produces "True" (Python str(True)) while Go produces "true"
// (fmt.Sprintf) — the test enforces the template's output, so it FAILS
// until formatToolCallArgument is fixed.
//
// Unique coverage vs. other byte-exact tests:
//   - Single tool call (not back-to-back) — only the \n\n content-to-tool
//     separator fires (content|trim truthy → \n\n before first tool call),
//     not the \n inter-tool separator
//   - Single tool response (not grouped) — one <tool_response> block under one
//     <|im_start|>user, closed immediately because the next message is not a
//     tool
//   - No Thinking field on assistant — splitQwen35ReasoningContent receives
//     messageThinking="" and content with no </think> tag, so Path 3
//     (fallthrough) fires: reasoning="" and remaining="I'll check." returned
//     unchanged
//   - String argument "Paris" — formatToolCallArgument hits the case string
//     path, returning the value verbatim; the official template uses
//     args_value|string (Python str()) which produces the same output for
//     strings
//   - Boolean argument true — formatToolCallArgument falls through to
//     fmt.Sprintf("%v", true) producing "true" (lowercase), but the official
//     template uses args_value|string which calls Python str(True) producing
//     "True" (capital T). This is a KNOWN DIVERGENCE (Gap 10) that this test
//     enforces: the expected string uses "True" (the template's ground truth),
//     so the test FAILS until formatToolCallArgument is fixed to match
//   - Tool definition without description — ToolFunction.Description="" with
//     json:"description,omitempty" correctly omits the field; the official
//     template's {{ tool | tojson }} passes through whatever the client sends,
//     so a tool without description produces no "description" key
//
// Think mode: the assistant at index 2 is before lastQueryIndex=4 (the "Thanks"
// user message). The official template's condition is loop.index0 >
// ns.last_query_index, which is 2 > 4 = false — no <think> wrapping
// regardless of enable_thinking. The template checks enable_thinking in exactly
// one place: the add_generation_prompt block. The entire output
// is identical between think=true and think=false except the generation prompt
// suffix.
func TestQwen35RendererUsesXMLToolCallingFormat(t *testing.T) {
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
							{Key: "verbose", Value: true},
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
						{
							Key: "verbose",
							Value: api.ToolProperty{
								Type: api.PropertyType{"boolean"},
							},
						},
					}),
					Required: []string{"location"},
				},
			},
		},
	}

	// HuggingFace Transformers ground truth for this tool definition, produced
	// by json.dumps(tool_dict, ensure_ascii=False) with default separators
	// (', ', ': ') and sort_keys=False. Verified byte-exact against the
	// official Qwen/Qwen3.5-27B chat template's {{ tool | tojson }} output.
	//
	// Key properties of this ground truth:
	//   - No "description" key (field not present in the input dict)
	//   - "properties" before "required" (struct field order in
	//     ToolFunctionParameters matches tojson insertion order; also p < r
	//     alphabetically, so even stock Jinja2 sort_keys=True would agree)
	//   - Two properties: "location" (string) and "verbose" (boolean)
	//   - Only "location" is required (verbose has no default but is not in
	//     required — matching the test's Required: []string{"location"})
	wantToolJSON := `{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "verbose": {"type": "boolean"}}, "required": ["location"]}}}`

	t.Run("think=true", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true}
		got, err := renderer.Render(msgs, tools, nil)
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// Targeted diagnostic: tool definition JSON must match HF ground truth.
		// This fires with a specific error before the byte-exact comparison
		// catches any deviation in the full output.
		gotToolJSON := ""
		for _, line := range strings.Split(got, "\n") {
			if strings.Contains(line, `"get_weather"`) {
				gotToolJSON = line
				break
			}
		}
		if gotToolJSON != wantToolJSON {
			t.Errorf(
				"tool definition JSON does not match HuggingFace ground truth.\n\n"+
					"The model was trained on tool definitions serialized by HuggingFace "+
					"Transformers' tojson filter (json.dumps with ensure_ascii=False, "+
					"sort_keys=False). Any byte-level deviation in the tool JSON changes "+
					"the token IDs in the system prompt, pushing the model out of the "+
					"training distribution.\n\n"+
					"Common causes:\n"+
					"  - HTML-escaped <, >, &: json.Marshal default; marshalWithSpaces in "+
					"model/renderers/json.go must use SetEscapeHTML(false) via jsonutil\n"+
					"  - 'required' before 'properties': ToolFunctionParameters struct "+
					"field order wrong in api/types.go\n"+
					"  - unexpected 'description' key: ToolFunction.Description omitempty "+
					"tag may be missing or the zero value changed\n\n"+
					"want: %s\n got: %s", wantToolJSON, gotToolJSON,
			)
		}

		// Targeted diagnostic: the \n\n separator between assistant content and
		// the first <tool_call>. The official template emits \n\n before the
		// first tool call when content|trim is truthy. Without this
		// separator, content and tool call XML merge on the same line — a token
		// sequence absent from training data.
		if !strings.Contains(got, "I'll check.\n\n<tool_call>") {
			t.Errorf(
				"missing \\n\\n separator between assistant content and first tool call.\n\n"+
					"The official Qwen 3.5 template emits '\\n\\n<tool_call>' when "+
					"content|trim is truthy (content is non-empty after stripping whitespace). "+
					"When content is empty, <tool_call> starts immediately with no separator. "+
					"In Go, this is the j==0 branch: if strings.TrimSpace(content) != \"\" → "+
					"write \\n\\n. Content here is \"I'll check.\" (truthy), so \\n\\n must "+
					"precede <tool_call>.\n\ngot:\n%s", got,
			)
		}

		// Targeted diagnostic: scalar boolean arguments must use Python str()
		// capitalization. The official Qwen 3.5 template uses
		// args_value|string (Jinja2's string filter, which calls Python str())
		// for all non-mapping, non-sequence values. Python str(True) produces
		// "True" (capital T), not "true". Go's formatToolCallArgument uses
		// fmt.Sprintf("%v", true) which produces "true" (lowercase).
		//
		// Fix: add a case bool: branch in formatToolCallArgument in
		// model/renderers/qwen3coder.go that returns "True"/"False".
		if strings.Contains(got, "<parameter=verbose>\ntrue\n</parameter>") {
			t.Errorf(
				"boolean argument rendered as 'true' (Go fmt.Sprintf) instead of "+
					"'True' (Python str(True)).\n\n"+
					"The official Qwen 3.5 template uses args_value|string for scalar tool "+
					"call arguments, which calls Python's str() function. str(True) produces "+
					"'True' (capital T), str(False) produces 'False' (capital F). Go's "+
					"formatToolCallArgument falls through to fmt.Sprintf(\"%%v\", true) which "+
					"produces 'true' (lowercase). The model was trained on 'True', not 'true'.\n\n"+
					"Fix: add case bool: in formatToolCallArgument() in "+
					"model/renderers/qwen3coder.go:\n"+
					"  case bool:\n"+
					"    if v { return \"True\" }\n"+
					"    return \"False\"\n\ngot:\n%s", got,
			)
		}

		// Targeted diagnostic: <|im_end|> must follow </tool_call> on the
		// assistant message. The official template emits <|im_end|>\n
		// unconditionally for every assistant message — it appears outside all
		// conditional blocks.
		if !strings.Contains(got, "</tool_call><|im_end|>") {
			t.Errorf(
				"missing <|im_end|> after </tool_call> on the assistant message.\n\n"+
					"The official Qwen 3.5 template emits <|im_end|>\\n unconditionally for "+
					"every assistant message. Without it, the model sees an unclosed assistant "+
					"turn — a prompt shape absent from all training data.\n\ngot:\n%s", got,
			)
		}

		want := `<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "verbose": {"type": "boolean"}}, "required": ["location"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are a helpful assistant.<|im_end|>
<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
I'll check.

<tool_call>
<function=get_weather>
<parameter=location>
Paris
</parameter>
<parameter=verbose>
True
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
22C
</tool_response><|im_end|>
<|im_start|>user
Thanks<|im_end|>
<|im_start|>assistant
<think>
`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch.\n\n"+
					"The expected string was derived by running the official Qwen/Qwen3.5-27B "+
					"Jinja2 chat template with the same message array and tools, then verified "+
					"byte-for-byte against the Go renderer output. Any deviation changes the "+
					"token IDs the model sees, pushing the input out of the training "+
					"distribution. In multi-turn agentic conversations, deviations in "+
					"historical messages also invalidate the KV cache from the divergence "+
					"point, forcing expensive recomputation of all subsequent tokens.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})

	// think=false: the ONLY difference from think=true is the generation prompt
	// suffix. All historical message rendering — tool definition JSON, assistant
	// content, tool call XML, tool response grouping, <|im_end|> closures — is
	// identical regardless of think mode. The official template checks
	// enable_thinking in exactly one place: the add_generation_prompt block.
	t.Run("think=false", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		got, err := renderer.Render(msgs, tools, &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// Same tool JSON check as think=true. Tool definitions are in the system
		// prompt, rendered before any assistant messages — completely independent
		// of enable_thinking. If this check fails only in think=false, it means
		// think mode is leaking into system prompt rendering, which the official
		// template never does.
		gotToolJSON := ""
		for _, line := range strings.Split(got, "\n") {
			if strings.Contains(line, `"get_weather"`) {
				gotToolJSON = line
				break
			}
		}
		if gotToolJSON != wantToolJSON {
			t.Errorf(
				"tool definition JSON does not match HuggingFace ground truth (think=false).\n\n"+
					"Tool definitions are in the system prompt and must be identical regardless "+
					"of think mode. The official template's {{ tool | tojson }} is outside all "+
					"enable_thinking conditionals. If this fails only in think=false, think mode "+
					"is leaking into system prompt rendering.\n\n"+
					"want: %s\n got: %s", wantToolJSON, gotToolJSON,
			)
		}

		// Same separator check as think=true — content-to-tool-call formatting
		// is independent of think mode.
		if !strings.Contains(got, "I'll check.\n\n<tool_call>") {
			t.Errorf(
				"missing \\n\\n separator between assistant content and first tool call (think=false).\n\n"+
					"Content-to-tool-call formatting is independent of think mode. The \\n\\n "+
					"separator appears because content|trim is truthy (\"I'll check.\"), not "+
					"because of enable_thinking.\n\ngot:\n%s", got,
			)
		}

		// Same boolean capitalization check as think=true — argument formatting
		// is independent of think mode.
		if strings.Contains(got, "<parameter=verbose>\ntrue\n</parameter>") {
			t.Errorf(
				"boolean argument rendered as 'true' instead of 'True' (think=false).\n\n"+
					"Argument formatting is independent of think mode. The official template "+
					"uses args_value|string (Python str(True) → 'True') for scalar booleans. "+
					"Go's formatToolCallArgument uses fmt.Sprintf which produces 'true'.\n\n"+
					"Fix: case bool: in formatToolCallArgument() in "+
					"model/renderers/qwen3coder.go\n\ngot:\n%s", got,
			)
		}

		// Same <|im_end|> check — the official template closes every assistant
		// message unconditionally regardless of think mode.
		if !strings.Contains(got, "</tool_call><|im_end|>") {
			t.Errorf(
				"missing <|im_end|> after </tool_call> in think=false mode.\n\n"+
					"The official Qwen 3.5 template closes every assistant message "+
					"unconditionally, regardless of enable_thinking.\n\ngot:\n%s", got,
			)
		}

		want := `<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "verbose": {"type": "boolean"}}, "required": ["location"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are a helpful assistant.<|im_end|>
<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
I'll check.

<tool_call>
<function=get_weather>
<parameter=location>
Paris
</parameter>
<parameter=verbose>
True
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
22C
</tool_response><|im_end|>
<|im_start|>user
Thanks<|im_end|>
<|im_start|>assistant
<think>

</think>

`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch (think=false).\n\n"+
					"The entire output is identical to think=true EXCEPT the generation "+
					"prompt suffix: <think>\\n\\n</think>\\n\\n (6 tokens matching the "+
					"official template's add_generation_prompt block at lines 149-153 for "+
					"enable_thinking=false) instead of <think>\\n (2 tokens for "+
					"enable_thinking=true/undefined). Historical message rendering is "+
					"unaffected by think mode — the official template checks "+
					"enable_thinking in exactly one place.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})
}

// TestQwen35RendererNoThinkPrefill verifies that when thinking is disabled via
// ThinkValue{Value: false}, the renderer emits the official empty thinking
// block prefill matching the template's add_generation_prompt block (lines
// 149-150: <think>\n\n</think>\n\n for enable_thinking=false).
//
// No tools, no system message, no assistant history — the simplest possible
// prompt shape. The entire output is one user message plus the generation
// prompt.
func TestQwen35RendererNoThinkPrefill(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "hello"},
	}

	// think=true: the generation prompt opens a thinking block for the model
	// to fill. The official template emits <think>\n when enable_thinking is
	// true or undefined.
	t.Run("think=true", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true}
		got, err := renderer.Render(msgs, nil, nil)
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		want := "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch (think=true).\n\n"+
					"The official template's add_generation_prompt block emits "+
					"<think>\\n when enable_thinking is true or undefined. "+
					"No tools → no system prompt. No system message → no system "+
					"turn. One user message → one <|im_start|>user turn.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})

	// think=false: the generation prompt emits a pre-closed empty thinking
	// block so the model skips reasoning. The official template emits
	// <think>\n\n</think>\n\n when enable_thinking is explicitly false.
	t.Run("think=false", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		got, err := renderer.Render(msgs, nil, &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		want := `<|im_start|>user
hello<|im_end|>
<|im_start|>assistant
<think>

</think>

`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch (think=false).\n\n"+
					"The official template's add_generation_prompt block emits "+
					"<think>\\n\\n</think>\\n\\n when enable_thinking is explicitly "+
					"false. The entire output is identical to think=true except the "+
					"generation prompt suffix.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})
}

// TestQwen35RendererBackToBackToolCallsAndResponses verifies byte-exact
// rendering of a multi-turn agentic conversation with parallel tool calls,
// grouped tool responses, and a follow-up user query.
//
// Ground truth: the expected strings were derived by tracing through the
// official Qwen/Qwen3.5-27B Jinja2 chat template (from tokenizer_config.json
// on HuggingFace, verified identical between Qwen3.5-27B and Qwen3.5-35B-A3B)
// with the same message array. Every byte was cross-checked against the
// template source (chat_template.jinja lines 45-154) and the Go renderer
// output (dumped to file and compared with diff).
//
// This test exercises five rendering properties simultaneously:
//
//  1. Pre-lastQueryIndex thinking omission. The assistant message is at index 2.
//     The last non-tool-response user message is "Summarize the results." at
//     index 5, so lastQueryIndex=5. Since 2 <= 5, the official template renders
//     the assistant message WITHOUT <think> wrapping — reasoning_content is
//     computed but discarded. This is a position-based rule, NOT
//     controlled by enable_thinking. The complementary scenario (post-
//     lastQueryIndex, where thinking IS preserved) is tested by
//     TestQwen35RendererInterleavedThinkingAndTools.
//
//  2. Back-to-back tool calls in one assistant turn. The first tool call is
//     preceded by \n\n (content|trim is truthy), subsequent tool calls by \n.
//     Each tool call follows the XML format:
//     <tool_call>\n<function=NAME>\n<parameter=NAME>\nVALUE\n
//     </parameter>\n</function>\n</tool_call>. The </tool_call> of the last
//     tool call is immediately followed by <|im_end|> with no intervening
//     whitespace — the template emits <|im_end|>\n unconditionally.
//
//  3. Grouped tool responses. Consecutive tool messages share a single
//     <|im_start|>user block (template lines 132-142). The <|im_end|> is
//     emitted only after the last tool message in the group.
//
//  4. Tool definition JSON in the system prompt. Two tools (add, multiply) with
//     integer-only parameters. The JSON uses spaced separators (": " and ", ")
//     matching HuggingFace's tojson override (json.dumps with default
//     separators). Field ordering follows Go struct declaration order, which
//     matches get_json_schema() insertion order for simple types.
//
//  5. Generation prompt for both think modes. think=true appends
//     <|im_start|>assistant\n<think>\n. think=false appends
//     <|im_start|>assistant\n<think>\n\n</think>\n\n. The entire output is
//     identical between modes except these trailing bytes — historical message
//     rendering is independent of enable_thinking.
//
// Tool call argument types: the arguments are Go int values (2, 3, 4, 5).
// formatToolCallArgument produces "2", "3", "4", "5" via fmt.Sprintf("%v").
// The official template uses args_value|string for scalar arguments (Python's
// str()), which also produces "2", "3", "4", "5" for Python ints. In real API
// use, JSON-deserialized values arrive as float64, but fmt.Sprintf("%v",
// float64(2)) also produces "2" — identical output for small whole numbers.
// Known divergences for other types (bool True/False, None/null, large
// integers like 1e15) are specified in Gap 10 of test_gap_analysis.md and do
// not affect this test.
func TestQwen35RendererBackToBackToolCallsAndResponses(t *testing.T) {
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

	t.Run("think=true", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true}
		got, err := renderer.Render(msgs, qwen35MathTools(), nil)
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// Targeted diagnostic: thinking content must NOT appear because the
		// assistant message is at index 2, before lastQueryIndex=5. The official
		// template discards reasoning_content for messages at or before
		// last_query_index. This is independent of enable_thinking.
		if strings.Contains(got, "Need to call add and multiply.") {
			t.Errorf(
				"historical thinking content leaked into pre-lastQueryIndex assistant message.\n\n"+
					"The assistant message is at index 2, before lastQueryIndex=5 (the "+
					"'Summarize the results.' user message). The official Qwen 3.5 template "+
					"discards reasoning_content for messages at or before "+
					"last_query_index — this is a position-based rule, NOT controlled by "+
					"enable_thinking. The thinking content should be silently omitted.\n\n"+
					"got:\n%s", got,
			)
		}

		want := `<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "add", "description": "Add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
{"type": "function", "function": {"name": "multiply", "description": "Multiply two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are a helpful assistant.<|im_end|>
<|im_start|>user
Run add and multiply.<|im_end|>
<|im_start|>assistant
I'll run both now.

<tool_call>
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
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
5
</tool_response>
<tool_response>
20
</tool_response><|im_end|>
<|im_start|>user
Summarize the results.<|im_end|>
<|im_start|>assistant
<think>
`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch.\n\n"+
					"The expected string was derived by tracing through the official "+
					"Qwen/Qwen3.5-27B Jinja2 chat template with the same message array. "+
					"Any byte-level deviation changes the token IDs the model sees, pushing "+
					"the input out of the training distribution. In multi-turn agentic "+
					"conversations, deviations also invalidate the KV cache from the "+
					"divergence point, forcing expensive recomputation of all subsequent "+
					"tokens.\n\n--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})

	// think=false: verifies that the ONLY difference from think=true is the
	// generation prompt suffix. Historical message rendering — including the
	// pre-lastQueryIndex thinking omission, tool call XML, tool response
	// grouping, and tool definitions — is identical regardless of think mode.
	// The official template checks enable_thinking in exactly one place: the
	// add_generation_prompt block.
	t.Run("think=false", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		got, err := renderer.Render(msgs, qwen35MathTools(), &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// Same thinking absence check as think=true. The thinking is omitted
		// because of lastQueryIndex position, NOT because of think=false. If
		// someone incorrectly tied thinking omission to the think parameter
		// instead of lastQueryIndex, both subtests would still pass here (the
		// thinking is omitted either way for this message position), but
		// TestQwen35RendererInterleavedThinkingAndTools/think=false would catch
		// the regression for post-lastQueryIndex messages.
		if strings.Contains(got, "Need to call add and multiply.") {
			t.Errorf(
				"historical thinking content leaked into pre-lastQueryIndex assistant message.\n\n"+
					"The assistant message is before lastQueryIndex, so reasoning is discarded "+
					"regardless of think mode. This is a position-based rule in the official "+
					"template, not controlled by enable_thinking.\n\n"+
					"got:\n%s", got,
			)
		}

		want := `<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "add", "description": "Add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
{"type": "function", "function": {"name": "multiply", "description": "Multiply two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are a helpful assistant.<|im_end|>
<|im_start|>user
Run add and multiply.<|im_end|>
<|im_start|>assistant
I'll run both now.

<tool_call>
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
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
5
</tool_response>
<tool_response>
20
</tool_response><|im_end|>
<|im_start|>user
Summarize the results.<|im_end|>
<|im_start|>assistant
<think>

</think>

`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch (think=false).\n\n"+
					"The entire output is identical to think=true EXCEPT the generation "+
					"prompt suffix: <think>\\n\\n</think>\\n\\n (6 tokens matching the "+
					"official template's add_generation_prompt block for "+
					"enable_thinking=false) instead of <think>\\n (2 tokens for "+
					"enable_thinking=true/undefined). Historical message rendering is "+
					"unaffected by think mode — the official template checks "+
					"enable_thinking in exactly one place.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})
}

// TestQwen35RendererStructuredToolArgumentsUseSpacedJSON verifies that when a
// tool call argument is a structured Go type (map or slice), the renderer
// produces spaced JSON separators (": " after colons, ", " after commas) and
// preserves literal HTML characters (<, >, &) without escaping them to
// \u003c, \u003e, \u0026.
//
// This test exercises the formatToolCallArgument → marshalQwenToolCallArgument
// code path in model/renderers/qwen3coder.go, which is shared by both the
// Qwen35Renderer and the Qwen3CoderRenderer. The spaced separators match the
// official Qwen 3.5 Jinja2 template's {{ args_value | tojson | safe }} filter
// for dict/list tool call arguments. Literal HTML characters match HuggingFace
// Transformers' tojson override with ensure_ascii=False.
//
// The byte-exact comparison additionally verifies:
//   - The <think>\n\n</think>\n\n wrapping for a tool-call-only assistant
//     message (empty reasoning + empty content) positioned after lastQueryIndex
//   - The <|im_end|> closing after tool calls (the prefill bug fix)
//   - The generation prompt suffix (think=true vs think=false)
func TestQwen35RendererStructuredToolArgumentsUseSpacedJSON(t *testing.T) {
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

	t.Run("think=true", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true}
		got, err := renderer.Render(msgs, nil, nil)
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// Targeted check: structured JSON arguments must use spaced separators
		// and preserve literal HTML characters, not HTML-escaped Unicode
		// sequences. This fires with a specific diagnostic before the byte-exact
		// comparison below catches any deviation.
		wantArg := `{"content": "if (x < 5 && y > 3) {}"}`
		if !strings.Contains(got, wantArg) {
			t.Errorf(
				"tool call argument not rendered with spaced, non-escaped JSON.\n\n"+
					"The official Qwen 3.5 template uses {{ args_value | tojson | safe }} for "+
					"dict/list arguments, producing spaced JSON (': ' after colons, ', ' "+
					"after commas) with literal <, >, & characters. Go's json.Marshal "+
					"HTML-escapes these by default; marshalQwenToolCallArgument must use "+
					"SetEscapeHTML(false).\n\n"+
					"expected argument substring: %s\ngot:\n%s", wantArg, got,
			)
		}

		want := `<|im_start|>user
call tool<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
<function=echo>
<parameter=payload>
{"content": "if (x < 5 && y > 3) {}"}
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>assistant
<think>
`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch.\n\n"+
					"Any byte-level deviation changes the token IDs the model sees, pushing "+
					"the input out of the training distribution. In multi-turn agentic "+
					"conversations, deviations also invalidate the KV cache from the "+
					"divergence point, forcing expensive recomputation of all subsequent "+
					"tokens.\n\n--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})

	t.Run("think=false", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		got, err := renderer.Render(msgs, nil, &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// The tool call argument formatting is independent of think mode — same
		// spaced JSON with literal HTML characters. The historical assistant
		// message retains its <think> wrapping unconditionally (the official
		// template never checks enable_thinking for history). Only the generation
		// prompt suffix differs: <think>\n\n</think>\n\n (empty block) for
		// think=false instead of <think>\n (open block) for think=true.
		want := `<|im_start|>user
call tool<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
<function=echo>
<parameter=payload>
{"content": "if (x < 5 && y > 3) {}"}
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
					"The historical assistant message must retain its <think> wrapping "+
					"unconditionally — the official Qwen 3.5 template never checks "+
					"enable_thinking when rendering history. The tool call argument "+
					"formatting must be identical regardless of think mode. Only the "+
					"generation prompt suffix should differ.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})
}

// TestQwen35RendererToolDefinitionsMatchOfficialTemplate verifies that tool
// definitions in the system prompt match the training data format produced by
// HuggingFace Transformers' get_json_schema() function, rendered through the
// official Qwen/Qwen3.5-27B tokenizer's apply_chat_template method.
//
// Ground truth: every wantTool* string below was generated by running
// get_json_schema() on a Python function with the specified signature, then
// rendering through apply_chat_template on the official Qwen/Qwen3.5-27B
// tokenizer with HuggingFace Transformers v5.3.0. HF overrides Jinja2's
// tojson filter with json.dumps(ensure_ascii=False) — critically, this does
// NOT pass sort_keys=True, so dict insertion order from get_json_schema() is
// preserved. Stock Jinja2's tojson uses sort_keys=True (alphabetical), which
// produces DIFFERENT key ordering. The model was trained with HF's override,
// so insertion order is the ground truth.
//
// Tool 1 (get_weather) exercises: no HTML escaping, properties before
// required, single-key items, items before description. This tool PASSES —
// Go's struct field order already matches get_json_schema()'s insertion order
// for these fields.
//
// Tool 2 (choose_color) exercises: enum/description field ordering. This tool
// FAILS — Go's ToolProperty struct declares Description (field 4) before Enum
// (field 5), but get_json_schema() inserts enum before description (the
// schema["enum"] assignment precedes schema["description"] in
// chat_template_utils.py). The test enforces the training data ordering; it
// fails until the ToolProperty struct in api/types.go is rewritten to declare
// Enum before Description.
//
// Known open divergences not yet tested (require ToolProperty struct rewrite):
//   - Field loss: nullable, additionalProperties, prefixItems silently
//     dropped during json.Unmarshal. Affects Optional[T], dict[K,V],
//     tuple[T1,T2] parameters.
//   - Multi-key items ordering: complex element types produce multi-key items
//     sub-objects that Go alphabetizes. Fix: Items *ToolProperty instead of
//     any.
//   Both are fixed by the same 9-field ToolProperty struct rewrite that also
//   fixes the enum/description ordering.
// TestQwen35RendererToolDefinitionsMatchOfficialTemplate verifies tool
// definition JSON in the system prompt against byte-exact HuggingFace
// Transformers ground truth, for both think=true and think=false.
//
// Ground truth: HF Transformers v5.3.0 get_json_schema() rendered through
// the official Qwen/Qwen3.5-27B tokenizer's apply_chat_template with HF's
// tojson override (json.dumps with ensure_ascii=False, sort_keys=False).
// The template was run with enable_thinking=True, enable_thinking=False,
// and enable_thinking=undefined — all three produce byte-identical system
// prompts (1566 bytes). enable_thinking is checked in exactly one place
// (inside if add_generation_prompt) and has zero effect on the system prompt,
// which is rendered with no reference to enable_thinking. The think=false
// subtest enforces this invariant.
//
// Tool 1: get_weather(location: str, filters: list[str])
//   Exercises: no HTML escaping, properties before required, single-key
//   items, items before description at property level, literal <, >, &.
//   PASSES.
//
// Tool 2: choose_color(color: Literal["red", "green", "blue"])
//   Exercises: enum/description field ordering. HF produces enum before
//   description. Go's ToolProperty struct produces the opposite. FAILS
//   by design until the ToolProperty struct rewrite (Gap 4).
func TestQwen35RendererToolDefinitionsMatchOfficialTemplate(t *testing.T) {
	// Tool definitions shared across both subtests.
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
								Type:        api.PropertyType{"array"},
								Items:       map[string]any{"type": "string"},
								Description: "Weather filter names",
							},
						},
					}),
					Required: []string{"location", "filters"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "choose_color",
				Description: "Pick a color",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsOrdered([]orderedProp{
						{
							Key: "color",
							Value: api.ToolProperty{
								Type:        api.PropertyType{"string"},
								Enum:        []any{"red", "green", "blue"},
								Description: "The color to use",
							},
						},
					}),
					Required: []string{"color"},
				},
			},
		},
	}

	msgs := []api.Message{{Role: "user", Content: "call tool"}}

	// HF ground truth for both tool JSON lines.
	wantTool1 := `{"type": "function", "function": {"name": "get_weather", "description": "Returns temperature in <fahrenheit> & <celsius>", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name with <tag> & symbol"}, "filters": {"type": "array", "items": {"type": "string"}, "description": "Weather filter names"}}, "required": ["location", "filters"]}}}`

	wantTool2 := `{"type": "function", "function": {"name": "choose_color", "description": "Pick a color", "parameters": {"type": "object", "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"], "description": "The color to use"}}, "required": ["color"]}}}`

	// extractSystemPrompt returns the system prompt portion of the rendered
	// output: everything from <|im_start|>system to (and including) the
	// first <|im_end|>.
	extractSystemPrompt := func(t *testing.T, rendered string) string {
		t.Helper()
		start := strings.Index(rendered, "<|im_start|>system")
		if start == -1 {
			t.Fatalf("no <|im_start|>system found in rendered output.\n\ngot:\n%s", rendered)
		}
		end := strings.Index(rendered[start:], "<|im_end|>")
		if end == -1 {
			t.Fatalf("no <|im_end|> after <|im_start|>system.\n\ngot:\n%s", rendered)
		}
		return rendered[start : start+end+len("<|im_end|>")]
	}

	// assertToolDefinitions runs all tool definition assertions on a
	// rendered output.
	assertToolDefinitions := func(t *testing.T, got string) {
		t.Helper()

		// No HTML escaping.
		for _, esc := range []string{"\\u003c", "\\u003e", "\\u0026"} {
			if strings.Contains(got, esc) {
				t.Errorf(
					"HTML-escaped sequence %q found in tool definition.\n\n"+
						"HuggingFace Transformers overrides Jinja2's tojson with "+
						"json.dumps(ensure_ascii=False). Go's default json.Marshal "+
						"HTML-escapes <, >, &; marshalWithSpaces must use "+
						"SetEscapeHTML(false) via jsonutil.Marshal.\n\ngot:\n%s", esc, got,
				)
			}
		}

		// Extract tool JSON lines.
		var gotTool1, gotTool2 string
		for _, line := range strings.Split(got, "\n") {
			if strings.Contains(line, `"get_weather"`) {
				gotTool1 = line
			}
			if strings.Contains(line, `"choose_color"`) {
				gotTool2 = line
			}
		}

		if gotTool1 != wantTool1 {
			t.Errorf(
				"Tool 1 (get_weather) does not match HuggingFace ground truth.\n\n"+
					"Common causes:\n"+
					"  - 'required' before 'properties': ToolFunctionParameters struct "+
					"field order wrong in api/types.go\n"+
					"  - 'description' before 'items': ToolProperty struct field order "+
					"wrong in api/types.go\n"+
					"  - HTML-escaped <, >, &: need SetEscapeHTML(false)\n\n"+
					"want: %s\n got: %s", wantTool1, gotTool1,
			)
		}

		if gotTool2 != wantTool2 {
			t.Errorf(
				"Tool 2 (choose_color) enum/description field ordering mismatch.\n\n"+
					"HuggingFace Transformers' get_json_schema() inserts 'enum' before "+
					"'description' at the property level (schema[\"enum\"] assignment "+
					"precedes schema[\"description\"] in chat_template_utils.py). HF's "+
					"tojson override uses sort_keys=False, preserving this insertion "+
					"order in the training data. Stock Jinja2 would alphabetize "+
					"(d < e → description first), but the model was trained with HF's "+
					"override, not stock Jinja2.\n\n"+
					"Go's ToolProperty struct in api/types.go declares Description "+
					"(field 4) before Enum (field 5), producing the opposite order.\n\n"+
					"Fix: rewrite the ToolProperty struct in api/types.go to the "+
					"9-field layout: {AnyOf, Type, Items *ToolProperty, "+
					"AdditionalProperties *ToolProperty, PrefixItems []ToolProperty, "+
					"Nullable *bool, Enum, Description, Properties}. This single "+
					"change fixes enum/description ordering, field loss, AND recursive "+
					"nested key ordering.\n\n"+
					"want: %s\n got: %s", wantTool2, gotTool2,
			)
		}
	}

	// Render with think=true (baseline).
	gotThinkTrue, err := (&Qwen35Renderer{isThinking: true}).Render(msgs, tools, nil)
	if err != nil {
		t.Fatalf("render failed (think=true): %v", err)
	}

	// Render with think=false.
	gotThinkFalse, err := (&Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}).Render(msgs, tools, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatalf("render failed (think=false): %v", err)
	}

	// Property 1 enforcement: system prompts must be byte-identical
	// between think modes. The template's system prompt block (lines
	// 45-60) has zero references to enable_thinking. Verified by running
	// the template — all three enable_thinking values produce a
	// byte-identical 1566-byte system prompt.
	sysThinkTrue := extractSystemPrompt(t, gotThinkTrue)
	sysThinkFalse := extractSystemPrompt(t, gotThinkFalse)
	if sysThinkTrue != sysThinkFalse {
		t.Errorf(
			"system prompt differs between think=true and think=false.\n\n"+
				"The official template's system prompt (lines 45-60) has zero "+
				"references to enable_thinking. Tool definition JSON must be "+
				"byte-identical regardless of think mode. If the system prompts "+
				"differ, someone introduced a think-mode dependency into the "+
				"tool/system rendering path.\n\n"+
				"think=true system prompt:\n%s\n\n"+
				"think=false system prompt:\n%s",
			sysThinkTrue, sysThinkFalse,
		)
	}

	t.Run("think=true", func(t *testing.T) {
		assertToolDefinitions(t, gotThinkTrue)
	})

	t.Run("think=false", func(t *testing.T) {
		assertToolDefinitions(t, gotThinkFalse)
	})
}

// TestQwen35RendererInterleavedThinkingAndTools verifies byte-exact rendering
// of two interleaved assistant+tool turns where both assistant messages are
// after lastQueryIndex and have non-empty Thinking fields.
//
// Ground truth: the expected strings were derived by running the official
// Qwen/Qwen3.5-27B Jinja2 chat template (from chat_template.jinja in
// tokenizer_config.json on HuggingFace, verified identical between
// Qwen3.5-27B and Qwen3.5-35B-A3B) with the same message array and tools.
// HuggingFace Transformers' tojson override (json.dumps with
// ensure_ascii=False, sort_keys=False) was applied. Both the Python template
// output and Go renderer output were saved to files and verified
// byte-identical with diff (1981 bytes for think=true, 1992 bytes for
// think=false).
//
// The think=false subtest is the primary regression detector for the fork's
// unconditional thinking block fix (commit 4044b63f). The official template
// checks loop.index0 > ns.last_query_index for <think> wrapping — no
// enable_thinking check. enable_thinking is checked in exactly one place:
// the generation prompt (inside if add_generation_prompt).
//
// The fork's fix has two parts that work together:
//   - splitQwen35ReasoningContent: removed the isThinking parameter so
//     reasoning extraction is unconditional — the upstream version gates on
//     isThinking, silently discarding stored reasoning when think=false.
//   - Rendering condition: removed the isThinking && prefix so <think>
//     wrapping is unconditional for messages after lastQueryIndex.
//
// If either part regresses, the think=false subtest fails:
//   - Re-adding isThinking to splitQwen35ReasoningContent causes the reasoning
//     TEXT to vanish from inside the <think> tags (extraction suppressed).
//   - Re-adding isThinking && to the rendering condition causes the <think>
//     TAGS THEMSELVES to vanish (wrapping suppressed).
//
// The think=true subtest would pass in both regression scenarios because
// isThinking is true, satisfying any re-added gate. This is why the pair
// of subtests is necessary: think=true alone provides zero protection.
//
// This test exercises six rendering properties simultaneously:
//
//  1. Post-lastQueryIndex thinking wrapping. Both assistant messages (indices
//     2 and 4) are after lastQueryIndex=1. The official template wraps them
//     with <think>\n{reasoning}\n</think>\n\n{content} unconditionally when
//     loop.index0 > ns.last_query_index, with no enable_thinking check.
//
//  2. Single tool call per turn. Each assistant turn has one tool call. The
//     first tool call is preceded by \n\n (content|trim is truthy). The
//     </tool_call> is immediately followed by <|im_end|> (unconditional).
//
//  3. Non-consecutive tool responses. Each tool response is separated by an
//     assistant message, so each gets its own <|im_start|>user block
//     (previtem.role != "tool" fires for both).
//
//  4. Tool definition JSON. Two tools (get_weather, get_uv) with simple
//     string parameters. Spaced separators matching HF's tojson override.
//
//  5. System prompt with tools. Tools-first pattern with # Tools header, tool
//     JSON, instructions, then system content appended.
//
//  6. Generation prompt for both think modes. think=true: <think>\n.
//     think=false: <think>\n\n</think>\n\n. The entire output is identical
//     between modes except these trailing bytes.
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
	// Both have non-empty Thinking fields. The official template wraps both
	// in <think> blocks when loop.index0 > ns.last_query_index, with no
	// enable_thinking check.
	//
	// These fragment variables are used by the think=false targeted
	// diagnostics to detect the most critical regression: thinking blocks
	// stripped from history when the client switches think mode.

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

		want := `<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}
{"type": "function", "function": {"name": "get_uv", "description": "Get UV index for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are a helpful assistant.<|im_end|>
<|im_start|>user
Plan a picnic in Paris.<|im_end|>
<|im_start|>assistant
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
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
22C
</tool_response><|im_end|>
<|im_start|>assistant
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
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
5
</tool_response><|im_end|>
<|im_start|>assistant
<think>
`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch.\n\n"+
					"The expected string was derived by running the official "+
					"Qwen/Qwen3.5-27B Jinja2 chat template with the same message "+
					"array and tools, using HuggingFace Transformers' tojson override "+
					"(json.dumps with ensure_ascii=False, sort_keys=False). Both "+
					"outputs were saved to files and verified byte-identical with "+
					"diff (1981 bytes). Any deviation changes the token IDs the "+
					"model sees, pushing the input out of the training distribution. "+
					"In multi-turn agentic conversations, deviations also invalidate "+
					"the KV cache from the divergence point, forcing expensive "+
					"recomputation of all subsequent tokens.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})

	// think=false: The critical regression test for the fork's unconditional
	// thinking block fix. The renderer is constructed with isThinking: true
	// (the struct default), then Render() is called with ThinkValue{Value: false}
	// to override it at runtime — exactly as happens when a client sends
	// think: false on a request after previously using think: true.
	//
	// The historical assistant messages MUST still contain their <think> blocks
	// with the full reasoning text. The official template wraps assistant
	// messages in <think> blocks when loop.index0 > ns.last_query_index — no
	// enable_thinking check. enable_thinking is checked in exactly one place:
	// the generation prompt (inside if add_generation_prompt).
	//
	// Only the generation prompt at the end should differ: think=false produces
	// <think>\n\n</think>\n\n instead of think=true's <think>\n.
	t.Run("think=false", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		got, err := renderer.Render(msgs, qwen35WeatherUVTools(), &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		// Targeted diagnostic: thinking blocks must be preserved in history
		// despite think=false. This fires before the byte-exact comparison
		// and gives a specific diagnosis pointing to the two code locations
		// where the isThinking gate could be re-added.
		if !strings.Contains(got, wantFirstTurn) {
			t.Errorf(
				"historical thinking block stripped from first assistant turn when think=false.\n\n"+
					"The official Qwen 3.5 Jinja2 template wraps assistant "+
					"messages in <think> blocks when loop.index0 > ns.last_query_index. "+
					"This condition has no enable_thinking check — it is purely "+
					"positional. Historical thinking blocks must be preserved "+
					"regardless of the current turn's think mode.\n\n"+
					"If this fails, the isThinking gate was likely re-added to either "+
					"splitQwen35ReasoningContent (suppresses reasoning extraction) or "+
					"the rendering condition (suppresses <think> tag wrapping).\n\n"+
					"got:\n%s", got,
			)
		}

		if !strings.Contains(got, wantSecondTurn) {
			t.Errorf(
				"historical thinking block stripped from second assistant turn when think=false.\n\n"+
					"Same cause as above — both assistant messages are after "+
					"lastQueryIndex and must retain their <think> blocks regardless "+
					"of think mode.\n\n"+
					"got:\n%s", got,
			)
		}

		want := `<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}
{"type": "function", "function": {"name": "get_uv", "description": "Get UV index for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

You are a helpful assistant.<|im_end|>
<|im_start|>user
Plan a picnic in Paris.<|im_end|>
<|im_start|>assistant
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
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
22C
</tool_response><|im_end|>
<|im_start|>assistant
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
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
5
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

`
		if got != want {
			t.Fatalf(
				"byte-exact output mismatch (think=false).\n\n"+
					"The entire output is identical to think=true EXCEPT the "+
					"generation prompt suffix: <think>\\n\\n</think>\\n\\n "+
					"(enable_thinking=false) instead of <think>\\n "+
					"(enable_thinking=true/undefined). "+
					"Historical message rendering — including <think> block wrapping "+
					"for both post-lastQueryIndex assistant messages, tool call XML, "+
					"tool definition JSON, and tool response grouping — is unaffected "+
					"by think mode. The official template checks enable_thinking in "+
					"exactly one place (the add_generation_prompt block).\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, want,
			)
		}
	})
}

// TestQwen35RendererAssistantPrefillWithThinking is the prefill property
// enforcer. It verifies 6 scenarios that together enforce Properties 1, 2,
// and 3 from the test gap analysis.
//
// "Prefill" is an Ollama-specific concept: when the last message is an
// assistant message without tool calls, the renderer leaves the turn open
// (no <|im_end|>, no generation prompt) so the model continues from where
// the text left off. This corresponds to the official template's
// add_generation_prompt=False behavior, minus the trailing <|im_end|>\n
// that the template always emits unconditionally.
//
// Every want string was verified by running the official Qwen/Qwen3.5-27B
// Jinja2 template (chat_template.jinja from tokenizer_config.json on
// HuggingFace) with add_generation_prompt=False, then stripping the
// trailing <|im_end|>\n. The template was run with enable_thinking=True,
// enable_thinking=False, and enable_thinking=undefined for every scenario —
// all three produce identical output, proving that enable_thinking has zero
// effect when add_generation_prompt=False. This is because enable_thinking
// is checked in exactly one place in the template: inside the
// if add_generation_prompt block.
//
// The 6 subtests enforce:
//
//   - Property 1 (think-independence): subtests 1/2 use the SAME messages
//     and share the SAME want constant. Subtests 3/4 do the same. If
//     think mode affects prefill output, the think=false subtest fails
//     while think=true passes — catching re-introduction of the isThinking
//     gate that the fork removed.
//
//   - Property 2 (positional <think> wrapping): subtests 5 and 6 have
//     assistant messages on BOTH sides of the lastQueryIndex boundary.
//     Before the boundary: reasoning is silently discarded, rendered as
//     plain content with no <think> tags. After the boundary: <think>
//     wrapping is unconditional, even if reasoning is empty.
//
//   - Property 3 (prefill suppresses <|im_end|> and generation prompt):
//     every subtest verifies the output does not end with <|im_end|> or
//     a generation prompt. Targeted diagnostics fire before the byte-exact
//     comparison to give specific error messages for these regressions.
func TestQwen35RendererAssistantPrefillWithThinking(t *testing.T) {
	// Ground truth for subtests 1 and 2: assistant with reasoning.
	// Template output (add_generation_prompt=False, any enable_thinking value)
	// minus trailing <|im_end|>\n.
	//
	// The assistant at index 1 is after lastQueryIndex=0 (the user at index 0),
	// so the template wraps with <think>\n{reasoning}\n</think>\n\n{content}
	// splitQwen35ReasoningContent extracts reasoning="Keep it short."
	// via Path 1 (explicit Thinking field). Content is unchanged.
	const wantWithReasoning = "<|im_start|>user\n" +
		"Write two words.<|im_end|>\n" +
		"<|im_start|>assistant\n" +
		"<think>\n" +
		"Keep it short.\n" +
		"</think>\n" +
		"\n" +
		"Hello world"

	// Ground truth for subtests 3 and 4: assistant without reasoning.
	// The assistant at index 1 is after lastQueryIndex=0, so the template
	// still wraps with <think>, but reasoning is empty — producing
	// <think>\n\n</think>\n\n (empty thinking block). This is correct per
	// the template: the wrapping condition is purely positional,
	// not gated on whether reasoning_content is non-empty.
	const wantNoReasoning = "<|im_start|>user\n" +
		"Write two words.<|im_end|>\n" +
		"<|im_start|>assistant\n" +
		"<think>\n" +
		"\n" +
		"</think>\n" +
		"\n" +
		"Hello world"

	// Ground truth for subtest 5: multi-turn agentic loop with tool history.
	// The first assistant (index 1) is BEFORE lastQueryIndex=3 (the "Now
	// summarize." user message at index 3). The template renders it as plain
	// content — reasoning "I should use the weather tool." is computed by
	// splitQwen35ReasoningContent but silently discarded. The last assistant
	// (index 4) is AFTER lastQueryIndex=3, so it gets <think> wrapping with
	// reasoning preserved.
	//
	// Tool call arguments exercise all formatToolCallArgument scalar paths:
	//   "Paris" (string) → verbatim passthrough
	//   true (bool) → "True" (Python str(True))
	//   float64(42) (integer-valued float) → "42" (strconv.FormatInt)
	//   nil → "None" (Python str(None))
	// Argument ordering matches dict insertion order in the template.
	const wantToolHistory = "<|im_start|>user\n" +
		"What's the weather?<|im_end|>\n" +
		"<|im_start|>assistant\n" +
		"Let me check.\n" +
		"\n" +
		"<tool_call>\n" +
		"<function=get_weather>\n" +
		"<parameter=location>\n" +
		"Paris\n" +
		"</parameter>\n" +
		"<parameter=verbose>\n" +
		"True\n" +
		"</parameter>\n" +
		"<parameter=count>\n" +
		"42\n" +
		"</parameter>\n" +
		"<parameter=empty>\n" +
		"None\n" +
		"</parameter>\n" +
		"</function>\n" +
		"</tool_call><|im_end|>\n" +
		"<|im_start|>user\n" +
		"<tool_response>\n" +
		"22C\n" +
		"</tool_response><|im_end|>\n" +
		"<|im_start|>user\n" +
		"Now summarize.<|im_end|>\n" +
		"<|im_start|>assistant\n" +
		"<think>\n" +
		"Simple answer from tool result.\n" +
		"</think>\n" +
		"\n" +
		"It's 22C in Paris."

	// Ground truth for subtest 6: both sides of the lastQueryIndex boundary.
	// First assistant (index 1, before lastQueryIndex=2): reasoning "Think
	// first." is discarded, rendered as plain "First answer." with no <think>.
	// Second assistant (index 3, after lastQueryIndex=2): no reasoning field,
	// rendered with empty <think> block.
	const wantBoundary = "<|im_start|>user\n" +
		"First question.<|im_end|>\n" +
		"<|im_start|>assistant\n" +
		"First answer.<|im_end|>\n" +
		"<|im_start|>user\n" +
		"Second question.<|im_end|>\n" +
		"<|im_start|>assistant\n" +
		"<think>\n" +
		"\n" +
		"</think>\n" +
		"\n" +
		"Second answer."

	// prefillDiagnostics runs targeted assertions that catch specific
	// regressions with clear error messages before the byte-exact
	// comparison. These use t.Errorf (not t.Fatalf) so all diagnostics
	// fire even if multiple things are wrong.
	prefillDiagnostics := func(t *testing.T, got string) {
		t.Helper()

		if strings.HasSuffix(got, "<|im_end|>") || strings.HasSuffix(got, "<|im_end|>\n") {
			t.Errorf(
				"prefill output ends with <|im_end|>.\n\n"+
					"Prefill must leave the assistant turn open for continuation. "+
					"The official template emits <|im_end|> unconditionally, "+
					"but Ollama's prefill deliberately omits it when the last message is "+
					"an assistant without tool calls. If <|im_end|> appears at the end, "+
					"the prefill condition is broken — check that "+
					"len(message.ToolCalls) == 0 is still in the guard and that "+
					"lastMessage is computed correctly.\n\ngot tail: %q",
				got[max(0, len(got)-60):],
			)
		}

		if strings.HasSuffix(got, "<think>\n") {
			t.Errorf(
				"prefill output ends with a think=true generation prompt.\n\n"+
					"Prefill must not append any generation prompt. The generation "+
					"prompt is gated on lastMessage && !prefill. "+
					"If a generation prompt appears, prefill is false when it should "+
					"be true.\n\ngot tail: %q",
				got[max(0, len(got)-60):],
			)
		}

		if strings.HasSuffix(got, "</think>\n\n") && strings.Contains(got[max(0, len(got)-40):], "<|im_start|>assistant") {
			t.Errorf(
				"prefill output ends with a think=false generation prompt.\n\n"+
					"Prefill must not append any generation prompt, regardless of "+
					"think mode. The generation prompt is gated on "+
					"lastMessage && !prefill. If a generation "+
					"prompt appears, prefill is false when it should be true.\n\n"+
					"got tail: %q",
				got[max(0, len(got)-80):],
			)
		}
	}

	// --- Subtest 1: think=true, assistant with reasoning ---
	// Baseline prefill scenario. The existing test, restructured.
	t.Run("think=true", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true}
		msgs := []api.Message{
			{Role: "user", Content: "Write two words."},
			{Role: "assistant", Thinking: "Keep it short.", Content: "Hello world"},
		}

		got, err := renderer.Render(msgs, nil, nil)
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		prefillDiagnostics(t, got)

		if got != wantWithReasoning {
			t.Fatalf(
				"byte-exact output mismatch.\n\n"+
					"Ground truth: official Qwen 3.5 template with "+
					"add_generation_prompt=False, trailing <|im_end|>\\n stripped.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, wantWithReasoning,
			)
		}
	})

	// --- Subtest 2: think=false, same messages, must be identical ---
	// Property 1 regression detector: if someone re-adds an isThinking gate
	// to splitQwen35ReasoningContent or to the <think> wrapping condition,
	// this subtest fails while subtest 1 passes. Uses the SAME want constant
	// as subtest 1 — a compile-time guarantee that the test specification
	// itself does not encode think-mode-dependent behavior.
	t.Run("think=false_preserves_reasoning", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		msgs := []api.Message{
			{Role: "user", Content: "Write two words."},
			{Role: "assistant", Thinking: "Keep it short.", Content: "Hello world"},
		}

		got, err := renderer.Render(msgs, nil, &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		prefillDiagnostics(t, got)

		// Targeted diagnostic: reasoning must survive think=false.
		if !strings.Contains(got, "Keep it short.") {
			t.Errorf(
				"reasoning text missing from think=false prefill output.\n\n"+
					"The official template extracts reasoning_content and wraps it in "+
					"<think> tags for messages after last_query_index. "+
					"This wrapping has no enable_thinking check — it is purely "+
					"positional. If the reasoning vanishes when think=false, the "+
					"isThinking gate has been re-introduced to either "+
					"splitQwen35ReasoningContent or the <think> wrapping condition.\n\n"+
					"got:\n%s", got,
			)
		}

		if got != wantWithReasoning {
			t.Fatalf(
				"byte-exact output mismatch (think=false).\n\n"+
					"The prefill output MUST be identical to think=true. The official "+
					"template was run with enable_thinking=True, False, and undefined "+
					"for add_generation_prompt=False — all three produce identical "+
					"output. enable_thinking is checked in exactly one place ("+
					"inside the add_generation_prompt block), which does not run for "+
					"prefill.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, wantWithReasoning,
			)
		}
	})

	// --- Subtest 3: think=true, no reasoning ---
	// Verifies that post-lastQueryIndex wrapping produces an empty <think>
	// block when there is no reasoning, not plain content without <think>.
	// The template's wrapping condition is loop.index0 >
	// ns.last_query_index — it does not check whether reasoning_content is
	// non-empty. An empty reasoning_content produces <think>\n\n</think>\n\n.
	t.Run("no_reasoning_think=true", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true}
		msgs := []api.Message{
			{Role: "user", Content: "Write two words."},
			{Role: "assistant", Content: "Hello world"},
		}

		got, err := renderer.Render(msgs, nil, nil)
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		prefillDiagnostics(t, got)

		// Targeted diagnostic: <think> wrapping must be present even with
		// no reasoning. The wrapping is positional, not content-dependent.
		if !strings.Contains(got, "<think>") {
			t.Errorf(
				"<think> tag missing from post-lastQueryIndex assistant.\n\n"+
					"The official template wraps every assistant message after "+
					"last_query_index with <think> tags, regardless "+
					"of whether reasoning_content is empty. An empty reasoning "+
					"produces <think>\\n\\n</think>\\n\\n (empty block). If <think> "+
					"is missing, the wrapping condition may have been changed to "+
					"check for non-empty reasoning.\n\ngot:\n%s", got,
			)
		}

		if got != wantNoReasoning {
			t.Fatalf(
				"byte-exact output mismatch.\n\n"+
					"Ground truth: official Qwen 3.5 template with "+
					"add_generation_prompt=False, trailing <|im_end|>\\n stripped. "+
					"Empty reasoning → <think>\\n\\n</think>\\n\\n wrapping.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, wantNoReasoning,
			)
		}
	})

	// --- Subtest 4: think=false, no reasoning, must be identical ---
	// Property 1 for the no-reasoning path. Uses the SAME want constant
	// as subtest 3.
	t.Run("no_reasoning_think=false", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		msgs := []api.Message{
			{Role: "user", Content: "Write two words."},
			{Role: "assistant", Content: "Hello world"},
		}

		got, err := renderer.Render(msgs, nil, &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		prefillDiagnostics(t, got)

		if got != wantNoReasoning {
			t.Fatalf(
				"byte-exact output mismatch (think=false, no reasoning).\n\n"+
					"The prefill output MUST be identical to think=true. "+
					"enable_thinking has zero effect on historical message rendering.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, wantNoReasoning,
			)
		}
	})

	// --- Subtest 5: think mode switch with full agentic tool history ---
	// The most realistic prefill scenario: a multi-turn tool-calling
	// conversation where the client replays history (originally generated
	// under think=true) and now requests think=false. The last assistant
	// message is a prefill.
	//
	// This subtest enforces Properties 1, 2, 3, 5, and 7 simultaneously:
	//   - P1: think=false, historical reasoning preserved in last assistant
	//   - P2: first assistant (index 1) BEFORE lastQueryIndex=3 → reasoning
	//     "I should use the weather tool." discarded, plain rendering;
	//     last assistant (index 4) AFTER lastQueryIndex=3 → <think> wrapping
	//   - P3: last assistant is prefill — no <|im_end|>, no generation prompt
	//   - P5: tool call arguments exercise all formatToolCallArgument scalar
	//     paths — string ("Paris"), bool (True), int (42), nil (None)
	//   - P7: single tool response with own <|im_start|>user block
	t.Run("think_switch_with_tool_history", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true}
		msgs := []api.Message{
			{Role: "user", Content: "What's the weather?"},
			{
				Role:     "assistant",
				Content:  "Let me check.",
				Thinking: "I should use the weather tool.",
				ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgsOrdered([]orderedArg{
							{Key: "location", Value: "Paris"},
							{Key: "verbose", Value: true},
							{Key: "count", Value: float64(42)},
							{Key: "empty", Value: nil},
						}),
					}},
				},
			},
			{Role: "tool", Content: "22C"},
			{Role: "user", Content: "Now summarize."},
			{
				Role:     "assistant",
				Content:  "It's 22C in Paris.",
				Thinking: "Simple answer from tool result.",
			},
		}

		got, err := renderer.Render(msgs, nil, &api.ThinkValue{Value: false})
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		prefillDiagnostics(t, got)

		// Targeted diagnostic: pre-lastQueryIndex assistant must NOT have
		// <think> wrapping. The first assistant's reasoning must be absent
		// from the output entirely — not just not wrapped, but not present.
		firstAssistantEnd := strings.Index(got, "</tool_call><|im_end|>")
		if firstAssistantEnd == -1 {
			t.Fatalf("cannot find </tool_call><|im_end|> in output — first assistant turn structure is wrong.\n\ngot:\n%s", got)
		}
		firstAssistantBlock := got[:firstAssistantEnd]

		if strings.Contains(firstAssistantBlock, "<think>") {
			t.Errorf(
				"first assistant (before lastQueryIndex) has <think> wrapping.\n\n"+
					"The official template wraps with <think> only when "+
					"loop.index0 > ns.last_query_index. The first assistant is at "+
					"index 1, lastQueryIndex=3 (the 'Now summarize.' user message). "+
					"1 > 3 is false, so no <think> wrapping. If <think> appears, "+
					"the lastQueryIndex computation or the wrapping condition is wrong.\n\n"+
					"first assistant block:\n%s", firstAssistantBlock,
			)
		}

		if strings.Contains(firstAssistantBlock, "I should use the weather tool") {
			t.Errorf(
				"first assistant's reasoning leaked into pre-lastQueryIndex output.\n\n"+
					"The official template (lines 102-103) discards reasoning_content "+
					"for messages at or before last_query_index. The rendered output "+
					"must be <|im_start|>assistant\\nLet me check. with zero trace of "+
					"the Thinking field. If reasoning appears, "+
					"splitQwen35ReasoningContent's output is being used in the wrong "+
					"branch.\n\nfirst assistant block:\n%s", firstAssistantBlock,
			)
		}

		// Targeted diagnostic: post-lastQueryIndex assistant must have
		// reasoning preserved despite think=false.
		lastAssistantStart := strings.LastIndex(got, "<|im_start|>assistant")
		if lastAssistantStart == -1 {
			t.Fatalf("cannot find last <|im_start|>assistant in output.\n\ngot:\n%s", got)
		}
		lastAssistantBlock := got[lastAssistantStart:]

		if !strings.Contains(lastAssistantBlock, "<think>\nSimple answer from tool result.\n</think>") {
			t.Errorf(
				"last assistant's reasoning missing from post-lastQueryIndex output "+
					"under think=false.\n\n"+
					"The official template wraps reasoning in <think> tags for all "+
					"messages after last_query_index, with no "+
					"enable_thinking check. If reasoning vanishes when think=false, "+
					"the isThinking gate has been re-introduced.\n\n"+
					"last assistant block:\n%s", lastAssistantBlock,
			)
		}

		if got != wantToolHistory {
			t.Fatalf(
				"byte-exact output mismatch (think=false, tool history).\n\n"+
					"Ground truth: official Qwen 3.5 template with "+
					"add_generation_prompt=False, trailing <|im_end|>\\n stripped. "+
					"Verified with enable_thinking=True, False, and undefined — all "+
					"identical.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, wantToolHistory,
			)
		}
	})

	// --- Subtest 6: lastQueryIndex boundary — both sides in one conversation ---
	// Two assistant messages: one before lastQueryIndex (reasoning discarded),
	// one after (empty <think> wrapping). This is the only test that verifies
	// both sides of the boundary byte-exact in a single render call. A
	// refactoring that changes the boundary computation while leaving each
	// side independently correct would be caught here.
	t.Run("pre_lastQueryIndex_reasoning_discarded", func(t *testing.T) {
		renderer := &Qwen35Renderer{isThinking: true}
		msgs := []api.Message{
			{Role: "user", Content: "First question."},
			{Role: "assistant", Thinking: "Think first.", Content: "First answer."},
			{Role: "user", Content: "Second question."},
			{Role: "assistant", Content: "Second answer."},
		}

		got, err := renderer.Render(msgs, nil, nil)
		if err != nil {
			t.Fatalf("render failed: %v", err)
		}

		prefillDiagnostics(t, got)

		// Targeted diagnostic: first assistant must NOT have <think> or reasoning.
		firstEnd := strings.Index(got, "<|im_end|>\n<|im_start|>user\nSecond")
		if firstEnd == -1 {
			t.Fatalf("cannot find boundary between first assistant and second user message.\n\ngot:\n%s", got)
		}
		firstBlock := got[:firstEnd]

		if strings.Contains(firstBlock, "<think>") {
			t.Errorf(
				"first assistant (before lastQueryIndex) has <think> wrapping.\n\n"+
					"lastQueryIndex=2 (the 'Second question.' user). First assistant "+
					"is at index 1. 1 > 2 is false → no <think> wrapping.\n\n"+
					"first block:\n%s", firstBlock,
			)
		}

		if strings.Contains(firstBlock, "Think first") {
			t.Errorf(
				"first assistant's reasoning 'Think first.' leaked into output.\n\n"+
					"Messages at or before lastQueryIndex discard reasoning. The "+
					"template (lines 102-103) uses only content in the else branch.\n\n"+
					"first block:\n%s", firstBlock,
			)
		}

		// Targeted diagnostic: second assistant (after lastQueryIndex) must have
		// empty <think> wrapping.
		if !strings.Contains(got, "<think>\n\n</think>\n\nSecond answer.") {
			t.Errorf(
				"second assistant (after lastQueryIndex) missing empty <think> wrapping.\n\n"+
					"The template wraps every post-lastQueryIndex assistant with <think> "+
					"even when reasoning is empty. Empty reasoning → "+
					"<think>\\n\\n</think>\\n\\n.\n\ngot:\n%s", got,
			)
		}

		if got != wantBoundary {
			t.Fatalf(
				"byte-exact output mismatch (lastQueryIndex boundary).\n\n"+
					"Ground truth: official Qwen 3.5 template with "+
					"add_generation_prompt=False, trailing <|im_end|>\\n stripped.\n\n"+
					"--- got ---\n%s\n--- want ---\n%s", got, wantBoundary,
			)
		}
	})
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
		// double-extract. Path 1 fires (messageThinking != "") and returns immediately.
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
		{
			name:            "multiple_close_tags_takes_after_last",
			content:         "<think>\nfirst\n</think>\nmiddle\n</think>\nend",
			messageThinking: "",
			wantReasoning:   "first",
			wantRemaining:   "end",
		},

		// Test 10: Nested <think> tags. The content has multiple <think>
		// open tags before the first </think>. The official template's
		// split('<think>')[-1] takes text after the LAST <think>, discarding
		// text between earlier <think> tags. The fork's
		// strings.LastIndex(before, "<think>") achieves the same: it finds
		// the last <think> in the text before the first </think>, not the
		// first <think>. A bug that used strings.Index instead of
		// strings.LastIndex for the <think> open tag would return
		// "outer<think>inner" instead of "inner".
		//
		// The text between the two </think> tags ("more") is silently
		// dropped by both the template (split('</think>') loses middle
		// segments) and the fork (firstClose != lastClose, so the gap is
		// skipped). This is correct: the model never produces nested tags,
		// so this is defensive handling of malformed third-party input.
		{
			name:            "nested_think_tags_lastindex_finds_innermost",
			content:         "<think>outer<think>inner</think>more</think>content",
			messageThinking: "",
			wantReasoning:   "inner",
			wantRemaining:   "content",
		},

		// Test 11: No <think> open tag with multiple </think> close tags.
		// This combines the else branch (open == -1, reasoning = before)
		// with lastClose != firstClose (multiple </think> tags).
		// Test 3 covers close-only with a single </think>. Test 9 covers
		// multiple </think> with a <think> present. This test covers the
		// combination: no <think> AND multiple </think>.
		//
		// The official template's split('<think>')[-1] on a string without
		// <think> returns the whole string (Python's "text".split('<think>')
		// == ["text"]), so reasoning is the full text before the first
		// </think>. The fork's else branch does the same.
		{
			name:            "no_open_tag_multiple_close_tags",
			content:         "inner</think>more</think>content",
			messageThinking: "",
			wantReasoning:   "inner",
			wantRemaining:   "content",
		},

		// Test 12: Empty remaining content from Path 2 extraction. The
		// </think> tag is at the very end of content with nothing after it.
		// All prior Path 2 tests produce non-empty remaining. This verifies
		// that TrimLeft("\n") on an empty string returns "" correctly, and
		// that the renderer can handle a model turn where the assistant
		// reasoned but produced no visible text (the response is entirely
		// tool calls rendered separately by the tool call loop).
		{
			name:            "close_tag_at_end_empty_remaining",
			content:         "<think>reasoning</think>",
			messageThinking: "",
			wantReasoning:   "reasoning",
			wantRemaining:   "",
		},

		// Test 13: <think> open tag present but NO </think> close tag.
		// Test 5 has no tags at all and reaches Path 3. This test has a
		// <think> tag but no matching </think>, which must ALSO reach
		// Path 3: the entry condition checks for </think> via
		// strings.Index, not <think>. If someone changed the condition to
		// check for <think> instead, this test would catch it. The content
		// — including the literal <think> — passes through unchanged.
		{
			name:            "open_tag_only_no_close_tag_path3",
			content:         "content<think>reasoning",
			messageThinking: "",
			wantReasoning:   "",
			wantRemaining:   "content<think>reasoning",
		},

		// Test 14: </think> at the very start of content with no preceding
		// text. This reaches Path 2 with empty reasoning via extraction —
		// NOT Path 3. The difference matters: before = "" (the text before
		// the first </think>), open = LastIndex("", "<think>") = -1, so the
		// else branch sets reasoning = before = "". After TrimSpace, reasoning
		// is "". Tests 5 and 8 also produce empty reasoning, but they reach
		// it via Path 3 (no </think> in content). This test verifies that
		// Path 2's else branch also produces empty reasoning correctly.
		{
			name:            "close_tag_at_start_empty_reasoning_via_path2",
			content:         "</think>content",
			messageThinking: "",
			wantReasoning:   "",
			wantRemaining:   "content",
		},

		// Test 15: Space immediately after </think> — remaining content must
		// preserve leading spaces. The official template uses .lstrip('\n')
		// (Python) which strips ONLY newline characters, not spaces or tabs.
		// The fork uses strings.TrimLeft(content, "\n") which is equivalent.
		// If someone changes this to strings.TrimSpace or TrimLeft("\n "),
		// the leading space would be incorrectly stripped.
		//
		// All prior Path 2 tests have only \n characters between </think> and
		// the remaining content (or nothing). This is the sole test that has
		// non-newline whitespace after </think>, enforcing that the function
		// strips newlines but preserves spaces — exactly as the model was
		// trained. The model's output has \n\n between </think> and content,
		// but third-party clients may store content with other whitespace
		// patterns, and the function must handle them correctly.
		{
			name:            "space_after_close_tag_preserved_in_remaining",
			content:         "<think>reasoning</think> content with leading space",
			messageThinking: "",
			wantReasoning:   "reasoning",
			wantRemaining:   " content with leading space",
		},

		// Test 16: Newline then space after </think> — the newline is
		// stripped but the space is preserved. This enforces the exact
		// boundary between "strip leading newlines" and "preserve everything
		// else" in the remaining content. The official template's
		// .lstrip('\n') on "\n content" produces " content". The fork's
		// TrimLeft("\n") on "\n content" produces " content".
		{
			name:            "newline_then_space_after_close_tag",
			content:         "<think>reasoning</think>\n content",
			messageThinking: "",
			wantReasoning:   "reasoning",
			wantRemaining:   " content",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotReasoning, gotRemaining := splitQwen35ReasoningContent(tt.content, tt.messageThinking)

			if gotReasoning != tt.wantReasoning {
				t.Errorf(
					"reasoning mismatch.\n\n"+
						"The reasoning value is placed inside <think>\\n{reasoning}\\n</think> by the "+
						"renderer's post-lastQueryIndex wrapping. A wrong value means the model sees "+
						"different tokens inside the thinking block than what it was trained on.\n\n"+
						"got:  %q\nwant: %q", gotReasoning, tt.wantReasoning,
				)
			}

			if gotRemaining != tt.wantRemaining {
				t.Errorf(
					"remaining content mismatch.\n\n"+
						"The remaining value is placed after </think>\\n\\n by the renderer's "+
						"post-lastQueryIndex wrapping. A wrong value means the model sees different "+
						"tokens in the visible content portion of the assistant message than what "+
						"it was trained on.\n\n"+
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
