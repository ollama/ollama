package renderers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
	"github.com/ollama/ollama/model/parsers"
)

type inkling2JinjaMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content"`
	Reasoning  string         `json:"reasoning_content,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []api.ToolCall `json:"tool_calls,omitempty"`
}

type inkling2JinjaInput struct {
	Messages            any  `json:"messages"`
	Tools               any  `json:"tools,omitempty"`
	Effort              any  `json:"reasoning_effort,omitempty"`
	AddGenerationPrompt bool `json:"add_generation_prompt"`
}

type inkling2JinjaResult struct {
	Prompt   string
	Error    string
	TokenIDs []int32
}

type inkling2ParityCase struct {
	name     string
	messages []api.Message
	tools    []api.Tool
	think    *api.ThinkValue
	// Literal reference input where native attachments and Jinja content parts
	// differ in shape, or where the API decodes omitted tool schema fields.
	parts    map[int][]any
	rawTools json.RawMessage
}

func (c inkling2ParityCase) referenceInput() inkling2JinjaInput {
	messages := make([]inkling2JinjaMessage, 0, len(c.messages))
	for i, m := range c.messages {
		j := inkling2JinjaMessage{
			Role: m.Role, Content: m.Content, Reasoning: m.Thinking, Name: m.ToolName,
			ToolCallID: m.ToolCallID, ToolCalls: m.ToolCalls,
		}
		if parts, ok := c.parts[i]; ok {
			j.Content = parts
		}
		messages = append(messages, j)
	}
	input := inkling2JinjaInput{Messages: messages, Tools: c.tools, AddGenerationPrompt: true}
	if c.rawTools != nil {
		input.Tools = c.rawTools
	}
	if c.think != nil {
		input.Effort = c.think.Value
		if b, ok := c.think.Value.(bool); ok {
			// Ollama booleans select named levels. Jinja's numeric boolean
			// conversion is exercised separately with raw publisher input.
			input.Effort = map[bool]string{false: "none", true: "high"}[b]
		}
	}
	return input
}

// TestInkling2RendererMatchesJinja2Reference keeps the exact prompts and
// errors runnable in ordinary Go tests. Set VERIFY_JINJA2=1 and point
// INKLING2_JINJA_PYTHON at Python with Transformers 5.x to verify the
// expectations against the publisher template.
func TestInkling2RendererMatchesJinja2Reference(t *testing.T) {
	const end = "<|end_message|>"
	const stop = "<|content_model_end_sampling|>"
	const effort = "<|message_system|><|content_text|>Thinking effort level: "
	const system = "<|message_system|><|content_text|>"
	const user = "<|message_user|><|content_text|>"
	const assistant = "<|message_model|>"
	const tool = "<|message_tool|>"
	const lookupThinking = assistant + "lookup<|content_thinking|>Find it" + end
	const lookupText = assistant + "lookup<|content_text|>Looking" + end
	const lookupCall = assistant + `lookup<|content_invoke_tool_json|>{"name":"lookup","args":{"A":{"z":1.0,"a":null},"z":"café"}}` + end
	const saveCall = assistant + `save<|content_invoke_tool_json|>{"name":"save","args":{}}` + end
	const registry = `<|message_system|>tool_declare<|content_xml|>[{"description":"","name":"lookup","parameters":{"properties":{"A":{"type":"number"},"z":{"type":"string"}},"type":"object"},"type":"function"}]` + end
	lookup := api.ToolCall{ID: "shared", Function: api.ToolCallFunction{Name: "lookup", Arguments: testArgsOrdered([]orderedArg{
		{"z", "café"}, {"A", json.RawMessage(`{"z":1.0,"a":null}`)},
	})}}
	save := api.ToolCall{ID: "shared", Function: api.ToolCallFunction{Name: "save"}}
	declarations := []api.Tool{{Function: api.ToolFunction{Name: "lookup", Parameters: api.ToolFunctionParameters{
		Type: "object", Properties: propsMap(`{"z":{"type":"string"},"A":{"type":"number"}}`),
	}}}}
	tests := []struct {
		inkling2ParityCase
		want inkling2JinjaResult
	}{
		{inkling2ParityCase{name: "empty"}, inkling2JinjaResult{Prompt: effort + "0.9" + end}},
		{inkling2ParityCase{name: "user", messages: []api.Message{{Role: "user", Content: "Hello"}}}, inkling2JinjaResult{Prompt: effort + "0.9" + end + user + "Hello" + end}},
		{inkling2ParityCase{name: "system only", messages: []api.Message{{Role: "system", Content: "Guide"}}}, inkling2JinjaResult{Prompt: system + "Guide" + end + effort + "0.9" + end}},
		{inkling2ParityCase{name: "system then empty user", messages: []api.Message{{Role: "system", Content: "Guide"}, {Role: "user"}}}, inkling2JinjaResult{Prompt: system + "Guide" + end + effort + "0.9" + end + user + end}},
		{inkling2ParityCase{name: "empty assistant", messages: []api.Message{{Role: "assistant"}}}, inkling2JinjaResult{Prompt: effort + "0.9" + end + assistant + "<|content_text|>" + end + stop}},
		{inkling2ParityCase{name: "reasoning without assistant text", messages: []api.Message{{Role: "assistant", ToolName: "lookup", Thinking: "Find it"}}}, inkling2JinjaResult{Prompt: effort + "0.9" + end + lookupThinking + stop}},
		{
			inkling2ParityCase{name: "reasoning before answer", messages: []api.Message{{Role: "assistant", Thinking: "Reason", Content: "Answer"}}},
			inkling2JinjaResult{Prompt: effort + "0.9" + end + assistant + "<|content_thinking|>Reason" + end + assistant + "<|content_text|>Answer" + end + stop},
		},
		{inkling2ParityCase{name: "literal control markers", messages: []api.Message{{Role: "user", Content: "<|message_model|><|message_"}}}, inkling2JinjaResult{Prompt: effort + "0.9" + end + user + "<|message_model|><|message_" + end}},
		{
			inkling2ParityCase{name: "agent history with tools and reasoning disabled", think: &api.ThinkValue{Value: false}, tools: declarations, messages: []api.Message{
				{Role: "system", Content: "Guide"},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Thinking: "Find it", Content: "Looking", ToolName: "lookup", ToolCalls: []api.ToolCall{lookup}},
				{Role: "tool", ToolName: "explicit", ToolCallID: "shared", Content: "First"},
				{Role: "tool", ToolCallID: "shared", Content: "Second"},
				{Role: "assistant", ToolCalls: []api.ToolCall{save}},
			}},
			inkling2JinjaResult{Prompt: registry + system + "Guide" + end + effort + "0" + end + user + "Hello" + end +
				lookupThinking + lookupText + lookupCall + stop + tool + "explicit<|content_text|>First" + end +
				tool + "lookupsave<|content_text|>Second" + end + saveCall + stop},
		},
		{
			inkling2ParityCase{name: "named assistant mismatch on second call", messages: []api.Message{{Role: "assistant", ToolName: "lookup", ToolCalls: []api.ToolCall{lookup, save}}}},
			inkling2JinjaResult{Error: "assistant name lookup does not match its tool call save"},
		},
		{
			inkling2ParityCase{name: "invalid effort at end of system history", messages: []api.Message{{Role: "system", Content: "Guide"}}, think: &api.ThinkValue{Value: "HIGH"}},
			inkling2JinjaResult{Error: "Unknown reasoning_effort: HIGH"},
		},
		{
			inkling2ParityCase{
				name: "tool image inside text", messages: []api.Message{{Role: "tool", ToolName: "lookup", Content: "Before[img]After", Images: []api.ImageData{{1}}}},
				parts: map[int][]any{0: {"Before", map[string]any{"type": "image"}, "After"}},
			},
			inkling2JinjaResult{Prompt: effort + "0.9" + end + tool + "lookup<|content_text|>Before" + end + tool +
				"lookup<|content_image|><|unused_200054|>" + end + tool + "lookup<|content_text|>After" + end},
		},
		{
			inkling2ParityCase{
				name: "audio is rejected", messages: []api.Message{{Role: "user", Images: []api.ImageData{lfm3AudioClip}}},
				parts: map[int][]any{0: {map[string]any{"type": "input_audio"}}},
			},
			inkling2JinjaResult{Error: "Inkling 2 does not accept audio input"},
		},
	}
	var reference []inkling2JinjaResult
	if os.Getenv("VERIFY_JINJA2") != "" {
		inputs := make([]inkling2JinjaInput, len(tests))
		for i, tc := range tests {
			inputs[i] = tc.referenceInput()
		}
		reference = inkling2RenderJinjaBatch(t, inputs)
	}
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RenderWithRenderer("inkling2", tc.messages, tc.tools, tc.think)
			var gotError string
			if err != nil {
				gotError = err.Error()
			}
			if got != tc.want.Prompt || gotError != tc.want.Error {
				t.Errorf("Go prompt = %q, error = %q\nwant = %q, error = %q", got, gotError, tc.want.Prompt, tc.want.Error)
			}
			if reference != nil && (reference[i].Prompt != tc.want.Prompt || reference[i].Error != tc.want.Error) {
				t.Errorf("Jinja prompt = %q, error = %q\nwant = %q, error = %q", reference[i].Prompt, reference[i].Error, tc.want.Prompt, tc.want.Error)
			}
		})
	}
}

// TestInkling2RendererJinjaParity compares the Go renderer with the pinned
// fixture using the actual Transformers Jinja engine. The same inputs test
// history framing as well as the prompt suffix at each terminal role.
func TestInkling2RendererJinjaParity(t *testing.T) {
	user := api.Message{Role: "user", Content: "What is 7 + 5? Answer with just the number."}
	system := api.Message{Role: "system", Content: "  Use examples.\n"}
	tool := api.Tool{Type: "function", Function: api.ToolFunction{
		Name: "lookup", Description: "Search café 🌍 <&>'",
		Parameters: api.ToolFunctionParameters{
			Type: "object",
			Properties: testPropsOrdered([]orderedProp{
				{"z", api.ToolProperty{Type: api.PropertyType{"array"}, Items: map[string]any{"type": []string{"number", "null"}, "maximum": json.Number("1e16"), "default": nil}}},
				{"A", api.ToolProperty{AnyOf: []api.ToolProperty{{Type: api.PropertyType{"null"}}, {Type: api.PropertyType{"string"}}}}},
			}),
			Required: []string{"A", "z"},
			Extra:    map[string]any{"additionalProperties": false},
			Defs:     json.RawMessage(`{"z":{"type":"string","default":0.00001},"A":{"type":"number","minimum":-1e-7}}`),
		},
	}}
	save := api.Tool{Type: "custom", Function: api.ToolFunction{Name: "save", Parameters: api.ToolFunctionParameters{
		Type: "object", Properties: api.NewToolPropertiesMap(),
	}}}
	rawRegistry := json.RawMessage(`[{"function":{"name":"lookup"}},{"type":"","function":{"name":"save","description":"","parameters":{"type":"object"}}}]`)
	zeroSchema := []api.Tool{
		{Function: api.ToolFunction{Name: "lookup"}},
		{Function: api.ToolFunction{Name: "save", Parameters: api.ToolFunctionParameters{Type: "object"}}},
	}
	mixed := testArgsOrdered([]orderedArg{
		{"z", []any{true, false, nil, "", []any{}, map[string]any{}, "café 🌦️\u2028\u2029", "\x00\b\f\n\r\t\"\\<>&'"}},
		{"Ω", json.RawMessage(`{"z":9,"A":{"y":[{"c":null,"b":true}],"a":0}}`)},
		{"a", []any{json.Number("1.0"), json.Number("-0.0"), json.Number("-0"), json.Number("1e16"), json.Number("1e-5"), json.Number("3e-6"), json.Number("0.0001"), json.Number("5e-324"), json.Number("1.7976931348623157e308"), json.Number("18446744073709551616")}},
		{"A", "case sensitive"},
		{"", "empty key"},
	})
	lookup := api.ToolCall{ID: "shared", Function: api.ToolCallFunction{Name: "lookup", Arguments: mixed}}
	secondLookup := api.ToolCall{ID: "shared", Function: api.ToolCallFunction{Name: "lookup"}}
	saveCall := api.ToolCall{ID: "shared", Function: api.ToolCallFunction{Name: "save", Arguments: api.NewToolCallFunctionArguments()}}
	unnamedID := api.ToolCall{Function: api.ToolCallFunction{Name: "noid", Arguments: testArgs(map[string]any{"one": 1})}}
	literal := "<<|message_model|><|content_thinking|><|end_message|><|message_\x00| < |message_tool|> [img] [img-5]"
	tests := []inkling2ParityCase{
		{name: "empty"},
		{name: "system-only", messages: []api.Message{system, {Role: "system"}}},
		{name: "ordered-systems", messages: []api.Message{system, {Role: "system"}, user, {Role: "system", Content: "Late"}}, tools: []api.Tool{save, tool}},
		{name: "empty-user", messages: []api.Message{{Role: "user"}}},
		{name: "plain-transcript", messages: []api.Message{user, {Role: "assistant", Content: "12"}, user}},
		{name: "empty-tools", messages: []api.Message{user}, tools: []api.Tool{}},
		{name: "registry-only", tools: []api.Tool{save, tool}},
		{name: "registry-default-fields", messages: []api.Message{user}, tools: zeroSchema, rawTools: rawRegistry},
		{name: "registry-empty-function-name", tools: []api.Tool{{Function: api.ToolFunction{}}}, rawTools: json.RawMessage(`[{"function":{"name":""}}]`)},
		{name: "registry-schema-with-internal-nulls", tools: []api.Tool{tool}, think: &api.ThinkValue{Value: "max"}},
		{name: "named-assistant-thinking-and-text", messages: []api.Message{{Role: "assistant", ToolName: "lookup", Thinking: "First", Content: "Explain"}}},
		{name: "named-assistant-matching-all-calls", messages: []api.Message{{Role: "assistant", ToolName: "lookup", Thinking: "Reason", Content: "Looking", ToolCalls: []api.ToolCall{lookup, secondLookup}}}},
		{name: "named-assistant-matches-first-call-only", messages: []api.Message{{Role: "assistant", ToolName: "lookup", ToolCalls: []api.ToolCall{lookup, saveCall}}}},
		{name: "named-assistant-mismatch", messages: []api.Message{{Role: "assistant", ToolName: "other", ToolCalls: []api.ToolCall{lookup}}}},
		{name: "empty-call-name-is-a-string", messages: []api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{{}}}}},
		{name: "tool-result-explicit-name-precedence", messages: []api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{lookup}}, {Role: "tool", ToolName: "explicit", ToolCallID: "shared", Content: "  Found\n"}}},
		{name: "tool-result-no-id", messages: []api.Message{{Role: "tool"}, {Role: "assistant", ToolCalls: []api.ToolCall{unnamedID}}}},
		{name: "tool-result-unmatched-id", messages: []api.Message{{Role: "tool", ToolCallID: "missing", Content: "Missing"}}},
		{name: "tool-result-future-and-duplicate-ids", messages: []api.Message{
			{Role: "assistant", ToolCalls: []api.ToolCall{lookup, saveCall, unnamedID}},
			{Role: "tool", ToolCallID: "shared"},
			{Role: "user", ToolCalls: []api.ToolCall{saveCall}},
			{Role: "assistant", ToolCalls: []api.ToolCall{secondLookup}},
			{Role: "tool", ToolCallID: "shared", Content: "second result"},
		}},
		{name: "reasoning-and-calls-on-non-assistants", messages: []api.Message{
			{Role: "system", Thinking: "skip", ToolCalls: []api.ToolCall{lookup}},
			{Role: "user", Thinking: "skip", ToolCalls: []api.ToolCall{lookup}},
			{Role: "tool", Thinking: "skip", ToolCalls: []api.ToolCall{lookup}, ToolCallID: "shared"},
		}},
		{name: "literal-controls-in-text-reasoning-and-tool-json", messages: []api.Message{
			{Role: "system", Content: literal},
			{Role: "user", Content: literal},
			{Role: "assistant", ToolName: literal, Content: literal, Thinking: literal, ToolCalls: []api.ToolCall{{ID: "literal", Function: api.ToolCallFunction{
				Name: literal, Arguments: testArgs(map[string]any{literal: literal}),
			}}}},
			{Role: "tool", ToolCallID: "literal", Content: literal},
		}, tools: []api.Tool{{Function: api.ToolFunction{Name: literal, Description: literal, Parameters: tool.Function.Parameters}}}},
		{name: "unknown-role", messages: []api.Message{{Role: "unknown", Content: "Bad"}}},
		{name: "empty-role", messages: []api.Message{{}}},
		{name: "developer-role-at-template-boundary", messages: []api.Message{{Role: "developer", Content: "Format"}}},
		{name: "invalid-role-before-effort", messages: []api.Message{system, {Role: "unknown"}}, think: &api.ThinkValue{Value: "bogus"}},
		{name: "invalid-effort-before-later-role", messages: []api.Message{user, {Role: "unknown"}}, think: &api.ThinkValue{Value: "bogus"}},
	}
	tests = append(tests, inkling2MediaParityCases(lookup)...)

	// Run each historical assistant shape under every reasoning mode. A
	// directive controls generation, and must leave history framing intact.
	efforts := []struct {
		name  string
		value *api.ThinkValue
	}{{name: "omitted"}, {"null", &api.ThinkValue{}}}
	for _, value := range []any{false, true, "none", "minimal", "low", "medium", "high", "xhigh", "max", " \tmedium\r\n ", "\u2003high\u00a0", "\x1flow\x1c"} {
		efforts = append(efforts, struct {
			name  string
			value *api.ThinkValue
		}{fmt.Sprintf("%T-%q", value, fmt.Sprint(value)), &api.ThinkValue{Value: value}})
	}
	for _, effort := range efforts {
		for _, content := range []string{"", "Looking that up."} {
			for _, thinking := range []string{"", "Check both tools."} {
				for _, calls := range [][]api.ToolCall{nil, {lookup}, {lookup, saveCall}} {
					name := fmt.Sprintf("effort/%s/text=%t/reasoning=%t/calls=%d", effort.name, content != "", thinking != "", len(calls))
					tests = append(tests, inkling2ParityCase{name: name, messages: []api.Message{system, user, {
						Role: "assistant", Content: content, Thinking: thinking, ToolCalls: calls,
					}}, think: effort.value, tools: []api.Tool{tool, save}})
				}
			}
		}
	}
	for _, value := range []string{"", " ", "none!", "HIGH", "0.125", "ultra"} {
		tests = append(tests, inkling2ParityCase{name: fmt.Sprintf("invalid-effort-%q", value), messages: []api.Message{user}, think: &api.ThinkValue{Value: value}})
	}
	pipeline := []api.Message{
		user,
		{Role: "assistant", Thinking: "Look it up.", ToolCalls: []api.ToolCall{lookup}},
		{Role: "tool", ToolCallID: "shared", Content: "12"},
		{Role: "assistant", Content: "Found it.", Thinking: "Save that.", ToolCalls: []api.ToolCall{saveCall}},
		{Role: "tool", ToolName: "save", ToolCallID: "shared"},
		{Role: "assistant", Content: "The result is 12."},
		user,
	}
	for _, effort := range efforts {
		tests = append(tests, inkling2ParityCase{
			name: "arithmetic/" + effort.name, messages: []api.Message{user}, think: effort.value,
		})
		for end := 1; end <= len(pipeline); end++ {
			tests = append(tests, inkling2ParityCase{
				name:     fmt.Sprintf("agent-transcript/%s/%d-%s", effort.name, end, pipeline[end-1].Role),
				messages: pipeline[:end], tools: []api.Tool{tool, save}, think: effort.value,
			})
		}
	}

	inputs := make([]inkling2JinjaInput, len(tests))
	for i, tc := range tests {
		inputs[i] = tc.referenceInput()
	}
	results := inkling2RenderJinjaBatch(t, inputs)
	tok := inkling2ReferenceTokenizer(t)
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RenderWithRenderer("inkling2", tc.messages, tc.tools, tc.think)
			want := results[i]
			if want.Error != "" || err != nil {
				if err == nil || err.Error() != want.Error || got != "" {
					t.Fatalf("Go error = %v, prompt = %q\nJinja error = %q, prompt = %q", err, got, want.Error, want.Prompt)
				}
				return
			}
			if got != want.Prompt {
				t.Fatalf("Go    %q\nJinja %q", got, want.Prompt)
			}
			if tok != nil {
				ids := tok.Encode(got, tok.AddBOS())
				if !slices.Equal(ids, want.TokenIDs) {
					t.Errorf("Go IDs    %v\nJinja IDs %v", ids, want.TokenIDs)
				}
			}
		})
	}
}

func TestInkling2ParserRendererToolLoopRoundTrip(t *testing.T) {
	const effort = "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>"
	const end = "<|end_message|>"
	const stop = "<|content_model_end_sampling|>"
	const model = "<|message_model|>"
	const lookupArgs = `{"id":17,"query":"check <|end_message|> café","tags":[true,null,{"account":9007199254740993}]}`
	const lookupBody = model + `lookup<|content_invoke_tool_json|>{"name":"lookup","args":` + lookupArgs + `}` + end
	const saveBody = model + `save<|content_invoke_tool_json|>{"name":"save","args":{}}` + end
	const registry = `<|message_system|>tool_declare<|content_xml|>[{"description":"","name":"lookup","parameters":{},"type":"function"},{"description":"","name":"save","parameters":{},"type":"function"}]` + end
	tools := []api.Tool{
		{Type: "function", Function: api.ToolFunction{Name: "lookup"}},
		{Type: "function", Function: api.ToolFunction{Name: "save"}},
	}
	refTools := json.RawMessage(`[{"type":"function","function":{"name":"lookup"}},{"type":"function","function":{"name":"save"}}]`)
	think := &api.ThinkValue{Value: "medium"}
	lookup := args(lookupArgs)
	lookup.Set("tags", json.RawMessage(`[true,null,{"account":9007199254740993}]`))
	turns := []struct {
		message    api.Message
		raw        string
		results    []api.Message
		rawResults string
	}{
		{
			message: api.Message{Role: "assistant", Content: "I’ll check and save that.", Thinking: "Use both tools.", ToolCalls: []api.ToolCall{
				{ID: "call_lookup", Function: api.ToolCallFunction{Name: "lookup", Arguments: lookup}},
				{ID: "call_save", Function: api.ToolCallFunction{Name: "save", Index: 1}},
			}},
			raw: model + "<|content_thinking|>Use both tools." + end + model + "<|content_text|>I’ll check and save that." + end + lookupBody + saveBody + stop,
			results: []api.Message{
				{Role: "tool", ToolCallID: "call_lookup", Content: "Found 17."},
				{Role: "tool", ToolCallID: "call_save", ToolName: "save", Content: "Saved."},
			},
			rawResults: "<|message_tool|>lookup<|content_text|>Found 17." + end + "<|message_tool|>save<|content_text|>Saved." + end,
		},
		{
			message: api.Message{Role: "assistant", Thinking: "Verify the save.", ToolCalls: []api.ToolCall{
				{ID: "call_verify", Function: api.ToolCallFunction{Name: "lookup", Arguments: lookup}},
			}},
			raw:        model + "<|content_thinking|>Verify the save." + end + lookupBody + stop,
			results:    []api.Message{{Role: "tool", ToolCallID: "call_verify", Content: "Verified."}},
			rawResults: "<|message_tool|>lookup<|content_text|>Verified." + end,
		},
		{
			message: api.Message{Role: "assistant", Thinking: "Both results agree.", Content: "The saved result is 17."},
			raw:     model + "<|content_thinking|>Both results agree." + end + model + "<|content_text|>The saved result is 17." + end + stop,
		},
	}

	const question = "Look up the result and save it."
	initial := api.Message{Role: "user", Content: question}
	var reference []inkling2JinjaResult
	if os.Getenv("VERIFY_JINJA2") != "" {
		expectedHistory := []api.Message{initial}
		var inputs []inkling2JinjaInput
		for _, turn := range turns {
			assistant := inkling2ParityCase{messages: []api.Message{turn.message}, think: think}
			inputs = append(inputs, assistant.referenceInput())
			expectedHistory = append(expectedHistory, turn.message)
			expectedHistory = append(expectedHistory, turn.results...)
			next := inkling2ParityCase{messages: expectedHistory, rawTools: refTools, think: think}
			inputs = append(inputs, next.referenceInput())
		}
		reference = inkling2RenderJinjaBatch(t, inputs)
	}
	history := []api.Message{initial}
	wantHistory := registry + effort + "<|message_user|><|content_text|>" + question + end
	parser := parsers.ParserForName("inkling2")
	for i, turn := range turns {
		raw := turn.raw
		if reference != nil {
			ref := reference[2*i]
			if ref.Error != "" || ref.Prompt != effort+raw {
				t.Fatalf("turn %d: Jinja assistant turn = %q, error = %q\nwant %q", i+1, ref.Prompt, ref.Error, effort+raw)
			}
			raw = ref.Prompt[len(effort):]
		}
		var message api.Message
		for split := range len(raw) + 1 {
			parser.Init(tools, nil, think)
			message = api.Message{Role: "assistant"}
			for j, chunk := range []string{raw[:split], raw[split:]} {
				content, thinking, calls, err := parser.Add(chunk, j == 1)
				if err != nil {
					t.Fatalf("turn %d split %d: parse: %v", i+1, split, err)
				}
				message.Content += content
				message.Thinking += thinking
				message.ToolCalls = append(message.ToolCalls, calls...)
			}
			if message.Content != turn.message.Content || message.Thinking != turn.message.Thinking || len(message.ToolCalls) != len(turn.message.ToolCalls) {
				t.Fatalf("turn %d split %d: content = %q, thinking = %q, tool calls = %v", i+1, split, message.Content, message.Thinking, message.ToolCalls)
			}
			for j, call := range message.ToolCalls {
				want := turn.message.ToolCalls[j]
				if call.Function.Index != want.Function.Index || call.Function.Name != want.Function.Name ||
					call.Function.Arguments.String() != want.Function.Arguments.String() {
					t.Fatalf("turn %d split %d: tool call %d (%q, index %d, args %s), want (%q, index %d, args %s)",
						i+1, split, j, call.Function.Name, call.Function.Index, call.Function.Arguments.String(),
						want.Function.Name, want.Function.Index, want.Function.Arguments.String())
				}
				// The server supplies the call IDs before clients append their results.
				message.ToolCalls[j].ID = want.ID
			}
		}
		history = append(history, message)
		history = append(history, turn.results...)
		wantHistory += turn.raw + turn.rawResults
		got, err := RenderWithRenderer("inkling2", history, tools, think)
		if err != nil || got != wantHistory {
			t.Fatalf("turn %d: Go history = %q, error = %v\nwant %q", i+1, got, err, wantHistory)
		}
		if reference != nil {
			if ref := reference[2*i+1]; ref.Error != "" || ref.Prompt != wantHistory {
				t.Fatalf("turn %d: Jinja history = %q, error = %q\nwant %q", i+1, ref.Prompt, ref.Error, wantHistory)
			}
		}
	}
}

// INKLING2_TOKENIZER_DIR optionally enables the production-tokenizer check
// against the publisher's tokenizer.json for the same exact rendered prompts.
func inkling2ReferenceTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	dir := os.Getenv("INKLING2_TOKENIZER_DIR")
	if dir == "" {
		return nil
	}
	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(read("tokenizer.json"), &tokenizer.TokenizerConfig{
		ConfigJSON:          read("config.json"),
		TokenizerConfigJSON: read("tokenizer_config.json"),
	})
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func inkling2MediaParityCases(call api.ToolCall) []inkling2ParityCase {
	image := map[string]any{"type": "image"}
	imageURL := map[string]any{"type": "image_url"}
	inputImage := map[string]any{"type": "input_image"}
	text := func(s string) map[string]any { return map[string]any{"type": "text", "text": s} }
	one := []api.ImageData{{1}}
	three := []api.ImageData{{1}, {2}, {3}}
	wav := lfm3AudioClip
	mp3 := api.ImageData{0xff, 0xfb, 0x90, 0x64}
	inputAudio := map[string]any{"type": "input_audio"}
	audio := map[string]any{"type": "audio"}
	return []inkling2ParityCase{
		{name: "user-image-only", messages: []api.Message{{Role: "user", Images: one}}, parts: map[int][]any{0: {image}}},
		{name: "user-image-plus-text", messages: []api.Message{{Role: "user", Content: "Read", Images: one}}, parts: map[int][]any{0: {imageURL, text("Read")}}},
		{name: "image-order-and-tool-result", messages: []api.Message{
			{Role: "user", Content: "[img]Compare[img] with the last [img]", Images: three},
			{Role: "assistant", ToolCalls: []api.ToolCall{call}},
			{Role: "tool", ToolCallID: call.ID, Content: "Before[img]After", Images: one},
			{Role: "tool", ToolName: "other", Images: one},
		}, parts: map[int][]any{
			0: {image, text("Compare"), inputImage, text(" with the last "), imageURL},
			2: {text("Before"), image, text("After")},
			3: {inputImage},
		}},
		{name: "indexed-images-and-unplaced-images", messages: []api.Message{
			{Role: "user", Content: "See [img-0]", Images: three},
			{Role: "tool", Content: "Result [img-3]", Images: one},
		}, parts: map[int][]any{0: {image, image, text("See "), image}, 1: {text("Result "), imageURL}}},
		{name: "mixed-indexed-and-anonymous-images", messages: []api.Message{
			{Role: "user", Content: "[img-0] [?] [img] then [img-999] or [img-]", Images: three},
		}, parts: map[int][]any{0: {image, image, text(" [?] "), image, text(" then [img-999] or [img-]")}}},
		{
			name: "assistant-image-error-after-effort-error", messages: []api.Message{{Role: "assistant", Images: one}},
			parts: map[int][]any{0: {image}}, think: &api.ThinkValue{Value: "bogus"},
		},
		{name: "leading-system-with-image", messages: []api.Message{{Role: "system", Images: one}}, parts: map[int][]any{0: {image}}},
		{
			name: "leading-system-image-error-before-effort-error", messages: []api.Message{{Role: "system", Images: one}, {Role: "user", Content: "Read"}},
			parts: map[int][]any{0: {image}}, think: &api.ThinkValue{Value: "bogus"},
		},
		{name: "assistant-with-image", messages: []api.Message{{Role: "assistant", Images: one}}, parts: map[int][]any{0: {image}}},
		{
			name: "assistant-image-error-before-mismatched-tool-name", messages: []api.Message{{
				Role: "assistant", ToolName: "other", Thinking: "Read this", Content: "First[img]", Images: one, ToolCalls: []api.ToolCall{call},
			}}, parts: map[int][]any{0: {text("First"), image}},
		},
		{
			name: "late-system-with-image", messages: []api.Message{{Role: "user", Content: "Hello"}, {Role: "system", Images: one}},
			parts: map[int][]any{1: {image}},
		},
		{name: "user-with-wav", messages: []api.Message{{Role: "user", Images: []api.ImageData{wav}}}, parts: map[int][]any{0: {inputAudio}}},
		{name: "user-with-id3", messages: []api.Message{{Role: "user", Images: []api.ImageData{[]byte("ID3")}}}, parts: map[int][]any{0: {inputAudio}}},
		{
			name: "tool-with-image-then-mp3", messages: []api.Message{{Role: "tool", ToolCallID: call.ID, Content: "Look[img] then hear[img]", Images: []api.ImageData{one[0], mp3}}},
			parts: map[int][]any{0: {text("Look"), image, text(" then hear"), audio}},
		},
		{
			name: "assistant-with-audio-before-image", messages: []api.Message{{Role: "assistant", Thinking: "Seen", Images: []api.ImageData{wav, one[0]}}},
			parts: map[int][]any{0: {audio, image}},
		},
		{
			name: "system-with-image-before-audio", messages: []api.Message{{Role: "system", Images: []api.ImageData{one[0], wav}}},
			parts: map[int][]any{0: {image, audio}},
		},
		{
			name: "system-with-reordered-audio-before-image", messages: []api.Message{{Role: "system", Content: "[img-1][img-0]", Images: []api.ImageData{one[0], wav}}},
			parts: map[int][]any{0: {audio, image}},
		},
	}
}

func inkling2RenderJinjaBatch(t *testing.T, inputs []inkling2JinjaInput) []inkling2JinjaResult {
	t.Helper()
	if os.Getenv("VERIFY_JINJA2") == "" {
		t.Skip("set VERIFY_JINJA2=1 to run publisher Jinja comparison")
	}
	verifyLFM3TemplateFixture(t, inkling2Template, inkling2TemplateSHA256)
	python := os.Getenv("INKLING2_JINJA_PYTHON")
	if python == "" {
		python = "python3"
	}
	input, err := json.Marshal(inputs)
	if err != nil {
		t.Fatal(err)
	}
	const script = `
import json
import sys
from pathlib import Path

from jinja2 import TemplateError
from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

template = _compile_jinja_template(Path(sys.argv[1]).read_text(encoding="utf-8"))
tokenizer = Tokenizer.from_file(str(Path(sys.argv[2]) / "tokenizer.json")) if sys.argv[2] else None
results = []
for kwargs in json.load(sys.stdin):
    try:
        prompt = template.render(**kwargs)
        result = {"prompt": prompt}
        if tokenizer is not None:
            result["tokenids"] = tokenizer.encode(prompt, add_special_tokens=False).ids
        results.append(result)
    except TemplateError as error:
        results.append({"error": str(error)})
print(json.dumps(results, ensure_ascii=True))
`
	cmd := exec.CommandContext(t.Context(), python, "-c", script, inkling2Template, os.Getenv("INKLING2_TOKENIZER_DIR"))
	cmd.Stdin = bytes.NewReader(input)
	output, err := cmd.Output()
	if err != nil {
		if exit, ok := err.(*exec.ExitError); ok {
			t.Fatalf("Jinja: %v\n%s", err, exit.Stderr)
		}
		t.Fatal(err)
	}
	var results []inkling2JinjaResult
	if err := json.Unmarshal(output, &results); err != nil {
		t.Fatal(err)
	}
	if len(results) != len(inputs) {
		t.Fatalf("received %d Jinja results for %d cases", len(results), len(inputs))
	}
	return results
}

// These cases exercise Jinja input shapes that api.Message, api.Tool, or
// api.ThinkValue flatten or reject before the renderer is called.
func TestInkling2PublisherOnlyJinjaBranches(t *testing.T) {
	const effortStart = "<|message_system|><|content_text|>Thinking effort level: "
	const end = "<|end_message|>"
	const userText = "<|message_user|><|content_text|>"
	const userImage = "<|message_user|><|content_image|><|unused_200054|>" + end
	const assistantHeader = "<|message_model|>"
	const stop = "<|content_model_end_sampling|>"
	const toolDeclaration = "<|message_system|>tool_declare<|content_xml|>"
	const lookup = assistantHeader + "lookup<|content_invoke_tool_json|>" + `{"name":"lookup","args":{}}` + end + stop
	const defaultEffort = effortStart + "0.9" + end

	type rawCase struct {
		name, messages, tools, effort string
		want                          inkling2JinjaResult
	}
	tests := []rawCase{
		{name: "explicit-null-effort", effort: "null", want: inkling2JinjaResult{Prompt: defaultEffort}},
		{name: "zero-numeric-effort", effort: "0", want: inkling2JinjaResult{Prompt: effortStart + "0" + end}},
		{name: "negative-zero-effort", effort: "-0.0", want: inkling2JinjaResult{Prompt: effortStart + "0" + end}},
		{name: "max-numeric-effort", effort: "0.99", want: inkling2JinjaResult{Prompt: effortStart + "0.99" + end}},
		{name: "nonpreset-numeric-effort", effort: "0.125", want: inkling2JinjaResult{Prompt: effortStart + "0.125" + end}},
		{name: "numeric-effort-below-range", effort: "-0.001", want: inkling2JinjaResult{Error: "reasoning_effort must be in [0.0, 0.99]"}},
		{name: "numeric-effort-above-range", effort: "0.991", want: inkling2JinjaResult{Error: "reasoning_effort must be in [0.0, 0.99]"}},
		{name: "raw-boolean-false-effort", effort: "false", want: inkling2JinjaResult{Prompt: effortStart + "0" + end}},
		{name: "raw-boolean-true-effort", effort: "true", want: inkling2JinjaResult{Error: "reasoning_effort must be in [0.0, 0.99]"}},
		{name: "tools-null", tools: "null", want: inkling2JinjaResult{Prompt: defaultEffort}},
		{name: "tools-empty", tools: "[]", want: inkling2JinjaResult{Prompt: defaultEffort}},
		{
			name: "bare-and-wrapped-declarations", tools: `[{"name":"a"},{"type":null,"function":{"name":"b","description":null,"parameters":null}},{"type":"other","function":{"name":"c","description":" ","parameters":{}}}]`,
			want: inkling2JinjaResult{Prompt: toolDeclaration + `[{"description":"","name":"a","parameters":{},"type":"function"},{"description":"","name":"b","parameters":{},"type":"function"},{"description":" ","name":"c","parameters":{},"type":"other"}]` + end + defaultEffort},
		},
		{name: "missing-user-content", messages: `[{"role":"user"}]`, want: inkling2JinjaResult{Prompt: defaultEffort}},
		{name: "null-user-content", messages: `[{"role":"user","content":null}]`, want: inkling2JinjaResult{Prompt: defaultEffort}},
		{name: "empty-user-parts", messages: `[{"role":"user","content":[]}]`, want: inkling2JinjaResult{Prompt: defaultEffort}},
		{name: "missing-assistant-content", messages: `[{"role":"assistant"}]`, want: inkling2JinjaResult{Prompt: defaultEffort + stop}},
		{name: "null-thinking-and-empty-calls", messages: `[{"role":"assistant","content":"","reasoning_content":null,"tool_calls":[]}]`, want: inkling2JinjaResult{Prompt: defaultEffort + assistantHeader + "<|content_text|>" + end + stop}},
		{
			name: "explicit-empty-names-use-call-id",
			messages: `[{"role":"assistant","name":"","content":"","tool_calls":[{"id":"call_1","function":{"name":"lookup"}}]},` +
				`{"role":"tool","name":"","tool_call_id":"call_1","content":"Found it"}]`,
			want: inkling2JinjaResult{Prompt: defaultEffort + lookup + "<|message_tool|>lookup<|content_text|>Found it" + end},
		},
		{
			name:     "explicit-empty-call-id",
			messages: `[{"role":"tool","tool_call_id":"","content":"No match"}]`,
			want:     inkling2JinjaResult{Prompt: defaultEffort + "<|message_tool|><|content_text|>No match" + end},
		},
		{
			name: "bare-and-typed-text-parts", messages: `[{"role":"user","content":["", "bare", {"text":"untyped"}, {"type":"text","text":"typed"}, {"type":"input_text","text":"input"}, {"type":"input_text","text":8}, {"type":"text","text":null}, {"type":"text"}, {}]}]`,
			want: inkling2JinjaResult{Prompt: defaultEffort + userText + end + userText + "bare" + end + userText + "untyped" + end + userText + "typed" + end + userText + "input" + end + userText + end + userText + end + userText + end + userText + end},
		},
		{
			name: "assistant-text-parts-between-reasoning-and-calls", messages: `[{"role":"assistant","name":"lookup","content":[{"type":"text","text":"one"},"two"],"reasoning_content":"think","tool_calls":[{"function":{"name":"lookup"}}]}]`,
			want: inkling2JinjaResult{Prompt: defaultEffort + assistantHeader + "lookup<|content_thinking|>think" + end + assistantHeader + "lookup<|content_text|>one" + end + assistantHeader + "lookup<|content_text|>two" + end + lookup},
		},
		{name: "image-aliases", messages: `[{"role":"user","content":[{"type":"input_image"},{"type":"image"},{"type":"image_url"}]}]`, want: inkling2JinjaResult{Prompt: defaultEffort + userImage + userImage + userImage}},
		{name: "system-image", messages: `[{"role":"system","content":[{"type":"input_image"}]}]`, want: inkling2JinjaResult{Error: "Image content is only supported for user and tool messages"}},
		{name: "unsupported-video", messages: `[{"role":"user","content":[{"type":"video"}]}]`, want: inkling2JinjaResult{Error: "Unsupported content part type: video"}},
		{name: "unsupported-unknown-part", messages: `[{"role":"tool","content":[{"type":"blob"}]}]`, want: inkling2JinjaResult{Error: "Unsupported content part type: blob"}},
		{name: "nonstring-call-name", messages: `[{"role":"assistant","tool_calls":[{"function":{"name":12}}]}]`, want: inkling2JinjaResult{Error: "tool call function name must be a string"}},
		{name: "null-call-name", messages: `[{"role":"assistant","tool_calls":[{"function":{"name":null}}]}]`, want: inkling2JinjaResult{Error: "tool call function name must be a string"}},
		{name: "absent-name-at-tool-lookup", messages: `[{"role":"tool","tool_call_id":"id","content":"result"},{"role":"assistant","tool_calls":[{"id":"id","function":{}}]}]`, want: inkling2JinjaResult{Error: "tool call function name must be a string"}},
		{name: "stringified-call-arguments", messages: `[{"role":"assistant","tool_calls":[{"function":{"name":"lookup","arguments":"{}"}}]}]`, want: inkling2JinjaResult{Error: "tool call arguments must be a parsed object, not a JSON string; canonicalize upstream"}},
	}
	for _, alias := range []string{"audio", "input_audio", "audio_url"} {
		tests = append(tests, rawCase{
			name: "unsupported-" + alias, messages: fmt.Sprintf(`[{"role":"tool","content":[{"type":%q}]}]`, alias),
			want: inkling2JinjaResult{Error: "Inkling 2 does not accept audio input"},
		})
	}
	for _, arguments := range []string{"null", "false", "0", `""`, "[]", "{}", "[1]", "true", "2"} {
		want := inkling2JinjaResult{Prompt: defaultEffort + lookup}
		if arguments == "[1]" || arguments == "true" || arguments == "2" {
			want = inkling2JinjaResult{Error: "tool call arguments must be an object"}
		}
		tests = append(tests, rawCase{
			name: "arguments-" + arguments, messages: fmt.Sprintf(`[{"role":"assistant","tool_calls":[{"function":{"name":"lookup","arguments":%s}}]}]`, arguments), want: want,
		})
	}
	inputs := make([]inkling2JinjaInput, len(tests))
	for i, tc := range tests {
		inputs[i] = inkling2JinjaInput{Messages: json.RawMessage("[]"), AddGenerationPrompt: true}
		if tc.messages != "" {
			inputs[i].Messages = json.RawMessage(tc.messages)
		}
		if tc.tools != "" {
			inputs[i].Tools = json.RawMessage(tc.tools)
		}
		if tc.effort != "" {
			inputs[i].Effort = json.RawMessage(tc.effort)
		}
	}
	results := inkling2RenderJinjaBatch(t, inputs)
	for i, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := results[i]; got.Prompt != tc.want.Prompt || got.Error != tc.want.Error {
				t.Fatalf("Jinja got %+q, want %+q", got, tc.want)
			}
		})
	}
}

func TestInkling2JinjaHasNoGenerationOpener(t *testing.T) {
	messages := []json.RawMessage{
		json.RawMessage(`[]`),
		json.RawMessage(`[{"role":"system","content":"guide"}]`),
		json.RawMessage(`[{"role":"user","content":"question"}]`),
		json.RawMessage(`[{"role":"assistant","content":"partial answer"}]`),
		json.RawMessage(`[{"role":"assistant","reasoning_content":"consider","tool_calls":[{"function":{"name":"lookup"}}]}]`),
		json.RawMessage(`[{"role":"tool","name":"lookup","content":"result"}]`),
	}
	inputs := make([]inkling2JinjaInput, 0, len(messages)*2)
	for _, history := range messages {
		inputs = append(inputs, inkling2JinjaInput{Messages: history, AddGenerationPrompt: true}, inkling2JinjaInput{Messages: history})
	}
	results := inkling2RenderJinjaBatch(t, inputs)
	for i := 0; i < len(results); i += 2 {
		if results[i].Error != "" || results[i+1].Error != "" || results[i].Prompt != results[i+1].Prompt {
			t.Errorf("history %s: generation true %+q, false %+q", messages[i/2], results[i], results[i+1])
		}
	}
}
