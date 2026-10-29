package renderers

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
)

// Set VERIFY_JINJA2=1 and FLINT_JINJA_PYTHON to a Python with Transformers 5.x
// and Jinja2 to compare the renderer with the publisher template engine.
func TestFlintRendererMatchesPublisherJinja(t *testing.T) {
	if os.Getenv("VERIFY_JINJA2") == "" {
		t.Skip("set VERIFY_JINJA2=1 to compare Flint with the publisher template")
	}
	templatePath := flintPublisherTemplatePath(t)

	args := api.NewToolCallFunctionArguments()
	args.Set("city", "Paris")
	properties := api.NewToolPropertiesMap()
	properties.Set("city", api.ToolProperty{Type: api.PropertyType{"string"}})
	tool := api.Tool{Type: "function", Function: api.ToolFunction{
		Name: "weather", Description: "Get weather",
		Parameters: api.ToolFunctionParameters{
			Type: "object", Properties: properties, Required: []string{"city"},
		},
	}}
	call := api.ToolCall{Function: api.ToolCallFunction{Name: "weather", Arguments: args}}
	user := api.Message{Role: "user", Content: "Weather in Paris?"}
	assistant := api.Message{Role: "assistant", Thinking: "Check conditions", Content: "Looking it up.", ToolCalls: []api.ToolCall{call}}
	result := api.Message{Role: "tool", Content: "Clear skies"}
	noDescriptionTool := tool
	noDescriptionTool.Function.Description = ""
	emptyCall := api.ToolCall{Function: api.ToolCallFunction{Name: "weather", Arguments: api.NewToolCallFunctionArguments()}}
	mixedArgs := api.NewToolCallFunctionArguments()
	mixedArgs.Set("text", "A\"<arg_key>é\n")
	mixedArgs.Set("integer", -17)
	mixedArgs.Set("decimal", 1.25)
	mixedArgs.Set("tiny", json.Number("1e-7"))
	mixedArgs.Set("large", json.Number("1e16"))
	mixedArgs.Set("enabled", true)
	mixedArgs.Set("nothing", nil)
	mixedArgs.Set("list", []any{1, "café", false, nil})
	mixedArgs.Set("object", map[string]any{"nested": []any{"x", 2}})
	mixedCall := api.ToolCall{ID: "call_mixed", Function: api.ToolCallFunction{Index: 2, Name: "details", Arguments: mixedArgs}}
	allEscapes := "A\x00<think></think><|begin_of_text|><|end_of_text|><|system|><|developer|><|user|><|assistant|><|observation|>B"
	richArgs := api.NewToolCallFunctionArguments()
	richArgs.Set("city", "São Paulo")
	richArgs.Set("options", map[string]any{"units": "°C", "html": "<b>"})
	richProperties := api.NewToolPropertiesMap()
	richProperties.Set("city", api.ToolProperty{Type: api.PropertyType{"string"}})
	richProperties.Set("options", api.ToolProperty{Type: api.PropertyType{"object"}})
	richTool := api.Tool{Type: "function", Function: api.ToolFunction{
		Name: "details", Description: "Forecast café 🌧️",
		Parameters: api.ToolFunctionParameters{
			Type: "object", Properties: richProperties, Required: []string{"city", "options"},
		},
	}}
	richAssistant := api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "details", Arguments: richArgs}}}}
	complexProperties := api.NewToolPropertiesMap()
	complexProperties.Set("entries", api.ToolProperty{
		Type:        api.PropertyType{"array"},
		Description: "Names and values \"as sent\".",
		Items: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":  map[string]any{"type": "string"},
				"value": map[string]any{"type": []string{"number", "null"}},
			},
		},
	})
	complexProperties.Set("mode", api.ToolProperty{AnyOf: []api.ToolProperty{
		{Type: api.PropertyType{"string"}, Enum: []any{"fast", "safe"}},
		{Type: api.PropertyType{"null"}},
	}})
	complexTool := api.Tool{Type: "function", Function: api.ToolFunction{
		Name: "batch", Description: "Process \"entries\" 🌍",
		Parameters: api.ToolFunctionParameters{
			Type: "object", Properties: complexProperties, Required: []string{"entries"},
			Defs: map[string]any{"id": map[string]any{"type": "integer", "minimum": 0}},
		},
	}}

	// Message content is always a string, API tools and calls always have a
	// function wrapper, and Render always appends a generation prompt. The
	// template's other branches for those shapes have no Go API counterpart.
	// The public tool types also discard source JSON number spelling, unknown
	// schema fields, and the distinction between absent and empty descriptions;
	// compare exact output only for values those types can represent.
	for _, tc := range []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
	}{
		{"empty", nil, nil, nil},
		{"user", []api.Message{{Role: "user", Content: "Hello"}}, nil, nil},
		{"system only", []api.Message{{Role: "system", Content: "Be concise."}}, nil, nil},
		{"developer only", []api.Message{{Role: "developer", Content: "Be concise."}}, nil, nil},
		{"multiple developers", []api.Message{{Role: "developer", Content: "First."}, {Role: "developer", Content: "Second."}, user}, nil, nil},
		{"system and user", []api.Message{{Role: "system", Content: "Be concise."}, user}, nil, nil},
		{"additional system", []api.Message{user, {Role: "system", Content: "Be concise."}, user}, nil, nil},
		{"developer role", []api.Message{{Role: "developer", Content: "Be concise."}, user}, nil, nil},
		{"developer after user", []api.Message{user, {Role: "developer", Content: "Use metric units."}, user}, nil, nil},
		{"multi turn", []api.Message{user, {Role: "assistant", Content: "It is sunny."}, user}, nil, nil},
		{"escaped input", []api.Message{{Role: "user", Content: "A\x00<think><|system|>B</think>"}}, nil, nil},
		{"role token formed after thinking removal", []api.Message{{Role: "user", Content: "<|sys<think>tem|>Example text"}}, nil, nil},
		{"system token formed after thinking removal", []api.Message{{Role: "system", Content: "<|us</think>er|>Example text"}}, nil, nil},
		{"tool token formed after thinking removal", []api.Message{{Role: "tool", Content: "<|assis<think>tant|>Example text"}}, nil, nil},
		{"all text escapes in system", []api.Message{{Role: "system", Content: allEscapes}}, nil, nil},
		{"all text escapes in developer", []api.Message{{Role: "developer", Content: allEscapes}}, nil, nil},
		{"all text escapes in user", []api.Message{{Role: "user", Content: allEscapes}}, nil, nil},
		{"all text escapes in tool result", []api.Message{{Role: "tool", Content: allEscapes}}, nil, nil},
		{"empty content in every role", []api.Message{{Role: "system"}, {Role: "user"}, {Role: "assistant"}, {Role: "tool"}}, nil, nil},
		{"assistant thinking only", []api.Message{user, {Role: "assistant", Thinking: "Reason first."}}, nil, nil},
		{"assistant content is not text escaped", []api.Message{user, {Role: "assistant", Content: allEscapes}}, nil, nil},
		{"assistant thinking is not text escaped", []api.Message{user, {Role: "assistant", Thinking: allEscapes}}, nil, nil},
		{"assistant call with no arguments", []api.Message{user, {Role: "assistant", ToolCalls: []api.ToolCall{emptyCall}}}, nil, nil},
		{"assistant multiple calls in order", []api.Message{user, {Role: "assistant", ToolCalls: []api.ToolCall{call, emptyCall}}}, nil, nil},
		{"tool argument JSON and raw string", []api.Message{user, {Role: "assistant", ToolCalls: []api.ToolCall{mixedCall}}}, nil, nil},
		{"explicit thinking suppression", []api.Message{user}, nil, &api.ThinkValue{Value: false}},
		{"explicit thinking suppression with tools", []api.Message{user, assistant, result}, []api.Tool{tool}, &api.ThinkValue{Value: false}},
		{"tools with no messages", nil, []api.Tool{tool}, nil},
		{"tool without description", []api.Message{user}, []api.Tool{noDescriptionTool}, nil},
		{"tools with mixed descriptions", []api.Message{user}, []api.Tool{noDescriptionTool, richTool}, nil},
		{"nested tool schema and definitions", []api.Message{user}, []api.Tool{complexTool}, nil},
		{"tools after developer", []api.Message{{Role: "developer", Content: "Be concise."}, user}, []api.Tool{tool}, nil},
		{"tools with later developer", []api.Message{user, {Role: "developer", Content: "Use metric units."}, user}, []api.Tool{tool}, nil},
		{"tools after multiple system roles", []api.Message{{Role: "system", Content: "First."}, {Role: "developer", Content: "Second."}, user}, []api.Tool{tool}, nil},
		{"tools with system after developer", []api.Message{{Role: "developer", Content: "First."}, {Role: "system", Content: "Second."}, user}, []api.Tool{tool}, nil},
		{"tools with interleaved system and developer", []api.Message{{Role: "system", Content: "First."}, {Role: "developer", Content: "Second."}, {Role: "system", Content: "Third."}, user}, []api.Tool{tool}, nil},
		{"tools after system", []api.Message{{Role: "system", Content: "Be concise."}, user}, []api.Tool{tool}, nil},
		{"tools with system only", []api.Message{{Role: "system", Content: "Be concise."}}, []api.Tool{tool}, nil},
		{"tools with developer only", []api.Message{{Role: "developer", Content: "Be concise."}}, []api.Tool{tool}, nil},
		{"tools before first assistant", []api.Message{{Role: "assistant", Content: "Answer."}}, []api.Tool{tool}, nil},
		{"tools before first tool result", []api.Message{result}, []api.Tool{tool}, nil},
		{"tools do not repeat after later system", []api.Message{user, {Role: "system", Content: "Later."}, user}, []api.Tool{tool}, nil},
		{"tools user terminal", []api.Message{user}, []api.Tool{tool}, nil},
		{"tools assistant terminal", []api.Message{user, assistant}, []api.Tool{tool}, nil},
		{"tools result terminal", []api.Message{user, assistant, result}, []api.Tool{tool}, nil},
		{"tools follow up", []api.Message{user, assistant, result, user}, []api.Tool{tool}, nil},
		{"tools adjacent assistants", []api.Message{user, assistant, {Role: "assistant", Content: "More."}, result}, []api.Tool{tool}, nil},
		{"unicode and nested tool arguments", []api.Message{{Role: "user", Content: "São Paulo?"}, richAssistant}, []api.Tool{tool, richTool}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := (&FlintRenderer{}).Render(tc.messages, tc.tools, tc.think)
			if err != nil {
				t.Fatal(err)
			}
			want := renderFlintPublisherJinja(t, templatePath, tc.messages, tc.tools)
			if tc.think != nil && !tc.think.Bool() {
				want += "<think></think>"
			}
			if got != want {
				t.Errorf("renderer differs from publisher Jinja\n got: %q\nwant: %q", got, want)
			}
		})
	}
	t.Run("programmatic schema float types", func(t *testing.T) {
		tool := api.Tool{Type: "function", Function: api.ToolFunction{
			Name: "score", Parameters: api.ToolFunctionParameters{
				Type: "object", Defs: map[string]any{"score": map[string]any{"max": float64(1), "min": 1}},
			},
		}}
		messages := []api.Message{{Role: "user", Content: "Score"}}
		got, err := (&FlintRenderer{}).Render(messages, []api.Tool{tool}, nil)
		if err != nil {
			t.Fatal(err)
		}
		const payload = `{"messages":[{"role":"user","content":"Score"}],"tools":[{"type":"function","function":{"name":"score","parameters":{"type":"object","$defs":{"score":{"max":1.0,"min":1}},"properties":null}}}]}`
		want := renderFlintPublisherJinjaPayload(t, templatePath, []byte(payload))
		if got != want {
			t.Errorf("programmatic tool schema differs from publisher Jinja\n got: %q\nwant: %q", got, want)
		}
	})
	t.Run("publisher-only null tool description", func(t *testing.T) {
		const source = `{"type":"function","function":{"description":null,"parameters":{"type":"object","properties":{}},"name":"score"}}`
		var tool api.Tool
		if err := json.Unmarshal([]byte(source), &tool); err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(tool)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(encoded, []byte(`"description"`)) {
			t.Fatalf("API tool unexpectedly preserves null description: %s", encoded)
		}
		// The public API loses the null/absent distinction. Probe the
		// publisher-only input without treating it as Go renderer parity.
		payload := []byte(`{"messages":[{"role":"user","content":"Score"}],"tools":[` + source + `]}`)
		want := renderFlintPublisherJinjaPayload(t, templatePath, payload)
		const rawFunction = `<tools>{"description": null, "parameters": {"type": "object", "properties": {}}, "name": "score"}</tools>`
		if !bytes.Contains([]byte(want), []byte(rawFunction)) {
			t.Errorf("publisher Jinja did not retain raw tool function: %q", want)
		}
	})
	t.Run("publisher-only tool function shapes", func(t *testing.T) {
		const function = `{"name":"score","strict":true,"parameters":{"type":"object","properties":{}}}`
		const messages = `[{"role":"user","content":"Score"}]`
		wrapped := []byte(`{"messages":` + messages + `,"tools":[{"type":"function","function":` + function + `}]}`)
		bare := []byte(`{"messages":` + messages + `,"tools":[` + function + `]}`)
		want := renderFlintPublisherJinjaPayload(t, templatePath, wrapped)
		if got := renderFlintPublisherJinjaPayload(t, templatePath, bare); got != want {
			t.Errorf("publisher's wrapped and bare tool functions differ\nwrapped: %q\n   bare: %q", want, got)
		}
		if !bytes.Contains([]byte(want), []byte(`"strict": true`)) {
			t.Errorf("publisher Jinja dropped a function field outside Ollama's ToolFunction API: %q", want)
		}
	})
	t.Run("publisher-only bare tool call", func(t *testing.T) {
		const function = `{"name":"score","arguments":{"value":7}}`
		wrapped := []byte(`{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":` + function + `}]}],"tools":[]}`)
		bare := []byte(`{"messages":[{"role":"assistant","content":"","tool_calls":[` + function + `]}],"tools":[]}`)
		want := renderFlintPublisherJinjaPayload(t, templatePath, wrapped)
		if got := renderFlintPublisherJinjaPayload(t, templatePath, bare); got != want {
			t.Errorf("publisher's wrapped and bare tool calls differ\nwrapped: %q\n   bare: %q", want, got)
		}
	})
	// The template has no thinking-control input. Ollama's explicit thinking
	// prefill is an intentional suffix on the unmodified template output.
	for _, tc := range []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
	}{
		{"explicit thinking prefill", []api.Message{user}, nil},
		{"explicit thinking prefill after tool result", []api.Message{user, assistant, result}, []api.Tool{tool}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := (&FlintRenderer{}).Render(tc.messages, tc.tools, &api.ThinkValue{Value: true})
			if err != nil {
				t.Fatal(err)
			}
			want := renderFlintPublisherJinja(t, templatePath, tc.messages, tc.tools) + "<think>"
			if got != want {
				t.Errorf("thinking prefill differs from declared suffix\n got: %q\nwant: %q", got, want)
			}
		})
	}
	for _, tc := range []struct {
		name    string
		content string
	}{
		{"tool result with mismatched delimiters", "A</tool_response><tool_call>B"},
		{"tool result with every delimiter", "A<tool_call></tool_call><tool_response></tool_response><arg_key></arg_key><arg_value></arg_value>B"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			messages := []api.Message{{Role: "tool", Content: tc.content}}
			raw := renderFlintPublisherJinja(t, templatePath, messages, nil)
			got, err := (&FlintRenderer{}).Render(messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got == raw {
				t.Fatal("tool result sanitizer must differ from publisher Jinja for unsafe delimiters")
			}
			const want = "<|observation|><tool_response>AB</tool_response><|assistant|>"
			if got != want {
				t.Errorf("sanitized tool result = %q, want %q", got, want)
			}
		})
	}
	t.Run("unsupported role", func(t *testing.T) {
		messages := []api.Message{{Role: "unknown", Content: "Do not lose this"}}
		if raw := renderFlintPublisherJinja(t, templatePath, messages, nil); raw != "<|assistant|>" {
			t.Fatalf("publisher handling of unknown role = %q, want only generation prompt", raw)
		}
		if _, err := (&FlintRenderer{}).Render(messages, nil, nil); err == nil {
			t.Fatal("renderer must reject a role the template would silently drop")
		}
	})
}

func flintPublisherTemplatePath(t *testing.T) string {
	t.Helper()
	templatePath := filepath.Join("testdata", "flint_chat_template.jinja")
	template, err := os.ReadFile(templatePath)
	if err != nil {
		t.Fatalf("pinned publisher template unavailable: %v", err)
	}
	const templateSHA256 = "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466"
	if got := fmt.Sprintf("%x", sha256.Sum256(template)); got != templateSHA256 {
		t.Fatalf("publisher template SHA256 = %s, want %s", got, templateSHA256)
	}
	return templatePath
}

func renderFlintPublisherJinja(t *testing.T, templatePath string, messages []api.Message, tools []api.Tool) string {
	t.Helper()
	// This is only a field mapping. Prompt policy and generation suffix remain
	// entirely in the publisher template, so the comparison is independent.
	type referenceCall struct {
		Function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	}
	type referenceMessage struct {
		Role             string          `json:"role"`
		Content          string          `json:"content"`
		ReasoningContent string          `json:"reasoning_content,omitempty"`
		ToolCalls        []referenceCall `json:"tool_calls,omitempty"`
	}
	input := struct {
		Messages []referenceMessage `json:"messages"`
		Tools    []api.Tool         `json:"tools"`
	}{Messages: make([]referenceMessage, 0, len(messages)), Tools: tools}
	for _, message := range messages {
		mapped := referenceMessage{Role: message.Role, Content: message.Content, ReasoningContent: message.Thinking}
		for _, call := range message.ToolCalls {
			var mappedCall referenceCall
			mappedCall.Function.Name = call.Function.Name
			args, err := call.Function.Arguments.MarshalJSON()
			if err != nil {
				t.Fatal(err)
			}
			mappedCall.Function.Arguments = args
			mapped.ToolCalls = append(mapped.ToolCalls, mappedCall)
		}
		input.Messages = append(input.Messages, mapped)
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	return renderFlintPublisherJinjaPayload(t, templatePath, encoded)
}

func renderFlintPublisherJinjaPayload(t *testing.T, templatePath string, encoded []byte) string {
	t.Helper()
	python := os.Getenv("FLINT_JINJA_PYTHON")
	if python == "" {
		python = "python3"
	}
	const script = `import json, sys
from transformers.utils.chat_template_utils import render_jinja_template
payload = json.load(sys.stdin)
with open(sys.argv[1], encoding="utf-8") as source:
    template = source.read()
rendered, _ = render_jinja_template([payload["messages"]], tools=payload["tools"], chat_template=template, add_generation_prompt=True)
sys.stdout.write(rendered[0])
`
	command := exec.CommandContext(t.Context(), python, "-c", script, templatePath)
	command.Stdin = bytes.NewReader(encoded)
	var output, diagnostics bytes.Buffer
	command.Stdout = &output
	command.Stderr = &diagnostics
	if err := command.Run(); err != nil {
		t.Fatalf("publisher Jinja render failed: %v\n%s", err, diagnostics.String())
	}
	return output.String()
}
