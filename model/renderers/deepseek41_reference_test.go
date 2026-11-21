package renderers

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// The verbatim template is from deepseek-ai/DeepSeek-V4.1-Flash-1017 at
// revision 08c8657f02291e11b6f6b63e0e16014f8e748518.
const deepseek41Template = "testdata/deepseek_v41_chat_template.jinja"

func TestDeepSeekV41PublisherTemplateDigest(t *testing.T) {
	data, err := os.ReadFile(deepseek41Template)
	if err != nil {
		t.Fatal(err)
	}
	const want = "d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3"
	digest := sha256.Sum256(data)
	if got := hex.EncodeToString(digest[:]); got != want {
		t.Fatalf("publisher template SHA256 = %s, want %s", got, want)
	}
}

// These cases exercise the publisher's output paths for supported Ollama chat
// roles: system placement, turn merging, assistant transitions, thinking state,
// tool schemas, calls, and results. Message fields for task, response_format,
// wo_eos, and namespace metadata have no Ollama API counterparts. The special
// latest_reminder role and arbitrary typed content blocks are outside Ollama's
// chat message contract. Image marker conversion is checked separately in
// deepseek41_test.go.
func TestDeepSeekV41PublisherJinjaParity(t *testing.T) {
	if os.Getenv("VERIFY_JINJA2") == "" {
		t.Skip("set VERIFY_JINJA2=1 for pinned publisher template parity")
	}
	requireQwen38Jinja(t)
	tool := api.Tool{Type: "function", Function: api.ToolFunction{
		Name: "calendar::find", Description: "Find events",
		Parameters: api.ToolFunctionParameters{Type: "object", Properties: api.NewToolPropertiesMap()},
	}}
	tool.Function.Parameters.Properties.Set("date", api.ToolProperty{Type: api.PropertyType{"string"}})
	secondTool := api.Tool{Type: "function", Function: api.ToolFunction{
		Name: "clock", Description: "Read time",
		Parameters: api.ToolFunctionParameters{Type: "object", Properties: api.NewToolPropertiesMap()},
	}}
	escapedTool := api.Tool{Type: "function", Function: api.ToolFunction{
		Name: "echo", Description: "It's <fine> & café",
		Parameters: api.ToolFunctionParameters{Type: "object", Properties: api.NewToolPropertiesMap()},
	}}
	args := api.NewToolCallFunctionArguments()
	args.Set("date", "tomorrow")
	args.Set("limit", 2)
	args.Set("filters", map[string]any{"private": false})
	call := api.ToolCall{ID: "call-a", Function: api.ToolCallFunction{Name: "calendar::find", Arguments: args}}
	otherArgs := api.NewToolCallFunctionArguments()
	otherArgs.Set("date", "today")
	other := api.ToolCall{ID: "call-b", Function: api.ToolCallFunction{Name: "calendar::find", Arguments: otherArgs}}
	user := api.Message{Role: "user", Content: "Find my events"}
	assistantCall := api.Message{Role: "assistant", Content: "Checking", Thinking: "I should look it up.", ToolCalls: []api.ToolCall{call, other}}
	resultA := api.Message{Role: "tool", Content: "Tomorrow: one event", ToolCallID: "call-a"}
	resultB := api.Message{Role: "tool", Content: "Today: none", ToolCallID: "call-b"}
	for _, tc := range []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
	}{
		{name: "default thinking", messages: []api.Message{user}},
		{name: "thinking true", messages: []api.Message{user}, think: &api.ThinkValue{Value: true}},
		{name: "chat false", messages: []api.Message{user}, think: &api.ThinkValue{Value: false}},
		{name: "low effort", messages: []api.Message{user}, think: &api.ThinkValue{Value: "low"}},
		{name: "medium effort", messages: []api.Message{user}, think: &api.ThinkValue{Value: "medium"}},
		{name: "high effort", messages: []api.Message{user}, think: &api.ThinkValue{Value: "high"}},
		{name: "max effort", messages: []api.Message{user}, think: &api.ThinkValue{Value: "max"}},
		{name: "empty transcript"},
		{name: "empty user content", messages: []api.Message{{Role: "user"}}},
		{name: "system only", messages: []api.Message{{Role: "system", Content: "Be brief."}}},
		{name: "system only chat", messages: []api.Message{{Role: "system", Content: "Be brief."}}, think: &api.ThinkValue{Value: false}},
		{name: "leading system", messages: []api.Message{{Role: "system", Content: "Be brief."}, user}},
		{name: "multiple systems", messages: []api.Message{{Role: "system", Content: "First"}, {Role: "system", Content: "Second"}, user}},
		{name: "midconversation system", messages: []api.Message{user, {Role: "assistant", Content: "Done."}, {Role: "system", Content: "New rule"}}},
		{name: "midconversation system chat", messages: []api.Message{user, {Role: "assistant", Content: "Done."}, {Role: "system", Content: "New rule"}}, think: &api.ThinkValue{Value: false}},
		{name: "adjacent users", messages: []api.Message{user, {Role: "user", Content: "Today too"}}},
		{name: "adjacent assistants", messages: []api.Message{user, {Role: "assistant", Content: "First"}, {Role: "assistant", Content: "Second"}}},
		{name: "assistant reasoning kept", messages: []api.Message{user, {Role: "assistant", Content: "Answer", Thinking: "Reasoning"}}},
		{name: "old assistant reasoning dropped", messages: []api.Message{user, {Role: "assistant", Content: "First", Thinking: "Old"}, {Role: "user", Content: "Again"}}},
		{name: "assistant terminal chat", messages: []api.Message{user, {Role: "assistant", Content: "Answer"}}, think: &api.ThinkValue{Value: false}},
		{name: "assistant reasoning ignored in chat", messages: []api.Message{user, {Role: "assistant", Content: "Answer", Thinking: "Hidden"}}, think: &api.ThinkValue{Value: false}},
		{name: "assistant without user", messages: []api.Message{{Role: "assistant", Content: "Answer", Thinking: "Reasoning"}}},
		{name: "tool registry after system", messages: []api.Message{{Role: "system", Content: "Be brief."}, user}, tools: []api.Tool{tool}, think: &api.ThinkValue{Value: false}},
		{name: "tool registry without messages", tools: []api.Tool{tool}},
		{name: "tool registry with system only", messages: []api.Message{{Role: "system", Content: "Be brief."}}, tools: []api.Tool{tool}},
		{name: "tool registry inserted system", messages: []api.Message{user}, tools: []api.Tool{tool}},
		{name: "multiple tool schemas", messages: []api.Message{user}, tools: []api.Tool{tool, secondTool}},
		{name: "tool schema escaping", messages: []api.Message{user}, tools: []api.Tool{escapedTool}},
		{name: "old assistant reasoning kept with tools", messages: []api.Message{user, {Role: "assistant", Content: "First", Thinking: "Old"}, {Role: "user", Content: "Again"}}, tools: []api.Tool{tool}},
		{name: "tool call without registry", messages: []api.Message{user, assistantCall}},
		{name: "empty tool arguments", messages: []api.Message{user, {Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "clock"}}}}}, tools: []api.Tool{secondTool}},
		{name: "assistant tool calls", messages: []api.Message{user, assistantCall}, tools: []api.Tool{tool}},
		{name: "assistant tool calls in chat", messages: []api.Message{user, assistantCall}, tools: []api.Tool{tool}, think: &api.ThinkValue{Value: false}},
		{name: "leading tool result", messages: []api.Message{resultA}},
		{name: "tool terminal", messages: []api.Message{user, assistantCall, resultA}, tools: []api.Tool{tool}},
		{name: "tool results reordered", messages: []api.Message{user, assistantCall, resultB, resultA}, tools: []api.Tool{tool}},
		{name: "tool result merged with user", messages: []api.Message{user, assistantCall, resultB, {Role: "user", Content: "And next week?"}}, tools: []api.Tool{tool}},
		{name: "interleaved result ordering", messages: []api.Message{user, assistantCall, resultB, {Role: "user", Content: "Also today"}, resultA}, tools: []api.Tool{tool}},
		{name: "unmatched tool result id", messages: []api.Message{user, assistantCall, {Role: "tool", Content: "Unknown", ToolCallID: "other"}, resultB}, tools: []api.Tool{tool}},
		{name: "tool loop final assistant", messages: []api.Message{user, assistantCall, resultA, {Role: "assistant", Content: "One event."}}, tools: []api.Tool{tool}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RenderWithRenderer("deepseek-v41", tc.messages, tc.tools, tc.think)
			if err != nil {
				t.Fatal(err)
			}
			want := renderDeepSeekV41Jinja(t, tc.messages, tc.tools, tc.think)
			if got != want {
				t.Fatalf("prompt differs from pinned publisher template:\ngot:  %q\nwant: %q", got, want)
			}
		})
	}
}

// The publisher template sees ordered image content blocks. Ollama supplies
// image bytes separately and uses numbered markers for the runner to replace.
func TestDeepSeekV41ImageMarkerDifference(t *testing.T) {
	if os.Getenv("VERIFY_JINJA2") == "" {
		t.Skip("set VERIFY_JINJA2=1 for pinned publisher template parity")
	}
	requireQwen38Jinja(t)
	got, err := RenderWithRenderer("deepseek-v41", []api.Message{{Role: "user", Content: "Look", Images: []api.ImageData{{1}}}}, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatal(err)
	}
	const want = "<｜begin▁of▁sentence｜><｜User｜>Look\n\n[img-0]<｜Assistant｜></think>"
	if got != want {
		t.Fatalf("Ollama image prompt = %q, want %q", got, want)
	}
	const publisherWant = "<｜begin▁of▁sentence｜><｜User｜>Look\n\n<｜deepseek_image｜><｜Assistant｜></think>"
	payload := map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Look"}, map[string]any{"type": "image"}}}},
		"tools":    []any{}, "add_generation_prompt": true, "enable_thinking": false,
	}
	if publisher := renderDeepSeekV41JinjaPayload(t, payload); publisher != publisherWant {
		t.Fatalf("publisher image prompt = %q, want %q", publisher, publisherWant)
	}
}

func renderDeepSeekV41Jinja(t *testing.T, messages []api.Message, tools []api.Tool, think *api.ThinkValue) string {
	t.Helper()
	if messages == nil {
		messages = []api.Message{}
	}
	payload := map[string]any{"messages": messages, "tools": tools, "add_generation_prompt": true}
	if think != nil && think.Value != nil {
		payload["enable_thinking"] = think.Bool()
		if think.IsString() {
			payload["reasoning_effort"] = think.Value
		}
	}
	return renderDeepSeekV41JinjaPayload(t, payload)
}

func renderDeepSeekV41JinjaPayload(t *testing.T, payload map[string]any) string {
	t.Helper()
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	path, err := filepath.Abs(deepseek41Template)
	if err != nil {
		t.Fatal(err)
	}
	const script = `
import json
import sys
from pathlib import Path
from transformers.utils.chat_template_utils import _compile_jinja_template
template = _compile_jinja_template(Path(sys.argv[1]).read_text())
print(template.render(**json.loads(sys.argv[2])), end="")
`
	cmd := qwen38PythonCommand(t, "-c", script, path, string(encoded))
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("publisher Jinja render failed: %v\n%s", err, stderr.String())
	}
	return stdout.String()
}
