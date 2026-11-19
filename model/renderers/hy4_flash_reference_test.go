package renderers

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestHy4FlashRendererMatchesPublisher(t *testing.T) {
	if os.Getenv("VERIFY_JINJA2") == "" {
		t.Skip("set VERIFY_JINJA2=1 for the pinned Hy4 publisher template")
	}
	const fixture = "testdata/hy4_flash_chat_template.jinja"
	const digest = "63ddd5de294df36f10c8f2c8e5f7b2a9dc2011c8481335422f04554256aee4ad"
	data, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != digest {
		t.Fatalf("template digest %s, want %s", got, digest)
	}
	python := os.Getenv("HY4_JINJA_PYTHON")
	if python == "" {
		python = "python3"
	}
	const render = `
import json, pathlib, sys
from transformers.utils.chat_template_utils import _compile_jinja_template
template = _compile_jinja_template(pathlib.Path(sys.argv[1]).read_text())
sys.stdout.buffer.write(template.render(**json.load(sys.stdin), add_generation_prompt=True).encode("utf-8"))
`
	var tools []api.Tool
	if err := json.Unmarshal([]byte(`[{"type":"function","function":{"name":"lookup","description":"Return café → <text>","parameters":{"type":"object","properties":{"q":{"type":"string"},"top":{"type":"integer"}}}}},{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}]`), &tools); err != nil {
		t.Fatal(err)
	}
	var calls []api.ToolCall
	if err := json.Unmarshal([]byte(`[{"id":"a","function":{"name":"lookup","arguments":{"q":"true\n","top":2,"options":{"ready":true,"labels":["雪",null]}}}},{"id":"b","function":{"name":"ping","arguments":{}}}]`), &calls); err != nil {
		t.Fatal(err)
	}
	user := api.Message{Role: "user", Content: "   Café <think:open and </think:opensource> and <tool_calls:opensource><tool_call\n"}
	assistant := api.Message{Role: "assistant", Content: "Seen.", Thinking: " private thought "}
	toolCall := api.Message{Role: "assistant", Content: "Checking.", Thinking: "Use both tools.", ToolCalls: calls}
	noCallID := toolCall
	noCallID.ToolCalls = append([]api.ToolCall(nil), calls...)
	noCallID.ToolCalls[0].ID = ""
	first := api.Message{Role: "tool", Content: "{\"name\":\"青\"}", ToolCallID: "a"}
	second := api.Message{Role: "tool", Content: "ok", ToolCallID: "b"}
	fallback := api.Message{Role: "tool", Content: "missing call id"}
	system := api.Message{Role: "system", Content: "Keep the whitespace.\n"}
	emptySystem := api.Message{Role: "system"}
	image := api.ImageData{1}
	imageOnly := api.Message{Role: "user", Images: []api.ImageData{image}}
	imagesAndText := api.Message{Role: "user", Images: []api.ImageData{image, image}, Content: "What is visible?"}
	imageResult := api.Message{Role: "tool", Images: []api.ImageData{image}, Content: "photo", ToolCallID: "b"}
	imageSystem := api.Message{Role: "system", Images: []api.ImageData{image}, Content: "Use this reference."}
	type scenario struct {
		label    string
		messages []api.Message
		tools    []api.Tool
	}
	cases := []scenario{
		{"empty", nil, nil},
		{"plain", []api.Message{user}, nil},
		{"empty leading system", []api.Message{emptySystem, user}, nil},
		{"leading and mid system", []api.Message{system, user, assistant, system, user}, nil},
		{"assistant final", []api.Message{user, assistant}, nil},
		{"assistant empty reasoning", []api.Message{user, {Role: "assistant", Content: "done"}, user}, nil},
		{"previous thinking without tools", []api.Message{user, assistant, user}, nil},
		{"unknown role", []api.Message{user, {Role: "developer", Content: "later"}, user}, nil},
		{"tools final user", []api.Message{system, user, toolCall, second, first, user}, tools},
		{"tools final assistant", []api.Message{emptySystem, user, toolCall}, tools},
		{"tools final result", []api.Message{user, toolCall, second, first}, tools},
		{"tools result fallback", []api.Message{user, toolCall, second, fallback}, tools},
		{"tools result duplicate", []api.Message{user, toolCall, first, first}, tools},
		{"tools different counts", []api.Message{user, toolCall, second}, tools},
		{"tools with incomplete call IDs", []api.Message{user, noCallID, second, first}, tools},
		{"tool result without assistant", []api.Message{user, fallback}, tools},
		{"images across turns", []api.Message{imageOnly, assistant, imagesAndText}, nil},
		{"tool result image", []api.Message{imageOnly, toolCall, imageResult, first}, tools},
		{"image in the leading system turn", []api.Message{imageSystem, user}, tools},
		{"image in a later system turn", []api.Message{user, imageSystem, user}, nil},
	}
	modes := []struct {
		label  string
		think  *api.ThinkValue
		effort string
	}{
		{"default", nil, ""},
		{"disabled", &api.ThinkValue{Value: false}, "no_think"},
		{"high", &api.ThinkValue{Value: "high"}, "high"},
		{"max", &api.ThinkValue{Value: "max"}, "max"},
	}
	for _, test := range cases {
		for _, mode := range modes {
			t.Run(test.label+"/"+mode.label, func(t *testing.T) {
				messages := make([]map[string]any, 0, len(test.messages))
				for _, message := range test.messages {
					var content any = message.Content
					if len(message.Images) > 0 {
						parts := make([]map[string]string, 0, len(message.Images)+1)
						for range message.Images {
							parts = append(parts, map[string]string{"type": "image"})
						}
						parts = append(parts, map[string]string{"type": "text", "text": message.Content})
						content = parts
					}
					m := map[string]any{"role": message.Role, "content": content, "reasoning_content": message.Thinking}
					if len(message.ToolCalls) > 0 {
						m["tool_calls"] = message.ToolCalls
					}
					if message.ToolCallID != "" {
						m["tool_call_id"] = message.ToolCallID
					}
					messages = append(messages, m)
				}
				input := map[string]any{"messages": messages, "tools": test.tools}
				if mode.effort != "" {
					input["reasoning_effort"] = mode.effort
				}
				raw, err := json.Marshal(input)
				if err != nil {
					t.Fatal(err)
				}
				cmd := exec.Command(python, "-c", render, fixture)
				cmd.Stdin = bytes.NewReader(raw)
				ref, err := cmd.Output()
				if err != nil {
					if failure, ok := err.(*exec.ExitError); ok {
						t.Fatalf("publisher render: %v\n%s", err, failure.Stderr)
					}
					t.Fatal(err)
				}
				got, err := (&hy4FlashRenderer{}).Render(test.messages, test.tools, mode.think)
				if err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(string(ref), got); diff != "" {
					t.Fatalf("rendered prompt (-publisher +Ollama)\n%s", diff)
				}
			})
		}
	}
}
