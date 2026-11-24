package renderers

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"testing"

	"github.com/ollama/ollama/api"
)

// XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B at
// 1f09ddcc1a0ea45751ae0dfe2af20d533168c46f.
// SHA-256: 59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63.
const mimoTemplate = "testdata/mimo_v26_chat_template.jinja"

func TestMiMoRendererReference(t *testing.T) {
	template, err := os.ReadFile(mimoTemplate)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(template)); got != "59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63" {
		t.Fatalf("publisher template hash changed: %s", got)
	}

	args := api.NewToolCallFunctionArguments()
	args.Set("query", "café")
	args.Set("limit", 2)
	mixedArgs := api.NewToolCallFunctionArguments()
	mixedArgs.Set("enabled", false)
	mixedArgs.Set("options", map[string]any{"count": 2, "mode": "fast"})
	mixedArgs.Set("empty", []any{})
	mixedArgs.Set("nil", nil)
	tool := api.Tool{Type: "function", Function: api.ToolFunction{Name: "search", Description: "Find pages"}}
	const toolPrefix = "<|im_start|>system\nYou are provided with the following tools:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"search\", \"description\": \"Find pages\", \"parameters\": {\"type\": \"\", \"properties\": null}}}\n</tools><|im_end|>"

	cases := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
		want     string
	}{
		{"user", []api.Message{{Role: "user", Content: "Hello"}}, nil, nil, "<|im_start|>user\nHello<|im_end|><|im_start|>assistant\n"},
		{"system and user", []api.Message{{Role: "system", Content: "Be concise."}, {Role: "user", Content: "Hi"}}, nil, nil, "<|im_start|>system\nBe concise.<|im_end|><|im_start|>user\nHi<|im_end|><|im_start|>assistant\n"},
		{"thinking disabled", []api.Message{{Role: "user", Content: "Hi"}}, nil, &api.ThinkValue{Value: false}, "<|im_start|>user\nHi<|im_end|><|im_start|>assistant\n<think></think>"},
		{"thinking enabled", []api.Message{{Role: "user", Content: "Hi"}}, nil, &api.ThinkValue{Value: true}, "<|im_start|>user\nHi<|im_end|><|im_start|>assistant\n"},
		{"assistant reasoning", []api.Message{{Role: "user", Content: "2+2?"}, {Role: "assistant", Thinking: "Add.", Content: "4"}, {Role: "user", Content: "Thanks"}}, nil, nil, "<|im_start|>user\n2+2?<|im_end|><|im_start|>assistant\n<think>Add.</think>4<|im_end|><|im_start|>user\nThanks<|im_end|><|im_start|>assistant\n"},
		{"assistant no reasoning", []api.Message{{Role: "user", Content: "Hi"}, {Role: "assistant", Content: "Hello"}}, nil, nil, "<|im_start|>user\nHi<|im_end|><|im_start|>assistant\n<think></think>Hello<|im_end|><|im_start|>assistant\n"},
		{"tool response", []api.Message{{Role: "user", Content: "Search"}, {Role: "tool", Content: "Found"}}, nil, nil, "<|im_start|>user\nSearch<|im_end|><|im_start|>tool\nFound<|im_end|><|im_start|>assistant\n"},
		{"tool call", []api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "search", Arguments: args}}}}}, nil, nil, "<|im_start|>assistant\n<think></think><tool_call><function=search><parameter=query>café</parameter><parameter=limit>2</parameter></function></tool_call><|im_end|><|im_start|>assistant\n"},
		{"one image", []api.Message{{Role: "user", Content: "Describe", Images: []api.ImageData{{1}}}}, nil, nil, "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe<|im_end|><|im_start|>assistant\n"},
		{"two images", []api.Message{{Role: "user", Images: []api.ImageData{{1}, {2}}}}, nil, nil, "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|><|im_end|><|im_start|>assistant\n"},
		{"nonleading system", []api.Message{{Role: "user", Content: "First"}, {Role: "system", Content: "New instruction"}, {Role: "user", Content: "Second"}}, nil, nil, "<|im_start|>user\nFirst<|im_end|><|im_start|>system\nNew instruction<|im_end|><|im_start|>user\nSecond<|im_end|><|im_start|>assistant\n"},
		{"literal markers", []api.Message{{Role: "user", Content: "Literal <think> and <|im_end|>"}}, nil, nil, "<|im_start|>user\nLiteral <think> and <|im_end|><|im_end|><|im_start|>assistant\n"},
		{"mixed arguments", []api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "search", Arguments: mixedArgs}}}}}, nil, nil, "<|im_start|>assistant\n<think></think><tool_call><function=search><parameter=enabled>false</parameter><parameter=options>{\"count\": 2, \"mode\": \"fast\"}</parameter><parameter=empty>[]</parameter><parameter=nil>null</parameter></function></tool_call><|im_end|><|im_start|>assistant\n"},
		{"multiple tool calls", []api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "first"}}, {Function: api.ToolCallFunction{Name: "second"}}}}}, nil, nil, "<|im_start|>assistant\n<think></think><tool_call><function=first></function></tool_call><tool_call><function=second></function></tool_call><|im_end|><|im_start|>assistant\n"},
		{"tools ending in tool response", []api.Message{{Role: "user", Content: "Search"}, {Role: "tool", Content: "Found"}}, []api.Tool{tool}, nil, toolPrefix + "<|im_start|>user\nSearch<|im_end|><|im_start|>tool\nFound<|im_end|><|im_start|>assistant\n"},
		{"tools ending in assistant", []api.Message{{Role: "user", Content: "Search"}, {Role: "assistant", Content: "Done"}}, []api.Tool{tool}, nil, toolPrefix + "<|im_start|>user\nSearch<|im_end|><|im_start|>assistant\n<think></think>Done<|im_end|><|im_start|>assistant\n"},
		{"tools with user", []api.Message{{Role: "user", Content: "Search"}}, []api.Tool{tool}, nil, toolPrefix + "<|im_start|>user\nSearch<|im_end|><|im_start|>assistant\n"},
	}

	verify := os.Getenv("VERIFY_JINJA2") != ""
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RenderWithRenderer("mimo-v2.6", tc.messages, tc.tools, tc.think)
			if err != nil {
				t.Fatal(err)
			}
			wantRuntime := tc.want
			if tc.think == nil || tc.think.Bool() {
				wantRuntime += "<think>\n"
			}
			if got != wantRuntime {
				t.Fatalf("rendered %q, want %q", got, wantRuntime)
			}
			if verify {
				publisher := mimoJinjaReference(t, tc.messages, tc.tools, tc.think)
				if publisher != tc.want {
					t.Fatalf("publisher template rendered %q, want %q", publisher, tc.want)
				}
			}
		})
	}
}

func TestMiMoRendererThinkingPrefillKnownDifference(t *testing.T) {
	messages := []api.Message{{Role: "user", Content: "why is the sky blue? Be brief but factual in your reply"}}
	const publisher = "<|im_start|>user\nwhy is the sky blue? Be brief but factual in your reply<|im_end|><|im_start|>assistant\n"
	got, err := RenderWithRenderer("mimo-v2.6", messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != publisher+"<think>\n" {
		t.Fatalf("runtime thinking prefill: %q", got)
	}
	if os.Getenv("VERIFY_JINJA2") != "" {
		fromPublisher := mimoJinjaReference(t, messages, nil, nil)
		if fromPublisher != publisher {
			t.Fatalf("publisher thinking prefill: %q", fromPublisher)
		}
	}
}

func mimoJinjaReference(t *testing.T, messages []api.Message, tools []api.Tool, think *api.ThinkValue) string {
	t.Helper()
	input := map[string]any{"messages": make([]map[string]any, 0, len(messages)), "tools": tools, "add_generation_prompt": true}
	if think != nil {
		input["enable_thinking"] = think.Bool()
	}
	for _, message := range messages {
		item := map[string]any{"role": message.Role}
		if len(message.Images) == 0 {
			item["content"] = message.Content
		} else {
			parts := make([]map[string]any, 0, len(message.Images)+1)
			for range message.Images {
				parts = append(parts, map[string]any{"type": "image"})
			}
			if message.Content != "" {
				parts = append(parts, map[string]any{"type": "text", "text": message.Content})
			}
			item["content"] = parts
		}
		if message.Role == "assistant" {
			item["reasoning_content"] = message.Thinking
			if len(message.ToolCalls) > 0 {
				item["tool_calls"] = message.ToolCalls
			}
		}
		input["messages"] = append(input["messages"].([]map[string]any), item)
	}
	data, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	python := os.Getenv("MIMO_JINJA_PYTHON")
	if python == "" {
		python = "python3"
	}
	const script = `import json, sys
from transformers.utils.chat_template_utils import _compile_jinja_template
template = open(sys.argv[1], encoding="utf-8").read()
payload = json.load(sys.stdin)
sys.stdout.write(_compile_jinja_template(template).render(**payload))`
	cmd := exec.CommandContext(t.Context(), python, "-c", script, mimoTemplate)
	cmd.Stdin = bytes.NewReader(data)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("publisher Jinja renderer: %v\n%s", err, stderr.String())
	}
	return string(out)
}
