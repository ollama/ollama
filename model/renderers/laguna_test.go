package renderers

import (
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

// lagunaToolJSON is the get_weather tool as serialized into <available_tools>,
// matching lagunaWeatherTool().
const lagunaToolJSON = `{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "City"}}}}}`

// TestLagunaRendererReferenceFlowCoverage checks the renderer against the Laguna
// chat template. Each want is byte-for-byte template output (verified by
// rendering chat_template.jinja), except that history tool-calls use the clean
// form — the template leaks Jinja indentation there.
func TestLagunaRendererReferenceFlowCoverage(t *testing.T) {
	weather := lagunaWeatherTool()
	think := func(v bool) *api.ThinkValue { return &api.ThinkValue{Value: v} }

	// system header is always emitted; with no system message the default is used
	defaultHeader := "〈|EOS|〉<system>\n\n" + lagunaDefaultSystem + "\n</system>\n"

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
		want     string
	}{
		{
			name:     "user_only_default",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			want:     defaultHeader + "<user>\nHello\n</user>\n<assistant>\n</think>",
		},
		{
			name:     "user_only_think",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			think:    think(true),
			want:     defaultHeader + "<user>\nHello\n</user>\n<assistant>\n<think>",
		},
		{
			name:     "user_only_nothink",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			think:    think(false),
			want:     defaultHeader + "<user>\nHello\n</user>\n<assistant>\n</think>",
		},
		{
			name: "first_system_is_header",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise.\n\n"},
				{Role: "user", Content: "Hi"},
			},
			want: "〈|EOS|〉<system>\n\nStay concise.\n</system>\n" +
				"<user>\nHi\n</user>\n<assistant>\n</think>",
		},
		{
			name: "additional_system",
			messages: []api.Message{
				{Role: "system", Content: "Primary."},
				{Role: "user", Content: "Hi"},
				{Role: "system", Content: "Secondary."},
			},
			want: "〈|EOS|〉<system>\n\nPrimary.\n</system>\n" +
				"<user>\nHi\n</user>\n" +
				"<system>\nSecondary.\n</system>\n" +
				"<assistant>\n</think>",
		},
		{
			name: "tools_in_header",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise."},
				{Role: "user", Content: "Weather?"},
			},
			tools: weather,
			think: think(true),
			want: "〈|EOS|〉<system>\n\nStay concise.\n\n### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n</available_tools>\n\n" +
				"Wrap your thinking in '<think>', '</think>' tags, followed by a function call. For each function call, return an unescaped XML-like object with function name and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n" +
				"<think> your thoughts here </think>\n" +
				"<tool_call>function-name\n<arg_key>argument-key</arg_key>\n<arg_value>value-of-argument-key</arg_value>\n</tool_call>" +
				"\n</system>\n" +
				"<user>\nWeather?\n</user>\n<assistant>\n<think>",
		},
		{
			name:     "tools_default",
			messages: []api.Message{{Role: "user", Content: "Weather?"}},
			tools:    weather,
			want: "〈|EOS|〉<system>\n\n" + lagunaDefaultSystem + "\n\n### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n</available_tools>\n\n" +
				"For each function call, return an unescaped XML-like object with function name and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n" +
				"<tool_call>function-name\n<arg_key>argument-key</arg_key>\n<arg_value>value-of-argument-key</arg_value>\n</tool_call>" +
				"\n</system>\n" +
				"<user>\nWeather?\n</user>\n<assistant>\n</think>",
		},
		{
			name: "assistant_history",
			messages: []api.Message{
				{Role: "user", Content: "Add these."},
				{
					Role:     "assistant",
					Content:  "\nCalling the tool.\n",
					Thinking: "Need addition.",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name: "add",
							Arguments: testArgsOrdered([]orderedArg{
								{Key: "a", Value: 2},
								{Key: "b", Value: 3},
							}),
						},
					}},
				},
				{Role: "tool", Content: "5"},
				{Role: "user", Content: "Thanks"},
			},
			think: think(true),
			want: defaultHeader +
				"<user>\nAdd these.\n</user>\n" +
				"<assistant>\n" +
				"<think>\nNeed addition.\n</think>\n" +
				"Calling the tool.\n" +
				"<tool_call>add\n" +
				"<arg_key>a</arg_key>\n<arg_value>2</arg_value>\n" +
				"<arg_key>b</arg_key>\n<arg_value>3</arg_value>\n" +
				"</tool_call>\n" +
				"</assistant>\n" +
				"<tool_response>\n5\n</tool_response>\n" +
				"<user>\nThanks\n</user>\n<assistant>\n<think>",
		},
		{
			name: "final_assistant_prefill",
			messages: []api.Message{
				{Role: "user", Content: "Complete this"},
				{Role: "assistant", Content: "Partial"},
			},
			want: defaultHeader + "<user>\nComplete this\n</user>\n<assistant>\n</think>\nPartial\n",
		},
	}

	renderer := &LagunaRenderer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderer.Render(tt.messages, tt.tools, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("renderer output mismatch vs template (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLagunaRendererMatchesLocalJinjaControlFlow(t *testing.T) {
	if os.Getenv("VERIFY_LAGUNA_JINJA2") == "" {
		t.Skip("set VERIFY_LAGUNA_JINJA2=1 to compare against the local Laguna chat_template.jinja")
	}
	python := "/Users/daniel/.codex/worktrees/7038/ollama/.venv/bin/python3"
	if _, err := os.Stat(python); err != nil {
		t.Fatalf("VERIFY_LAGUNA_JINJA2 requires %s with jinja2 installed", python)
	}

	tests := []struct {
		name     string
		messages []api.Message
		think    *api.ThinkValue
	}{
		{
			name:     "user_only",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
		},
		{
			name: "system_user",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise.\n"},
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "additional_system_and_tool_response",
			messages: []api.Message{
				{Role: "system", Content: "Primary."},
				{Role: "user", Content: "Weather?"},
				{Role: "assistant", Content: "Calling."},
				{Role: "tool", Content: "Sunny"},
				{Role: "system", Content: "Secondary."},
			},
		},
		{
			name:     "thinking_enabled",
			messages: []api.Message{{Role: "user", Content: "Think briefly."}},
			think:    &api.ThinkValue{Value: true},
		},
		{
			name:     "thinking_disabled",
			messages: []api.Message{{Role: "user", Content: "Answer directly."}},
			think:    &api.ThinkValue{Value: false},
		},
	}

	renderer := &LagunaRenderer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderer.Render(tt.messages, nil, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			for _, modelDir := range []string{
				"/Users/daniel/Models/poolside/laguna-xs-23-04-2026",
			} {
				want := renderLagunaChatTemplate(t, python, modelDir, tt.messages, tt.think)
				if diff := cmp.Diff(want, got); diff != "" {
					t.Fatalf("%s mismatch (-chat_template +renderer):\n%s", modelDir, diff)
				}
			}
		})
	}
}

func renderLagunaChatTemplate(t *testing.T, python, modelDir string, messages []api.Message, think *api.ThinkValue) string {
	t.Helper()

	type templateMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	templateMessages := make([]templateMessage, 0, len(messages))
	for _, msg := range messages {
		templateMessages = append(templateMessages, templateMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	messagesJSON, err := json.Marshal(templateMessages)
	if err != nil {
		t.Fatalf("failed to marshal messages: %v", err)
	}

	enableThinking := "False"
	if think != nil && think.Bool() {
		enableThinking = "True"
	}

	script := `
import json
import sys
from transformers import AutoTokenizer

model_dir = sys.argv[1]
messages = json.loads(sys.argv[2])
enable_thinking = sys.argv[3] == "True"
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
print(tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=enable_thinking,
), end="")
`
	cmd := exec.Command(python, "-c", script, modelDir, string(messagesJSON), enableThinking)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("chat_template render failed: %v\nstderr: %s", err, stderr.String())
	}
	return stdout.String()
}

func lagunaWeatherTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"location"},
				Properties: testPropsOrdered([]orderedProp{{
					Key: "location",
					Value: api.ToolProperty{
						Type:        api.PropertyType{"string"},
						Description: "City",
					},
				}}),
			},
		},
	}}
}
