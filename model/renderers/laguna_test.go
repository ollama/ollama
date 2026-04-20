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

const (
	lagunaDirectDirective = "You should respond directly without using chain-of-thought reasoning tags."
	lagunaThinkDirective  = "You should use chain-of-thought reasoning. Put your reasoning inside <think> </think> tags before your response."
)

func TestLagunaRendererReferenceFlowCoverage(t *testing.T) {
	weather := lagunaWeatherTool()

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
		want     string
	}{
		{
			name:     "user_only_thinking_default_on",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			want: "" +
				"〈|EOS|〉<system>\n" +
				lagunaThinkDirective +
				"\n</system>\n" +
				"<user>\nHello\n</user>\n" +
				"<assistant>\n",
		},
		{
			name:     "user_only_thinking_enabled",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			think:    &api.ThinkValue{Value: true},
			want: "" +
				"〈|EOS|〉<system>\n" +
				lagunaThinkDirective +
				"\n</system>\n" +
				"<user>\nHello\n</user>\n" +
				"<assistant>\n",
		},
		{
			name:     "user_only_thinking_disabled",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			think:    &api.ThinkValue{Value: false},
			want: "" +
				"〈|EOS|〉<system>\n" +
				lagunaDirectDirective +
				"\n</system>\n" +
				"<user>\nHello\n</user>\n" +
				"<assistant>\n",
		},
		{
			name: "first_system_is_header",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise.\n\n"},
				{Role: "user", Content: "Hi"},
			},
			want: "" +
				"〈|EOS|〉<system>\n" +
				lagunaThinkDirective +
				"\nStay concise." +
				"\n</system>\n" +
				"<user>\nHi\n</user>\n" +
				"<assistant>\n",
		},
		{
			name: "additional_system_message_renders_in_loop",
			messages: []api.Message{
				{Role: "system", Content: "Primary."},
				{Role: "user", Content: "Hi"},
				{Role: "system", Content: "Secondary."},
			},
			want: "" +
				"〈|EOS|〉<system>\n" +
				lagunaThinkDirective +
				"\nPrimary." +
				"\n</system>\n" +
				"<user>\nHi\n</user>\n" +
				"<system>\nSecondary.\n</system>\n" +
				"<assistant>\n",
		},
		{
			name: "tools_in_header",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise."},
				{Role: "user", Content: "Weather?"},
			},
			tools: weather,
			think: &api.ThinkValue{Value: true},
			want: "" +
				"〈|EOS|〉<system>\n" +
				lagunaThinkDirective +
				"\nStay concise." +
				"\n\n### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" +
				`{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "City"}}}}}` + "\n" +
				"</available_tools>\n\n" +
				"For each function call, return a json object with function name and arguments within '<tool_call>' and '</tool_call>' tags:\n" +
				"<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>" +
				"\n</system>\n" +
				"<user>\nWeather?\n</user>\n" +
				"<assistant>\n",
		},
		{
			name: "tools_default_thinking_on_when_unspecified",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
			},
			tools: weather,
			want: "" +
				"〈|EOS|〉<system>\n" +
				lagunaThinkDirective +
				"\n\n### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" +
				`{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "City"}}}}}` + "\n" +
				"</available_tools>\n\n" +
				"For each function call, return a json object with function name and arguments within '<tool_call>' and '</tool_call>' tags:\n" +
				"<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>" +
				"\n</system>\n" +
				"<user>\nWeather?\n</user>\n" +
				"<assistant>\n",
		},
		{
			name: "assistant_history_with_thinking_content_tool_and_response",
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
			think: &api.ThinkValue{Value: true},
			want: "" +
				"〈|EOS|〉<system>\n" +
				lagunaThinkDirective +
				"\n</system>\n" +
				"<user>\nAdd these.\n</user>\n" +
				"<assistant>\n" +
				"<think>Need addition.</think>\n" +
				"Calling the tool.\n" +
				"<tool_call>add\n" +
				"<arg_key>a</arg_key>\n<arg_value>2</arg_value>\n" +
				"<arg_key>b</arg_key>\n<arg_value>3</arg_value>\n" +
				"</tool_call>\n" +
				"</assistant>\n" +
				"<tool_response>\n5\n</tool_response>\n" +
				"<user>\nThanks\n</user>\n" +
				"<assistant>\n",
		},
		{
			name: "final_assistant_prefill_is_continued",
			messages: []api.Message{
				{Role: "user", Content: "Complete this"},
				{Role: "assistant", Content: "Partial"},
			},
			want: "" +
				"〈|EOS|〉<system>\n" +
				lagunaThinkDirective +
				"\n</system>\n" +
				"<user>\nComplete this\n</user>\n" +
				"<assistant>\nPartial\n",
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
				t.Fatalf("renderer output mismatch (-want +got):\n%s", diff)
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

	enableThinking := "True"
	if think != nil && !think.Bool() {
		enableThinking = "False"
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
