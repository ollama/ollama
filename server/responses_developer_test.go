package server

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/template"
)

func TestResponsesDeveloperInstructionsSurviveTruncation(t *testing.T) {
	tmpl, err := template.Parse(`{{ .System }} {{ .Prompt }} {{ .Response }}`)
	if err != nil {
		t.Fatal(err)
	}
	roles, err := template.Parse(`{{ range .Messages }}{{ .Role }}:{{ .Content }} {{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	models := map[string]Model{
		"renderer":            {Config: testConfigWithRenderer("qwen3.5")},
		"legacy template":     {Template: tmpl},
		"role-aware template": {Template: roles},
	}
	const policy = "DEVELOPER_POLICY_7F3A"
	body, err := json.Marshal(map[string]any{"model": "test", "input": []any{
		map[string]any{"role": "developer", "content": policy},
		map[string]any{"role": "user", "content": strings.Repeat("OLD_HISTORY ", 200)},
		map[string]any{"role": "assistant", "content": "Earlier answer"},
		map[string]any{"role": "user", "content": "LATEST_REQUEST"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for name, m := range models {
		t.Run(name, func(t *testing.T) {
			var request openai.ResponsesRequest
			if err := json.Unmarshal(body, &request); err != nil {
				t.Fatal(err)
			}
			chat, err := openai.FromResponsesRequest(request)
			if err != nil {
				t.Fatal(err)
			}
			if chat.Messages[0].Role != "developer" {
				t.Fatal("API conversion rewrote the developer role")
			}
			opts := api.Options{Runner: api.Runner{NumCtx: 64}}
			prompt, _, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, chat.Messages, nil, &api.ThinkValue{Value: false}, true)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(prompt, policy) || !strings.Contains(prompt, "LATEST_REQUEST") {
				t.Fatalf("prompt lost the developer policy or latest request: %q", prompt)
			}
			if strings.Contains(prompt, "OLD_HISTORY") {
				t.Fatal("test did not exercise history truncation")
			}
			if name == "role-aware template" && !strings.Contains(prompt, "developer:"+policy) {
				t.Fatal("truncation rewrote the native developer role")
			}
		})
	}
}

func TestNativeChatTruncationPreservesInstructionRoles(t *testing.T) {
	messages := []api.Message{
		{Role: "system", Content: "SYSTEM_POLICY"},
		{Role: "developer", Content: "DEVELOPER_POLICY"},
		{Role: "user", Content: strings.Repeat("OLD_HISTORY ", 200)},
		{Role: "assistant", Content: "Old answer"},
		{Role: "developer", Content: "LATER_POLICY"},
		{Role: "user", Content: "LATEST_REQUEST"},
	}
	original := slices.Clone(messages)
	runner := &mockRunner{TemplateFn: func(_ context.Context, request llm.ChatRequest) (string, error) {
		var prompt strings.Builder
		for _, message := range request.Messages {
			prompt.WriteString(message.Role + ":" + message.Content + " ")
		}
		return prompt.String(), nil
	}}
	got, err := truncateNativeChatMessages(t.Context(), &Model{}, runner, &api.Options{Runner: api.Runner{NumCtx: 4}}, llm.ChatRequest{Messages: messages}, true)
	if err != nil {
		t.Fatal(err)
	}
	want := []api.Message{messages[0], messages[1], messages[4], messages[5]}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("truncated messages = %#v, want %#v", got, want)
	}
	if !reflect.DeepEqual(messages, original) {
		t.Fatal("truncation mutated the source history")
	}
}
