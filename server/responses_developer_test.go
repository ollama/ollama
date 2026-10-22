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

func TestDeveloperInstructionsSurviveTruncation(t *testing.T) {
	tmpl, err := template.Parse(`{{ .System }} {{ .Prompt }} {{ .Response }}`)
	if err != nil {
		t.Fatal(err)
	}
	roles, err := template.Parse(`{{ range .Messages }}{{ .Role }}:{{ .Content }} {{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	models := map[string]Model{
		"qwen3.5":             {Config: testConfigWithRenderer("qwen3.5")},
		"qwen3-coder":         {Config: testConfigWithRenderer("qwen3-coder")},
		"cogito":              {Config: testConfigWithRenderer("cogito")},
		"legacy template":     {Template: tmpl},
		"role-aware template": {Template: roles},
	}
	messages := []api.Message{
		{Role: "system", Content: "SYSTEM_POLICY"},
		{Role: "developer", Content: "DEVELOPER_POLICY"},
		{Role: "user", Content: strings.Repeat("OLD_HISTORY ", 200)},
		{Role: "assistant", Content: "Earlier answer"},
		{Role: "developer", Content: "LATER_POLICY"},
		{Role: "user", Content: "LATEST_REQUEST"},
	}
	original := slices.Clone(messages)
	for name, m := range models {
		t.Run(name, func(t *testing.T) {
			opts := api.Options{Runner: api.Runner{NumCtx: 64}}
			prompt, _, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, messages, nil, &api.ThinkValue{Value: false}, true)
			if err != nil {
				t.Fatal(err)
			}
			position := 0
			for _, marker := range []string{"SYSTEM_POLICY", "DEVELOPER_POLICY", "LATER_POLICY", "LATEST_REQUEST"} {
				index := strings.Index(prompt[position:], marker)
				if index < 0 {
					t.Fatalf("prompt lost or reordered %q: %q", marker, prompt)
				}
				position += index + len(marker)
			}
			if strings.Contains(prompt, "OLD_HISTORY") {
				t.Fatal("test did not exercise history truncation")
			}
			if name == "role-aware template" && !strings.Contains(prompt, "developer:DEVELOPER_POLICY") {
				t.Fatal("truncation rewrote the native developer role")
			}
			if !reflect.DeepEqual(messages, original) {
				t.Fatal("truncation or rendering mutated the source history")
			}
		})
	}
}

func TestResponsesDeveloperInstructionsOverrideModelSystem(t *testing.T) {
	var request openai.ResponsesRequest
	if err := json.Unmarshal([]byte(`{"model":"test","input":[
		{"role":"developer","content":"CLIENT_POLICY"},
		{"role":"user","content":"Continue"}
	]}`), &request); err != nil {
		t.Fatal(err)
	}
	chat, err := openai.FromResponsesRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Config: testConfigWithRenderer("qwen3-coder"), System: "MODEL_DEFAULT"}
	messages := chatMessagesForRequest(m, chat.Messages)
	prompt, err := renderPrompt(m, messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "CLIENT_POLICY") || strings.Contains(prompt, "MODEL_DEFAULT") {
		t.Fatalf("saved model instructions displaced the developer policy: %q", prompt)
	}
	if len(messages) != len(chat.Messages) || messages[0].Role != "developer" || chat.Messages[0].Role != "developer" {
		t.Fatal("request preparation changed the developer role or added default instructions")
	}
	if m.System != "MODEL_DEFAULT" {
		t.Fatal("request preparation changed the saved model instructions")
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
