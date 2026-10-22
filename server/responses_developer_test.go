package server

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/template"
)

func TestResponsesDeveloperInstructionsPublishedTemplate(t *testing.T) {
	// Exact llama3.2:latest template layer:
	// sha256:966de95ca8a62200913e3f8bfbf84c8494536f1b94b49166851e76644e966396.
	source, err := os.ReadFile("../template/published/llama3.2.gotmpl")
	if err != nil {
		t.Fatal(err)
	}
	for _, custom := range []bool{false, true} {
		name, saved := "published", string(source)
		if custom {
			name = "custom developer support"
			saved += `{{ range .Messages }}{{ if eq .Role "developer" }}<custom-developer>{{ .Content }}</custom-developer>{{ end }}{{ end }}`
		}
		t.Run(name, func(t *testing.T) {
			tmpl, err := template.Parse(saved)
			if err != nil {
				t.Fatal(err)
			}
			var systemPrompt string
			for _, role := range []string{"system", "developer"} {
				body := `{"model":"llama3.2","instructions":"SYSTEM_POLICY","input":[
                    {"role":"` + role + `","content":"PROJECT_POLICY"},
                    {"role":"user","content":"Hello"},
                    {"role":"assistant","content":"Hello."},
                    {"role":"` + role + `","content":"LATER_POLICY"},
                    {"role":"user","content":"Continue"}
                ]}`
				var request openai.ResponsesRequest
				if err := json.Unmarshal([]byte(body), &request); err != nil {
					t.Fatal(err)
				}
				chat, err := openai.FromResponsesRequest(request)
				if err != nil {
					t.Fatal(err)
				}
				original := slices.Clone(chat.Messages)
				m := &Model{Template: tmpl}
				prompt, err := renderPrompt(m, chatMessagesForRequest(m, chat.Messages), chat.Tools, nil)
				if err != nil {
					t.Fatal(err)
				}
				position := 0
				for _, policy := range []string{"SYSTEM_POLICY", "PROJECT_POLICY", "LATER_POLICY"} {
					index := strings.Index(prompt[position:], policy)
					if index < 0 || strings.Count(prompt, policy) != 1 {
						t.Fatalf("%s lost, duplicated, or reordered %q: %q", role, policy, prompt)
					}
					position += index + len(policy)
				}
				if role == "system" {
					systemPrompt = prompt
				} else if custom {
					if !strings.Contains(prompt, "<custom-developer>PROJECT_POLICY</custom-developer>") {
						t.Fatal("custom template lost its developer role semantics")
					}
				} else if prompt != systemPrompt {
					t.Fatal("published template did not use its system fallback")
				}
				if chat.Messages[1].Role != role || !reflect.DeepEqual(chat.Messages, original) {
					t.Fatal("prompt rendering mutated the Responses source history")
				}
			}
		})
	}
}

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
				if index < 0 || strings.Count(prompt, marker) != 1 {
					t.Fatalf("prompt lost, duplicated, or reordered %q: %q", marker, prompt)
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
	t.Run("adjacent developer messages", func(t *testing.T) {
		messages := []api.Message{
			{Role: "system", Content: "SYSTEM_POLICY"},
			{Role: "developer", Content: "DEVELOPER_POLICY"},
			{Role: "developer", Content: "LATER_POLICY"},
			{Role: "user", Content: strings.Repeat("OLD_HISTORY ", 200)},
			{Role: "user", Content: "LATEST_REQUEST"},
		}
		prompt, _, err := chatPrompt(t.Context(), &Model{Template: roles}, mockRunner{}.Tokenize,
			&api.Options{Runner: api.Runner{NumCtx: 4}}, messages, nil, nil, true)
		if err != nil {
			t.Fatal(err)
		}
		for _, marker := range []string{"SYSTEM_POLICY", "DEVELOPER_POLICY", "LATER_POLICY", "LATEST_REQUEST"} {
			if strings.Count(prompt, marker) != 1 {
				t.Fatalf("prompt must contain %q once: %q", marker, prompt)
			}
		}
		if strings.Contains(prompt, "OLD_HISTORY") {
			t.Fatal("test did not exercise history truncation")
		}
	})
}

func TestResponsesDeveloperInstructionsPreserveModelSystem(t *testing.T) {
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
	for _, renderer := range []string{"qwen3-coder", "functiongemma"} {
		t.Run(renderer, func(t *testing.T) {
			m := &Model{Config: testConfigWithRenderer(renderer), System: "MODEL_DEFAULT"}
			messages := chatMessagesForRequest(m, chat.Messages)
			want := append([]api.Message{{Role: "system", Content: m.System}}, chat.Messages...)
			if !reflect.DeepEqual(messages, want) {
				t.Fatalf("request preparation lost the saved system or changed instruction roles: %#v", messages)
			}
			prompt, err := renderPrompt(m, messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if system, developer := strings.Index(prompt, "MODEL_DEFAULT"), strings.Index(prompt, "CLIENT_POLICY"); system < 0 || developer <= system {
				t.Fatalf("prompt lost or reordered the saved system and developer policy: %q", prompt)
			}
			if chat.Messages[0].Role != "developer" || m.System != "MODEL_DEFAULT" {
				t.Fatal("request preparation changed the source role or saved model instructions")
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
