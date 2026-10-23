package server

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/template"
)

func requirePromptMarkers(t *testing.T, prompt string, markers ...string) {
	t.Helper()
	position := 0
	for _, marker := range markers {
		require.Equal(t, 1, strings.Count(prompt, marker), "prompt lost or duplicated %q: %q", marker, prompt)
		index := strings.Index(prompt[position:], marker)
		require.NotEqual(t, -1, index, "prompt reordered %q: %q", marker, prompt)
		position += index + len(marker)
	}
}

func TestResponsesDeveloperInstructionsPublishedTemplate(t *testing.T) {
	// Exact llama3.2:latest template layer:
	// sha256:966de95ca8a62200913e3f8bfbf84c8494536f1b94b49166851e76644e966396.
	source, err := os.ReadFile("../template/published/llama3.2.gotmpl")
	require.NoError(t, err)
	for _, custom := range []bool{false, true} {
		name, saved := "published", string(source)
		if custom {
			name = "custom developer support"
			saved += `{{ range .Messages }}{{ if eq .Role "developer" }}<custom-developer>{{ .Content }}</custom-developer>{{ end }}{{ end }}`
		}
		t.Run(name, func(t *testing.T) {
			tmpl, err := template.Parse(saved)
			require.NoError(t, err)
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
				require.NoError(t, json.Unmarshal([]byte(body), &request))
				chat, err := openai.FromResponsesRequest(request)
				require.NoError(t, err)
				original := slices.Clone(chat.Messages)
				m := &Model{Template: tmpl}
				prompt, err := renderPrompt(m, chatMessagesForRequest(m, chat.Messages), chat.Tools, nil)
				require.NoError(t, err)
				requirePromptMarkers(t, prompt, "SYSTEM_POLICY", "PROJECT_POLICY", "LATER_POLICY")
				if role == "system" {
					systemPrompt = prompt
				} else if custom {
					require.Contains(t, prompt, "<custom-developer>PROJECT_POLICY</custom-developer>")
				} else {
					require.Equal(t, systemPrompt, prompt, "published template must use its system fallback")
				}
				require.Equal(t, role, chat.Messages[1].Role)
				require.Equal(t, original, chat.Messages, "prompt rendering mutated the Responses source history")
			}
		})
	}
}

func TestDeveloperInstructionsSurviveTruncation(t *testing.T) {
	tmpl, err := template.Parse(`{{ .System }} {{ .Prompt }} {{ .Response }}`)
	require.NoError(t, err)
	roles, err := template.Parse(`{{ range .Messages }}{{ .Role }}:{{ .Content }} {{ end }}`)
	require.NoError(t, err)
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
			require.NoError(t, err)
			requirePromptMarkers(t, prompt, "SYSTEM_POLICY", "DEVELOPER_POLICY", "LATER_POLICY", "LATEST_REQUEST")
			require.NotContains(t, prompt, "OLD_HISTORY", "test must exercise history truncation")
			if name == "role-aware template" {
				require.Contains(t, prompt, "developer:DEVELOPER_POLICY")
			}
			require.Equal(t, original, messages, "truncation or rendering mutated the source history")
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
		require.NoError(t, err)
		requirePromptMarkers(t, prompt, "SYSTEM_POLICY", "DEVELOPER_POLICY", "LATER_POLICY", "LATEST_REQUEST")
		require.NotContains(t, prompt, "OLD_HISTORY", "test must exercise history truncation")
	})
}

func TestResponsesDeveloperInstructionsPreserveModelSystem(t *testing.T) {
	var request openai.ResponsesRequest
	require.NoError(t, json.Unmarshal([]byte(`{"model":"test","input":[
		{"role":"developer","content":"CLIENT_POLICY"},
		{"role":"user","content":"Continue"}
	]}`), &request))
	chat, err := openai.FromResponsesRequest(request)
	require.NoError(t, err)
	for _, renderer := range []string{"qwen3-coder", "functiongemma"} {
		t.Run(renderer, func(t *testing.T) {
			m := &Model{Config: testConfigWithRenderer(renderer), System: "MODEL_DEFAULT"}
			messages := chatMessagesForRequest(m, chat.Messages)
			want := append([]api.Message{{Role: "system", Content: m.System}}, chat.Messages...)
			require.Equal(t, want, messages, "request preparation must preserve saved system and instruction roles")
			prompt, err := renderPrompt(m, messages, nil, nil)
			require.NoError(t, err)
			requirePromptMarkers(t, prompt, "MODEL_DEFAULT", "CLIENT_POLICY")
			require.Equal(t, "developer", chat.Messages[0].Role)
			require.Equal(t, "MODEL_DEFAULT", m.System)
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
	require.NoError(t, err)
	want := []api.Message{messages[0], messages[1], messages[4], messages[5]}
	require.Equal(t, want, got)
	require.Equal(t, original, messages, "truncation mutated the source history")
}
