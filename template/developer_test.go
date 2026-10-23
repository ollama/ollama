package template

import (
	"bytes"
	"io/fs"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
)

func TestTemplatesDeveloperRoleFallback(t *testing.T) {
	names, err := fs.Glob(templatesFS, "*.gotmpl")
	require.NoError(t, err)
	names = append(names, "published/llama3.2.gotmpl")
	cases := map[string][]api.Message{
		"leading": {{Role: "developer", Content: "DEVELOPER_POLICY"}, {Role: "user", Content: "Hello"}},
		"mixed history": {
			{Role: "system", Content: "SYSTEM_POLICY"},
			{Role: "developer", Content: "DEVELOPER_POLICY"},
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Earlier answer"},
			{Role: "developer", Content: "LATER_POLICY"},
			{Role: "user", Content: "Continue"},
		},
	}
	sources := map[string]string{"legacy": `{{ .System }}|{{ .Prompt }}|{{ .Response }}`}
	for _, name := range names {
		source, err := templatesFS.ReadFile(name)
		require.NoError(t, err)
		sources[name] = string(source)
	}
	for name, source := range sources {
		t.Run(name, func(t *testing.T) {
			tmpl, err := Parse(source)
			require.NoError(t, err)
			for label, messages := range cases {
				t.Run(label, func(t *testing.T) {
					original := slices.Clone(messages)
					systemMessages := slices.Clone(messages)
					for i := range systemMessages {
						if systemMessages[i].Role == "developer" {
							systemMessages[i].Role = "system"
						}
					}
					var got, want bytes.Buffer
					require.NoError(t, tmpl.Execute(&got, Values{Messages: messages}))
					require.NoError(t, tmpl.Execute(&want, Values{Messages: systemMessages}))
					require.Equal(t, want.String(), got.String(), "developer prompt must match system fallback")
					position := 0
					for _, message := range original {
						if message.Role != "system" && message.Role != "developer" {
							continue
						}
						index := strings.Index(got.String()[position:], message.Content)
						if index < 0 || strings.Count(got.String(), message.Content) != 1 {
							t.Fatalf("instruction %q was lost, duplicated, or reordered: %q", message.Content, got.String())
						}
						position += index + len(message.Content)
					}
					require.Equal(t, original, messages, "template fallback mutated the source history")
				})
			}
		})
	}
}

func TestSavedTemplateDeveloperFallbackWithCRLF(t *testing.T) {
	source, err := templatesFS.ReadFile("chatml.gotmpl")
	require.NoError(t, err)
	tmpl, err := Parse(" \r\n" + strings.ReplaceAll(string(source), "\n", "\r\n") + "\r\n ")
	require.NoError(t, err)
	var out bytes.Buffer
	require.NoError(t, tmpl.Execute(&out, Values{Messages: []api.Message{{Role: "developer", Content: "POLICY"}, {Role: "user", Content: "Hello"}}}))
	require.Contains(t, out.String(), "<|im_start|>system\r\nPOLICY<|im_end|>")
}

func TestLegacyTemplateInstructionCompatibility(t *testing.T) {
	for _, role := range []string{"system", "developer"} {
		for _, toolLoop := range []bool{false, true} {
			name := role + "/adjacent"
			messages := []api.Message{{Role: "system", Content: "FIRST"}}
			want := "FIRST\n\nSECOND|REQUEST|"
			if toolLoop {
				name = role + "/tool_loop"
				want = "SECOND|REQUEST|"
				messages = append(messages,
					api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{
						Name: "check", Arguments: api.NewToolCallFunctionArguments(),
					}}}},
					api.Message{Role: "tool", ToolCallID: "call_1", Content: "OK"},
				)
			}
			messages = append(messages, api.Message{Role: role, Content: "SECOND"}, api.Message{Role: "user", Content: "REQUEST"})
			t.Run(name, func(t *testing.T) {
				tmpl, err := Parse(`{{ .System }}|{{ .Prompt }}|{{ .Response }}`)
				require.NoError(t, err)
				original := slices.Clone(messages)
				var out bytes.Buffer
				require.NoError(t, tmpl.Execute(&out, Values{Messages: messages}))
				require.Equal(t, want, out.String(), "must preserve the existing system behavior")
				if role == "developer" {
					require.Equal(t, original, messages, "legacy fallback mutated the source history")
				}
			})
		}
	}
}

func TestCustomTemplateKeepsInstructionRolesSeparate(t *testing.T) {
	// A custom template owns its role semantics. Developer turns must not also
	// appear in System when the template renders them from Messages.
	tmpl, err := Parse(`{{ .System }}|{{ range .Messages }}{{ .Role }}:{{ .Content }};{{ end }}`)
	require.NoError(t, err)
	messages := []api.Message{
		{Role: "system", Content: "Base policy"},
		{Role: "developer", Content: "Project policy"},
		{Role: "developer", Content: "Supplemental policy"},
		{Role: "user", Content: "First request"},
		{Role: "assistant", Content: "First answer"},
		{Role: "developer", Content: "Updated policy"},
		{Role: "user", Content: "Continue"},
	}
	original := slices.Clone(messages)
	var out bytes.Buffer
	require.NoError(t, tmpl.Execute(&out, Values{Messages: messages}))
	want := "Base policy|system:Base policy;developer:Project policy\n\nSupplemental policy;user:First request;assistant:First answer;developer:Updated policy;user:Continue;"
	require.Equal(t, want, out.String())
	require.Equal(t, original, messages, "template mutated the source history")
}
