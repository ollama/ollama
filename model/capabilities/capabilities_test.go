package capabilities

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func TestFromChatTemplate(t *testing.T) {
	cases := []struct {
		name         string
		chatTemplate string
		want         []model.Capability
	}{
		{
			name:         "none",
			chatTemplate: "plain chat template",
		},
		{
			name:         "tools",
			chatTemplate: "{% if tools %}{{ tools }}{% endif %}",
			want:         []model.Capability{model.CapabilityTools},
		},
		{
			name:         "tool call",
			chatTemplate: "<tool_call>{{ message }}</tool_call>",
			want:         []model.Capability{model.CapabilityTools},
		},
		{
			name:         "thinking tags",
			chatTemplate: "<think>{{ content }}</think>",
			want:         []model.Capability{model.CapabilityThinking},
		},
		{
			name:         "split thinking tags",
			chatTemplate: `{{ message.content.split("</think>")[-1] }}`,
			want:         []model.Capability{model.CapabilityThinking},
		},
		{
			name:         "split reasoning content is not thinking support",
			chatTemplate: `{{ message.reasoning_content }}{{ message.content.split("</think>")[-1] }}`,
		},
		{
			name:         "tools and thinking",
			chatTemplate: "{% if tools %}{{ tools }}{% endif %}<think>{{ content }}</think>",
			want:         []model.Capability{model.CapabilityTools, model.CapabilityThinking},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := FromChatTemplate(tt.chatTemplate); !slices.Equal(got, tt.want) {
				t.Fatalf("FromChatTemplate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestChatTemplateHasToolRoundTrip(t *testing.T) {
	cases := []struct {
		name         string
		chatTemplate string
		want         bool
	}{
		{
			name:         "tool declaration only",
			chatTemplate: "{% if tools %}{{ tools }}{% endif %}",
		},
		{
			name:         "tool calls and tool response",
			chatTemplate: "tool_calls tool_response",
			want:         true,
		},
		{
			name:         "assistant tool call and tool role",
			chatTemplate: `assistant_tool_call message.role == "tool"`,
			want:         true,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := ChatTemplateHasToolRoundTrip(tt.chatTemplate); got != tt.want {
				t.Fatalf("ChatTemplateHasToolRoundTrip() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFromGoTemplate(t *testing.T) {
	tmpl, err := template.Parse(`
{{- if .Tools }}tools{{ end }}
{{- if .Suffix }}suffix{{ end }}
{{- range .Messages }}
{{- if .Thinking }}<think>{{ .Thinking }}</think>{{ end }}
{{- end }}
`)
	if err != nil {
		t.Fatal(err)
	}

	want := []model.Capability{model.CapabilityTools, model.CapabilityInsert, model.CapabilityThinking}
	got, err := FromGoTemplate(tmpl)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got, want) {
		t.Fatalf("FromGoTemplate() = %v, want %v", got, want)
	}
}

func TestGoTemplateHasToolRoundTrip(t *testing.T) {
	tmpl, err := template.Parse(`
{{- if .Tools }}tools{{ end }}
{{- range .Messages }}
{{- if .ToolCalls }}toolcalls{{ end }}
{{- if eq .Role "tool" }}tool result{{ end }}
{{- end }}
`)
	if err != nil {
		t.Fatal(err)
	}

	if !GoTemplateHasToolRoundTrip(tmpl) {
		t.Fatal("expected GoTemplateHasToolRoundTrip to detect tool round trip")
	}
}

func TestFromParser(t *testing.T) {
	got := FromParser(parserCapabilities{
		tools:    true,
		thinking: true,
	})
	want := []model.Capability{model.CapabilityTools, model.CapabilityThinking}

	if !slices.Equal(got, want) {
		t.Fatalf("FromParser() = %v, want %v", got, want)
	}
}

type parserCapabilities struct {
	tools    bool
	thinking bool
}

func (p parserCapabilities) HasToolSupport() bool {
	return p.tools
}

func (p parserCapabilities) HasThinkingSupport() bool {
	return p.thinking
}
