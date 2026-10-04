package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type mockRenderer struct{}

func (m *mockRenderer) Render(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	return "mock-output", nil
}

func (m *mockRenderer) LeadingBOS() string {
	return ""
}

func TestBuiltInRendererStillWorks(t *testing.T) {
	tests := []struct {
		name string
	}{
		{name: "qwen3-coder"},
		{name: "qwen3.5"},
		{name: "qwen3.8"},
		{name: "nemotron-3-nano"},
		{name: "nemotron-3.5-nano"},
	}

	messages := []api.Message{
		{Role: "user", Content: "Hello"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := RenderWithRenderer(tt.name, messages, nil, nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result == "" {
				t.Fatalf("expected non-empty result from %s renderer", tt.name)
			}
		})
	}
}

func TestLeadingBOSForRenderer(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{name: "gemma4", want: "<bos>"},
		{name: "gemma4-small", want: "<bos>"},
		{name: "gemma4-large", want: "<bos>"},
		{name: "functiongemma", want: "<bos>"},
		{name: "lfm2", want: "<|startoftext|>"},
		{name: "lfm2-thinking", want: "<|startoftext|>"},
		{name: "laguna", want: "〈|EOS|〉"},
		{name: "poolside-v1", want: "〈|EOS|〉"},
		{name: "deepseek3.1", want: "<｜begin▁of▁sentence｜>"},
		{name: "cogito", want: "<｜begin▁of▁sentence｜>"},
		{name: "qwen3-coder", want: ""},
		{name: "unknown", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := LeadingBOSForRenderer(tt.name); got != tt.want {
				t.Fatalf("LeadingBOSForRenderer(%q) = %q, want %q", tt.name, got, tt.want)
			}
		})
	}
}

func TestUnknownRendererReturnsError(t *testing.T) {
	_, err := RenderWithRenderer("nonexistent-renderer", nil, nil, nil)
	if err == nil {
		t.Error("expected error for unknown renderer")
	}
}

func (m *mockRenderer) Thinking() *model.Thinking { return nil }
