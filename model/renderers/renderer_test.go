package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

type mockRenderer struct{}

func (m *mockRenderer) Render(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	return "mock-output", nil
}

func (m *mockRenderer) LeadingBOS() string {
	return ""
}

func TestRegisterCustomRenderer(t *testing.T) {
	// Register a custom renderer
	Register("custom-renderer", func() Renderer {
		return &mockRenderer{}
	})

	// Retrieve and use it
	result, err := RenderWithRenderer("custom-renderer", nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "mock-output" {
		t.Errorf("expected 'mock-output', got %q", result)
	}
}

func TestBuiltInRendererStillWorks(t *testing.T) {
	tests := []struct {
		name string
	}{
		{name: "qwen3-coder"},
		{name: "qwen3.5"},
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

func TestOverrideBuiltInRenderer(t *testing.T) {
	// Override the built-in renderer
	Register("qwen3-coder", func() Renderer {
		return &mockRenderer{}
	})

	// Should get the override
	result, err := RenderWithRenderer("qwen3-coder", nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "mock-output" {
		t.Errorf("expected 'mock-output' from override, got %q", result)
	}
}

func TestUnknownRendererReturnsError(t *testing.T) {
	_, err := RenderWithRenderer("nonexistent-renderer", nil, nil, nil)
	if err == nil {
		t.Error("expected error for unknown renderer")
	}
}
