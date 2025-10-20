package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

type mockRenderer struct{}

func (m *mockRenderer) Render(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	return "mock-output", nil
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
	// Test that qwen3-coder still works
	messages := []api.Message{
		{Role: "user", Content: "Hello"},
	}

	result, err := RenderWithRenderer("qwen3-coder", messages, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result from qwen3-coder renderer")
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
