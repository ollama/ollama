package renderers

import (
	"strings"
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

func TestLFM2RendererUsesGlobalImageTagSetting(t *testing.T) {
	orig := RenderImgTags
	t.Cleanup(func() {
		RenderImgTags = orig
	})

	msgs := []api.Message{
		{Role: "user", Content: "Describe", Images: []api.ImageData{api.ImageData("img")}},
	}

	RenderImgTags = true
	withImgTag, err := RenderWithRenderer("lfm2.5", msgs, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(withImgTag, "[img]Describe") {
		t.Fatalf("expected [img] placeholder, got: %q", withImgTag)
	}

	RenderImgTags = false
	withTemplateTag, err := RenderWithRenderer("lfm2.5", msgs, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(withTemplateTag, "<image>Describe") {
		t.Fatalf("expected <image> placeholder, got: %q", withTemplateTag)
	}
}
