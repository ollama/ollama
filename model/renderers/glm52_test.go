package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGLM52RendererBasic(t *testing.T) {
	renderer := GLM52Renderer{}

	messages := []api.Message{
		{Role: "user", Content: "What is 2+2?"},
	}

	rendered, err := renderer.Render(messages, nil, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	if !contains(rendered, "[gMASK]<sop>") {
		t.Fatalf("expected [gMASK]<sop> prefix, got: %s", rendered)
	}
	if !contains(rendered, "<|user|>") {
		t.Fatalf("expected <|user|> tag, got: %s", rendered)
	}
	if !contains(rendered, "What is 2+2?") {
		t.Fatalf("expected user message content, got: %s", rendered)
	}
}

func TestGLM52RendererWithTools(t *testing.T) {
	renderer := GLM52Renderer{}

	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "add",
				Description: "Add two numbers",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"a": {Type: api.PropertyType{"number"}},
						"b": {Type: api.PropertyType{"number"}},
					},
				},
			},
		},
	}

	messages := []api.Message{
		{Role: "user", Content: "Add 5 and 3"},
	}

	rendered, err := renderer.Render(messages, tools, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	if !contains(rendered, "# Tools") {
		t.Fatalf("expected Tools header, got: %s", rendered)
	}
	if !contains(rendered, "<tools>") {
		t.Fatalf("expected <tools> tag, got: %s", rendered)
	}
	if !contains(rendered, "add") {
		t.Fatalf("expected tool name 'add', got: %s", rendered)
	}
}

func TestGLM52RendererThinkingEnabled(t *testing.T) {
	renderer := GLM52Renderer{}

	messages := []api.Message{
		{Role: "user", Content: "Solve this complex problem"},
	}

	thinkValue := api.ThinkValue{Type: api.ThinkValueBool, Bool: true}
	rendered, err := renderer.Render(messages, nil, &thinkValue)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	if !contains(rendered, "<think>") {
		t.Fatalf("expected <think> tag when thinking enabled, got: %s", rendered)
	}
}

func TestGLM52RendererThinkingDisabled(t *testing.T) {
	renderer := GLM52Renderer{}

	messages := []api.Message{
		{Role: "user", Content: "Answer quickly"},
	}

	thinkValue := api.ThinkValue{Type: api.ThinkValueBool, Bool: false}
	rendered, err := renderer.Render(messages, nil, &thinkValue)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	if contains(rendered, "<think>") {
		t.Fatalf("unexpected <think> tag when thinking disabled, got: %s", rendered)
	}
}

func TestGLM52RendererMultiTurn(t *testing.T) {
	renderer := GLM52Renderer{}

	messages := []api.Message{
		{Role: "user", Content: "First question?"},
		{Role: "assistant", Content: "First answer"},
		{Role: "user", Content: "Follow-up question?"},
	}

	rendered, err := renderer.Render(messages, nil, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	if !contains(rendered, "First question?") {
		t.Fatalf("expected first user message, got: %s", rendered)
	}
	if !contains(rendered, "First answer") {
		t.Fatalf("expected assistant message, got: %s", rendered)
	}
	if !contains(rendered, "Follow-up question?") {
		t.Fatalf("expected second user message, got: %s", rendered)
	}
}

func TestGLM52RendererComplexPromptWithLongContext(t *testing.T) {
	renderer := GLM52Renderer{}

	// Simulate a complex prompt with extended content
	longContent := "This is a long prompt. "
	for i := 0; i < 100; i++ {
		longContent += "This is additional context to simulate a longer prompt. "
	}

	messages := []api.Message{
		{Role: "user", Content: longContent},
	}

	rendered, err := renderer.Render(messages, nil, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	// Verify the long content is preserved
	if !contains(rendered, "This is a long prompt") {
		t.Fatalf("expected long prompt to be preserved, got: %s", rendered[:100])
	}
}

func TestGLM52RendererEmptyMessages(t *testing.T) {
	renderer := GLM52Renderer{}

	messages := []api.Message{}

	rendered, err := renderer.Render(messages, nil, nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}

	// Should still have the prefix and thinking tag
	if !contains(rendered, "[gMASK]<sop>") {
		t.Fatalf("expected prefix even with empty messages, got: %s", rendered)
	}
	if !contains(rendered, "<think>") {
		t.Fatalf("expected thinking tag, got: %s", rendered)
	}
}

func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
