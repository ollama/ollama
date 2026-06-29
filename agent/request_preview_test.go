package agent

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildChatRequestPreviewBuildsModelRequest(t *testing.T) {
	tools := api.Tools{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "read",
			Description: "read a file",
			Parameters:  api.ToolFunctionParameters{Type: "object"},
		},
	}}
	preview := BuildChatRequestPreview(RunOptions{
		Model:        "llama3.2",
		SystemPrompt: "You are Ollama.",
		Format:       "json",
		Options:      map[string]any{"temperature": 0.1},
	}, []api.Message{{Role: "user", Content: "hello"}}, tools)

	if preview.Request.Model != "llama3.2" {
		t.Fatalf("model = %q, want llama3.2", preview.Request.Model)
	}
	if got := string(preview.Request.Format); got != `"json"` {
		t.Fatalf("format = %q, want quoted json", got)
	}
	if len(preview.Request.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(preview.Request.Messages))
	}
	if preview.Request.Messages[0].Role != "system" || preview.Request.Messages[0].Content != "You are Ollama." {
		t.Fatalf("system message = %#v", preview.Request.Messages[0])
	}
	if preview.Request.Messages[1].Role != "user" || preview.Request.Messages[1].Content != "hello" {
		t.Fatalf("user message = %#v", preview.Request.Messages[1])
	}
	if len(preview.Request.Tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(preview.Request.Tools))
	}
	if preview.PromptTokens <= 0 {
		t.Fatalf("prompt tokens = %d, want positive", preview.PromptTokens)
	}
}
