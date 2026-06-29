package agent

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildChatRequestBuildsModelRequest(t *testing.T) {
	tools := api.Tools{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "read",
			Description: "read a file",
			Parameters:  api.ToolFunctionParameters{Type: "object"},
		},
	}}
	opts := RunOptions{
		Model:        "llama3.2",
		SystemPrompt: "You are Ollama.",
		Format:       "json",
		Options:      map[string]any{"temperature": 0.1},
	}
	messages := []api.Message{{Role: "user", Content: "hello"}}
	req := buildChatRequest(opts, messages, tools)

	if req.Model != "llama3.2" {
		t.Fatalf("model = %q, want llama3.2", req.Model)
	}
	if got := string(req.Format); got != `"json"` {
		t.Fatalf("format = %q, want quoted json", got)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(req.Messages))
	}
	if req.Messages[0].Role != "system" || req.Messages[0].Content != "You are Ollama." {
		t.Fatalf("system message = %#v", req.Messages[0])
	}
	if req.Messages[1].Role != "user" || req.Messages[1].Content != "hello" {
		t.Fatalf("user message = %#v", req.Messages[1])
	}
	if len(req.Tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(req.Tools))
	}
	if tokens := estimateChatRequestTokens(opts, messages, tools); tokens <= 0 {
		t.Fatalf("prompt tokens = %d, want positive", tokens)
	}
}
