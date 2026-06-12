package tools

import (
	"testing"

	coreagent "github.com/ollama/ollama/agent"
)

func TestWebToolsDoNotRequireApproval(t *testing.T) {
	if coreagent.ToolRequiresApproval(NewWebSearch(), map[string]any{"query": "ollama"}) {
		t.Fatal("web search should not require approval")
	}
	if coreagent.ToolRequiresApproval(NewWebFetch(), map[string]any{"url": "https://ollama.com"}) {
		t.Fatal("web fetch should not require approval")
	}
}
