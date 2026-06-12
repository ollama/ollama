package tools

import "testing"

func TestWebToolsRequireApproval(t *testing.T) {
	if !NewWebSearch().RequiresApproval(map[string]any{"query": "ollama"}) {
		t.Fatal("web search should require approval")
	}
	if !NewWebFetch().RequiresApproval(map[string]any{"url": "https://ollama.com"}) {
		t.Fatal("web fetch should require approval")
	}
}
