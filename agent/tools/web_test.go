package tools

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func TestWebToolsDoNotRequireApproval(t *testing.T) {
	if coreagent.ToolRequiresApproval(NewWebSearch(), map[string]any{"query": "ollama"}) {
		t.Fatal("web search should not require approval")
	}
	if coreagent.ToolRequiresApproval(NewWebFetch(), map[string]any{"url": "https://ollama.com"}) {
		t.Fatal("web fetch should not require approval")
	}
}

func TestWebFetchBoundsContentBeforeReturning(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/experimental/web_fetch" {
			t.Fatalf("path = %q, want /api/experimental/web_fetch", r.URL.Path)
		}
		var req api.WebFetchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req.URL != "https://ollama.com" {
			t.Fatalf("request URL = %q, want https://ollama.com", req.URL)
		}
		if err := json.NewEncoder(w).Encode(api.WebFetchResponse{
			Title:   "Ollama",
			Content: strings.Repeat("x", maxWebFetchContentRunes+25),
		}); err != nil {
			t.Fatal(err)
		}
	}))
	defer ts.Close()
	t.Setenv("OLLAMA_HOST", ts.URL)

	result, err := NewWebFetch().Execute(t.Context(), coreagent.ToolContext{}, map[string]any{
		"url": "https://ollama.com",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "[web_fetch content truncated: omitted 25 characters]") {
		t.Fatalf("content missing truncation marker: %q", result.Content)
	}
	if count := strings.Count(result.Content, "x"); count != maxWebFetchContentRunes {
		t.Fatalf("captured content count = %d, want %d", count, maxWebFetchContentRunes)
	}
}
