package tools

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func TestWebToolsRequireApproval(t *testing.T) {
	if !coreagent.ToolRequiresApproval((&WebSearch{}), map[string]any{"query": "ollama"}) {
		t.Fatal("web search should require approval")
	}
	if !coreagent.ToolRequiresApproval((&WebFetch{}), map[string]any{"url": "https://ollama.com"}) {
		t.Fatal("web fetch should require approval")
	}
}

var webToolCases = []struct {
	name string
	tool coreagent.Tool
	args map[string]any
	path string
}{
	{"search", &WebSearch{}, map[string]any{"query": "ollama"}, "/api/experimental/web_search"},
	{"fetch", &WebFetch{}, map[string]any{"url": "https://ollama.com"}, "/api/experimental/web_fetch"},
}

// runWebTool executes tool against a stub server that responds to every
// request with status and body, returning the resulting error.
func runWebTool(t *testing.T, tool coreagent.Tool, args map[string]any, path string, status int, body string) error {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != path {
			t.Fatalf("path = %q, want %q", r.URL.Path, path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(ts.Close)
	t.Setenv("OLLAMA_HOST", ts.URL)
	_, err := tool.Execute(t.Context(), coreagent.ToolContext{}, args)
	return err
}

func TestWebToolsReportAuthenticationError(t *testing.T) {
	for _, tt := range webToolCases {
		t.Run(tt.name, func(t *testing.T) {
			err := runWebTool(t, tt.tool, tt.args, tt.path, http.StatusUnauthorized,
				`{"error":"unauthorized","signin_url":"https://ollama.com/signin"}`)
			if !errors.Is(err, ErrWebAuthRequired) {
				t.Fatalf("error = %v, want %v", err, ErrWebAuthRequired)
			}
		})
	}
}

func TestWebToolsPreserveNonAuthenticationErrors(t *testing.T) {
	for _, tt := range webToolCases {
		t.Run(tt.name, func(t *testing.T) {
			err := runWebTool(t, tt.tool, tt.args, tt.path, http.StatusTooManyRequests,
				`{"error":"web search quota exceeded"}`)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), "web search quota exceeded") {
				t.Fatalf("error = %q, want original error message", err)
			}
		})
	}
}

func TestWebFetchRejectsUnsupportedScheme(t *testing.T) {
	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{name: "file scheme", url: "file:///etc/passwd", wantErr: true},
		{name: "data scheme", url: "data:text/plain,secret", wantErr: true},
		{name: "ftp scheme", url: "ftp://example.com/secret", wantErr: true},
		{name: "http allowed", url: "http://example.com", wantErr: false},
		{name: "https allowed", url: "https://example.com", wantErr: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := (&WebFetch{}).Execute(t.Context(), coreagent.ToolContext{}, map[string]any{"url": tt.url})
			if tt.wantErr && err == nil {
				t.Fatal("expected unsupported scheme to be rejected")
			}
			// For allowed schemes we expect an error only from the missing
			// server/auth path, not from scheme validation. The http/https
			// cases reach the client and may fail on connection/auth; we only
			// assert that the error is NOT a scheme error.
			if !tt.wantErr && err != nil && strings.Contains(err.Error(), "unsupported URL scheme") {
				t.Fatalf("http/https rejected as unsupported: %v", err)
			}
		})
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

	result, err := (&WebFetch{}).Execute(t.Context(), coreagent.ToolContext{}, map[string]any{
		"url": "https://ollama.com",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "[tool output truncated: showing first ~") ||
		!strings.Contains(result.Content, "omitted ~7 tokens") ||
		!strings.Contains(result.Content, "Use a narrower request or search query") {
		t.Fatalf("content missing truncation marker: %q", result.Content)
	}
	if count := strings.Count(result.Content, "x"); count != maxWebFetchContentRunes {
		t.Fatalf("captured content count = %d, want %d", count, maxWebFetchContentRunes)
	}
}
