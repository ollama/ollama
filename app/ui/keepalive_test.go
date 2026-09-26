//go:build windows || darwin

package ui

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
)

func TestChatKeepaliveDuringSlowPrefill(t *testing.T) {
	old := keepaliveInterval
	keepaliveInterval = 20 * time.Millisecond
	t.Cleanup(func() { keepaliveInterval = old })

	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"capabilities":["completion"]}`))
		case "/api/chat":
			// Simulate a long prompt being processed before the first token.
			time.Sleep(10 * keepaliveInterval)
			w.Header().Set("Content-Type", "application/x-ndjson")
			w.Write([]byte(`{"model":"test","message":{"role":"assistant","content":"hi"},"done":true}` + "\n"))
		default:
			w.Write([]byte(`{}`))
		}
	}))
	defer ollama.Close()
	t.Setenv("OLLAMA_HOST", ollama.URL)

	testStore := &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	defer testStore.Close()
	server := &Server{Store: testStore}

	body, _ := json.Marshal(map[string]any{"model": "test", "prompt": "hello"})
	req := httptest.NewRequest("POST", "/api/v1/chat/new", bytes.NewReader(body))
	req.SetPathValue("id", "new")
	rr := httptest.NewRecorder()

	if err := server.chat(rr, req); err != nil {
		t.Fatalf("chat() error = %v", err)
	}

	out := rr.Body.String()
	first := strings.Index(out, `"content":"hi"`)
	if first < 0 {
		t.Fatalf("response missing chat content:\n%q", out)
	}
	if !strings.Contains(out[:first], "\n\n") {
		t.Errorf("expected keepalive newlines before first token, got:\n%q", out[:first])
	}

	// Every non-blank line must still be valid JSON.
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		if !json.Valid([]byte(line)) {
			t.Errorf("invalid JSONL line: %q", line)
		}
	}
}
