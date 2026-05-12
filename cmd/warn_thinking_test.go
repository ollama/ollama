package cmd

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// Test that a warning is printed when thinking is requested but not supported.
func TestWarnMissingThinking(t *testing.T) {
	cases := []struct {
		capabilities []model.Capability
		expectWarn   bool
	}{
		{capabilities: []model.Capability{model.CapabilityThinking}, expectWarn: false},
		{capabilities: []model.Capability{}, expectWarn: true},
	}

	for _, tc := range cases {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/show" || r.Method != http.MethodPost {
				t.Fatalf("unexpected request to %s %s", r.URL.Path, r.Method)
			}
			var req api.ShowRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			resp := api.ShowResponse{Capabilities: tc.capabilities}
			if err := json.NewEncoder(w).Encode(resp); err != nil {
				t.Fatalf("encode response: %v", err)
			}
		}))
		defer srv.Close()

		t.Setenv("OLLAMA_HOST", srv.URL)
		client, err := api.ClientFromEnvironment()
		if err != nil {
			t.Fatal(err)
		}
		oldStderr := os.Stderr
		r, w, _ := os.Pipe()
		os.Stderr = w
		ensureThinkingSupport(t.Context(), client, "m")
		w.Close()
		os.Stderr = oldStderr
		out, _ := io.ReadAll(r)

		warned := strings.Contains(string(out), "warning:")
		if tc.expectWarn && !warned {
			t.Errorf("expected warning, got none")
		}
		if !tc.expectWarn && warned {
			t.Errorf("did not expect warning, got: %s", string(out))
		}
	}
}
