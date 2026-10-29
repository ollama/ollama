package launch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestProcessContextWindowMatchesLatestAlias(t *testing.T) {
	got := processContextWindow("ornith", &api.ProcessResponse{
		Models: []api.ProcessModelResponse{
			{Name: "other:latest", Model: "other:latest", ContextLength: 32768},
			{Name: "ornith:latest", Model: "ornith:latest", ContextLength: 262144},
		},
	})
	if got != 262144 {
		t.Fatalf("context window = %d, want 262144", got)
	}
}

func TestLoadedContextLength(t *testing.T) {
	for _, tt := range []struct {
		name       string
		loadError  bool
		status     int
		processes  string
		wantLength int
	}{
		{"allocated context", false, 200, `{"models":[{"name":"local:latest","context_length":8192}]}`, 8192},
		{"load failure", true, 200, "", 0},
		{"lookup unsupported", false, 404, `{"error":"not found"}`, 0},
		{"lookup unavailable", false, 503, `{"error":"unavailable"}`, 0},
		{"older context metadata", false, 200, `{"models":[{"name":"local:latest"}]}`, 0},
		{"selected model absent", false, 200, `{"models":[{"name":"other:latest","context_length":32768}]}`, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/generate":
					var req api.GenerateRequest
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Model != "local" || req.Prompt != "" || len(req.Options) != 0 {
						t.Errorf("expected load-only request without option overrides: %+v, %v", req, err)
					}
					if tt.loadError {
						w.WriteHeader(http.StatusInternalServerError)
						fmt.Fprintln(w, `{"error":"model requires more system memory"}`)
						return
					}
					fmt.Fprintln(w, `{"done":true,"done_reason":"load"}`)
				case "/api/ps":
					if tt.loadError {
						t.Error("looked up context after a failed load")
					}
					w.WriteHeader(tt.status)
					fmt.Fprint(w, tt.processes)
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			length, err := loadedContextLength("local")
			if length != tt.wantLength || (err != nil) != tt.loadError {
				t.Fatalf("loadedContextLength = %d, %v; want %d, loadError=%v", length, err, tt.wantLength, tt.loadError)
			}
			if tt.loadError && !strings.Contains(err.Error(), "model requires more system memory") {
				t.Fatalf("load error cause lost: %v", err)
			}
		})
	}
}

func TestLoadedContextLengthBoundsMetadataWait(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/generate" {
			fmt.Fprintln(w, `{"done":true,"done_reason":"load"}`)
			return
		}
		select {
		case <-r.Context().Done():
		case <-time.After(10 * time.Second):
			t.Error("optional metadata request was not cancelled promptly")
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	started := time.Now()
	length, err := loadedContextLength("local")
	if length != 0 || err != nil {
		t.Fatalf("stalled metadata must allow fallback after successful load: %d, %v", length, err)
	}
	if elapsed := time.Since(started); elapsed > 10*time.Second {
		t.Fatalf("optional metadata held up launch for %v", elapsed)
	}
}
