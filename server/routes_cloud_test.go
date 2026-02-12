package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

func TestStatusHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := Server{}
	w := createRequest(t, s.StatusHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp api.CloudStatusResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if !resp.Cloud.Disabled {
		t.Fatalf("expected cloud.disabled true, got false")
	}
	if resp.Cloud.Source != "env" {
		t.Fatalf("expected cloud.source env, got %q", resp.Cloud.Source)
	}
}

func TestCloudDisabledBlocksRemoteOperations(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := Server{}

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "test-cloud",
		RemoteHost: "example.com",
		From:       "test",
		Info: map[string]any{
			"capabilities": []string{"completion"},
		},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("chat remote blocked", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model:    "test-cloud",
			Messages: []api.Message{{Role: "user", Content: "hi"}},
		})
		if w.Code != http.StatusForbidden {
			t.Fatalf("expected status 403, got %d", w.Code)
		}
		if got := w.Body.String(); got != `{"error":"ollama cloud is disabled: remote inference is unavailable"}` {
			t.Fatalf("unexpected response: %s", got)
		}
	})

	t.Run("generate remote blocked", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-cloud",
			Prompt: "hi",
		})
		if w.Code != http.StatusForbidden {
			t.Fatalf("expected status 403, got %d", w.Code)
		}
		if got := w.Body.String(); got != `{"error":"ollama cloud is disabled: remote inference is unavailable"}` {
			t.Fatalf("unexpected response: %s", got)
		}
	})

	t.Run("show remote blocked", func(t *testing.T) {
		w := createRequest(t, s.ShowHandler, api.ShowRequest{
			Model: "test-cloud",
		})
		if w.Code != http.StatusForbidden {
			t.Fatalf("expected status 403, got %d", w.Code)
		}
		if got := w.Body.String(); got != `{"error":"ollama cloud is disabled: remote model details are unavailable"}` {
			t.Fatalf("unexpected response: %s", got)
		}
	})
}
