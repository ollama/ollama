package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestAliasShadowingRejected(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := Server{}
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "shadowed-model",
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

	w = createRequest(t, s.CreateAliasHandler, aliasEntry{Alias: "shadowed-model", Target: "other-model"})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", w.Code)
	}
}

func TestAliasResolvesForChatRemote(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var remoteModel string
	rs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req api.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		remoteModel = req.Model

		w.Header().Set("Content-Type", "application/json")
		resp := api.ChatResponse{
			Model:      req.Model,
			Done:       true,
			DoneReason: "load",
		}
		if err := json.NewEncoder(w).Encode(&resp); err != nil {
			t.Fatal(err)
		}
	}))
	defer rs.Close()

	p, err := url.Parse(rs.URL)
	if err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_REMOTES", p.Hostname())

	s := Server{}
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "target-model",
		RemoteHost: rs.URL,
		From:       "test",
		Info: map[string]any{
			"capabilities": []string{"completion"},
		},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	w = createRequest(t, s.CreateAliasHandler, aliasEntry{Alias: "alias-model", Target: "target-model"})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	w = createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    "alias-model",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Stream:   &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp api.ChatResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if resp.Model != "alias-model" {
		t.Fatalf("expected response model to be alias-model, got %q", resp.Model)
	}

	if remoteModel != "test" {
		t.Fatalf("expected remote model to be 'test', got %q", remoteModel)
	}
}
