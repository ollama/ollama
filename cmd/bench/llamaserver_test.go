package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// llamaServerMock serves /props and a streamed /completion, recording each
// request body.
func llamaServerMock(t *testing.T, status int) (string, *[]llamaServerReq) {
	t.Helper()
	var reqs []llamaServerReq
	mux := http.NewServeMux()
	mux.HandleFunc("GET /props", func(w http.ResponseWriter, r *http.Request) {})
	mux.HandleFunc("POST /completion", func(w http.ResponseWriter, r *http.Request) {
		var req llamaServerReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		reqs = append(reqs, req)
		if status != http.StatusOK {
			http.Error(w, "model failed to load", status)
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", `{"content":"x","stop":false}`)
		fmt.Fprintf(w, "data: %s\n\n", `{"content":"","stop":true,"timings":{"cache_n":60,"prompt_n":40,"prompt_ms":10,"predicted_n":1,"predicted_ms":5}}`)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return strings.TrimPrefix(server.URL, "http://"), &reqs
}

func TestLlamaServerBackend(t *testing.T) {
	addr, reqs := llamaServerMock(t, http.StatusOK)
	fOpt := createTestFlagOptions()
	fOpt.runner = &addr
	backend, err := newDirectBackend(fOpt)
	if err != nil {
		t.Fatal(err)
	}
	if backend.Name() != "llama-server" {
		t.Fatalf("detected %s, want llama-server", backend.Name())
	}

	res, err := backend.Complete(context.Background(), directParams(fOpt, modePrefill, "x"))
	if err != nil {
		t.Fatal(err)
	}
	if got := (*reqs)[0].NPredict; got != 1 {
		t.Errorf("prefill-only n_predict = %d, want 1", got)
	}
	if res.promptEvalCount != 100 || res.cachedPromptCount == nil || *res.cachedPromptCount != 60 {
		t.Errorf("prompt %d cached %v, want 100 and 60", res.promptEvalCount, res.cachedPromptCount)
	}
	if res.evalCount != 1 || res.promptEvalDuration != 10*time.Millisecond {
		t.Errorf("eval %d prompt duration %v, want 1 and 10ms", res.evalCount, res.promptEvalDuration)
	}
}

func TestLlamaServerBackendReportsErrorBody(t *testing.T) {
	addr, _ := llamaServerMock(t, http.StatusInternalServerError)
	fOpt := createTestFlagOptions()
	fOpt.runner = &addr
	b := newLlamaServerBackend(fOpt)
	_, err := b.Complete(context.Background(), directParams(fOpt, modeBoth, "x"))
	if err == nil || !strings.Contains(err.Error(), "model failed to load") {
		t.Fatalf("err = %v, want the response body in the error", err)
	}
}
