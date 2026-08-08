package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestCompletionForwardsFormat(t *testing.T) {
	var got CompletionRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode wire request: %v", err)
		}
		json.NewEncoder(w).Encode(CompletionResponse{Done: true}) //nolint:errcheck
	}))
	defer srv.Close()

	c := &Client{
		port:   srv.Listener.Addr().(*net.TCPAddr).Port,
		client: http.DefaultClient,
		status: llm.NewStatusWriter(io.Discard),
	}
	err := c.Completion(context.Background(), llm.CompletionRequest{
		Prompt:  "p",
		Format:  json.RawMessage(`{"type":"object"}`),
		Options: &api.Options{},
	}, func(llm.CompletionResponse) {})
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if string(got.Format) != `{"type":"object"}` {
		t.Errorf("wire Format = %q, want the request's format forwarded", got.Format)
	}
}

func TestCompletionRejectsRawGrammar(t *testing.T) {
	c := &Client{client: http.DefaultClient, status: llm.NewStatusWriter(io.Discard)}
	err := c.Completion(context.Background(), llm.CompletionRequest{
		Prompt:  "p",
		Grammar: `root ::= "x"`,
		Options: &api.Options{},
	}, func(llm.CompletionResponse) {})
	var se api.StatusError
	if !errors.As(err, &se) {
		t.Fatalf("Completion with Grammar: err = %v, want api.StatusError", err)
	}
	if se.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", se.StatusCode)
	}
}

func TestRequestCompileFormat(t *testing.T) {
	for _, c := range []string{"", "null"} {
		req := &Request{CompletionRequest: CompletionRequest{Format: json.RawMessage(c)}}
		if err := req.compileFormat(); err != nil {
			t.Errorf("compileFormat(%q): %v", c, err)
		}
		if req.Constraint != nil {
			t.Errorf("compileFormat(%q): unexpected constraint", c)
		}
	}

	req := &Request{CompletionRequest: CompletionRequest{Format: json.RawMessage(`"json"`)}}
	if err := req.compileFormat(); err != nil {
		t.Fatalf("compileFormat(json): %v", err)
	}
	if req.Constraint == nil {
		t.Fatal("compileFormat(json): no constraint")
	}

	req = &Request{CompletionRequest: CompletionRequest{Format: json.RawMessage(`"yaml"`)}}
	if err := req.compileFormat(); err == nil {
		t.Fatal("compileFormat(yaml): expected error")
	}
}
