package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// newTestClient returns a Client pointed at a fake runner that streams a
// single successful completion response, plus a counter of requests received.
func newTestClient(t *testing.T) (*Client, *atomic.Int64) {
	t.Helper()

	var hits atomic.Int64
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		resp, err := json.Marshal(CompletionResponse{Content: "hi", Done: true})
		if err != nil {
			t.Error(err)
			return
		}
		w.Write(append(resp, '\n'))
	}))
	t.Cleanup(ts.Close)

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}

	return &Client{
		modelName: "test-model",
		port:      port,
		client:    ts.Client(),
	}, &hits
}

func TestCompletionRejectsFormat(t *testing.T) {
	cases := []struct {
		name   string
		format json.RawMessage
		reject bool
	}{
		{name: "unset", format: nil, reject: false},
		{name: "empty", format: json.RawMessage(``), reject: false},
		{name: "null", format: json.RawMessage(`null`), reject: false},
		{name: "empty string", format: json.RawMessage(`""`), reject: false},
		{name: "json mode", format: json.RawMessage(`"json"`), reject: true},
		{name: "schema", format: json.RawMessage(`{"type":"object"}`), reject: true},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			c, hits := newTestClient(t)

			var got []llm.CompletionResponse
			err := c.Completion(context.Background(), llm.CompletionRequest{
				Prompt: "Say hi",
				Format: tt.format,
			}, func(r llm.CompletionResponse) {
				got = append(got, r)
			})

			if tt.reject {
				var serr api.StatusError
				if !errors.As(err, &serr) {
					t.Fatalf("expected api.StatusError, got %v", err)
				}
				if serr.StatusCode != http.StatusBadRequest {
					t.Errorf("expected status %d, got %d", http.StatusBadRequest, serr.StatusCode)
				}
				if !strings.Contains(serr.ErrorMessage, "structured outputs") {
					t.Errorf("unexpected error message: %q", serr.ErrorMessage)
				}
				if hits.Load() != 0 {
					t.Errorf("request should not reach the runner, got %d requests", hits.Load())
				}
				if len(got) != 0 {
					t.Errorf("no responses should be delivered, got %d", len(got))
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if len(got) != 1 || got[0].Content != "hi" {
					t.Errorf("expected one response with content %q, got %+v", "hi", got)
				}
				if hits.Load() != 1 {
					t.Errorf("expected exactly one runner request, got %d", hits.Load())
				}
			}
		})
	}
}
