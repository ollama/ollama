package mlxrunner

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestCompletionPropagatesFormat(t *testing.T) {
	wantFormat := json.RawMessage(`{"type":"object","required":["answer"]}`)

	var got CompletionRequest
	client := &Client{port: 1, client: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			return nil, err
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"Done":true}` + "\n")),
			Request:    r,
		}, nil
	})}}

	err := client.Completion(t.Context(), llm.CompletionRequest{
		Format: wantFormat,
	}, func(response llm.CompletionResponse) {
		if !response.Done {
			t.Error("completion response is not done")
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	if string(got.Format) != string(wantFormat) {
		t.Errorf("format = %s, want %s", got.Format, wantFormat)
	}
}

func TestCompletionRejectsRawGrammar(t *testing.T) {
	called := false
	client := &Client{client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		called = true
		return nil, errors.New("unexpected request")
	})}}

	err := client.Completion(t.Context(), llm.CompletionRequest{Grammar: `root ::= "ok"`}, func(llm.CompletionResponse) {})
	if err == nil || !strings.Contains(err.Error(), "raw grammar is not supported") {
		t.Fatalf("Completion error = %v, want unsupported raw grammar", err)
	}
	if called {
		t.Fatal("Completion sent a raw grammar request to the MLX runner")
	}
}
