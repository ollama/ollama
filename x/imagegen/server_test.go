package imagegen

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func newCompletionTestServer(handler func(*http.Request) string) *Server {
	return &Server{
		port: 11434,
		done: make(chan error, 1),
		client: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				body := handler(req)
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(body)),
					Request:    req,
				}, nil
			}),
		},
	}
}

func TestCompletionReturnsImageData(t *testing.T) {
	s := newCompletionTestServer(func(r *http.Request) string {
		if r.URL.Path != "/completion" {
			t.Fatalf("path = %q, want /completion", r.URL.Path)
		}

		var req Request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req.Prompt != "test prompt" || req.Width != 512 || req.Height != 256 || req.Steps != 7 || req.Seed != 42 {
			t.Fatalf("unexpected request: %+v", req)
		}
		if len(req.Images) != 1 || string(req.Images[0]) != "input-image" {
			t.Fatalf("images = %q, want input-image", req.Images)
		}

		return `{"step":1,"total":2}` + "\n" +
			`{"done":true,"image":"base64png"}` + "\n"
	})

	var responses []llm.CompletionResponse
	err := s.Completion(context.Background(), llm.CompletionRequest{
		Prompt: "test prompt",
		Width:  512,
		Height: 256,
		Steps:  7,
		Seed:   42,
		Images: []llm.ImageData{{Data: []byte("input-image")}},
	}, func(resp llm.CompletionResponse) {
		responses = append(responses, resp)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(responses) != 2 {
		t.Fatalf("responses = %d, want 2", len(responses))
	}
	if responses[0].Step != 1 || responses[0].TotalSteps != 2 || responses[0].Done {
		t.Fatalf("progress response = %+v", responses[0])
	}
	if !responses[1].Done || responses[1].Image != "base64png" {
		t.Fatalf("final response = %+v", responses[1])
	}
}

func TestCompletionEOFBeforeDoneReturnsError(t *testing.T) {
	s := newCompletionTestServer(func(r *http.Request) string {
		return `{"step":1,"total":2}` + "\n"
	})

	var responses []llm.CompletionResponse
	err := s.Completion(context.Background(), llm.CompletionRequest{Prompt: "test prompt"}, func(resp llm.CompletionResponse) {
		responses = append(responses, resp)
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "closed response before completion") {
		t.Fatalf("error = %v", err)
	}
	if len(responses) != 1 || responses[0].Done {
		t.Fatalf("responses = %+v, want one non-done progress response", responses)
	}
}
