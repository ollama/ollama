package mlxrunner

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestCompletionForwardsThink(t *testing.T) {
	boolPtr := func(v bool) *bool { return &v }

	testCases := []struct {
		name  string
		think *api.ThinkValue
		want  *bool
	}{
		{name: "unset", think: nil, want: nil},
		{name: "enabled", think: &api.ThinkValue{Value: true}, want: boolPtr(true)},
		{name: "disabled", think: &api.ThinkValue{Value: false}, want: boolPtr(false)},
		{name: "level maps to enabled", think: &api.ThinkValue{Value: "high"}, want: boolPtr(true)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var got completionRequest

			rt := roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if r.URL.Path != "/completion" {
					t.Fatalf("request path = %q, want %q", r.URL.Path, "/completion")
				}

				if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
					return nil, err
				}

				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader("{\"done\":true}\n")),
					Request:    r,
				}, nil
			})

			c := &Client{
				port: 11434,
				client: &http.Client{
					Transport: rt,
				},
			}

			err := c.Completion(context.Background(), llm.CompletionRequest{
				Prompt: "hello",
				Think:  tc.think,
			}, func(llm.CompletionResponse) {})
			if err != nil {
				t.Fatalf("completion request failed: %v", err)
			}

			if got.Prompt != "hello" {
				t.Fatalf("prompt = %q, want %q", got.Prompt, "hello")
			}

			switch {
			case tc.want == nil && got.Think != nil:
				t.Fatalf("think = %v, want nil", *got.Think)
			case tc.want != nil && got.Think == nil:
				t.Fatalf("think = nil, want %v", *tc.want)
			case tc.want != nil && got.Think != nil && *tc.want != *got.Think:
				t.Fatalf("think = %v, want %v", *got.Think, *tc.want)
			}
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}
