package mlxrunner

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestClientCompletionAbortsRepeatedTokens(t *testing.T) {
	var body strings.Builder
	for range 32 {
		fmt.Fprintln(&body, `{"Content":"repeat"}`)
	}
	fmt.Fprintln(&body, `{"Done":true}`)

	c := &Client{
		client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(body.String())),
			}, nil
		})},
	}

	var responses []llm.CompletionResponse
	if err := c.Completion(context.Background(), llm.CompletionRequest{}, func(resp llm.CompletionResponse) {
		responses = append(responses, resp)
	}); err != nil {
		t.Fatal(err)
	}

	if got, want := len(responses), 31; got != want {
		t.Fatalf("received %d responses, want %d", got, want)
	}
	if responses[len(responses)-1].Done {
		t.Fatal("received final response after repeat limit")
	}
}
