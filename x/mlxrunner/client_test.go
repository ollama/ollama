package mlxrunner

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

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestCompletionForwardsMedia(t *testing.T) {
	var got CompletionRequest
	client := &Client{
		port: 1,
		client: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				if err := json.NewDecoder(req.Body).Decode(&got); err != nil {
					t.Fatal(err)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader(`{"Done":true}` + "\n")),
					Header:     make(http.Header),
				}, nil
			}),
		},
	}

	media := llm.NewMediaData(7, []byte("\x89PNG\r\n\x1a\nimage"))
	err := client.Completion(context.Background(), llm.CompletionRequest{
		Prompt: "describe [img-7]",
		Media:  []llm.MediaData{media},
	}, func(llm.CompletionResponse) {})
	if err != nil {
		t.Fatal(err)
	}

	if got.Prompt != "describe [img-7]" {
		t.Fatalf("prompt = %q", got.Prompt)
	}
	if len(got.Media) != 1 {
		t.Fatalf("media count = %d, want 1", len(got.Media))
	}
	if got.Media[0].ID != media.ID || got.Media[0].Kind != llm.MediaKindImage ||
		string(got.Media[0].Data) != string(media.Data) {
		t.Fatalf("media = %+v, want %+v", got.Media[0], media)
	}
}
