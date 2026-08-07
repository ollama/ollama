package mlxrunner

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// The MLX wire protocol has no media field, so any media reaching the client
// would be silently dropped and the model would answer from text alone.
// Completion must refuse the request instead.
func TestCompletionRejectsMedia(t *testing.T) {
	c := &Client{}

	err := c.Completion(context.Background(), llm.CompletionRequest{
		Prompt: "describe this image",
		Media:  []llm.MediaData{{Data: []byte("png-bytes"), ID: 0, Kind: llm.MediaKindImage}},
	}, func(llm.CompletionResponse) {
		t.Fatal("no completion response expected for a rejected request")
	})
	if err == nil {
		t.Fatal("expected error for media payload, got nil")
	}

	var serr api.StatusError
	if !errors.As(err, &serr) {
		t.Fatalf("expected api.StatusError, got %T: %v", err, err)
	}
	if serr.StatusCode != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, serr.StatusCode)
	}
	if !strings.Contains(serr.ErrorMessage, "does not support image or audio input") {
		t.Errorf("error message %q does not name the missing media support", serr.ErrorMessage)
	}
}
