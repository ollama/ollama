package llm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

// The Embedding endpoint takes its media as bare blobs, with no id to match
// against, so a prompt that names the images with the "[img-N]" placeholders
// the chat path uses is what says which image goes where.
//
// The media markers are positional: llama-server replaces each occurrence of
// the marker with the next entry in multimodal_data. So the data has to be in
// the order the markers appear in the prompt, and the number of markers has
// to match the number of images.
//
// These drive the runner end to end against a fake llama-server, so they pin
// the request that is actually sent.

// embeddingTestRunner starts a fake llama-server that captures the embedding
// request, and returns a runner pointed at it.
func embeddingTestRunner(t *testing.T, captured *map[string]any, mediaMarker string) *llamaServerRunner {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/v1/embeddings":
			if err := json.NewDecoder(r.Body).Decode(captured); err != nil {
				t.Errorf("invalid request body: %v", err)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"data":[{"embedding":[0.1],"tokens_evaluated":3}],"usage":{"prompt_tokens":3}}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)

	port, err := strconv.Atoi(srv.URL[strings.LastIndex(srv.URL, ":")+1:])
	if err != nil {
		t.Fatalf("parse port: %v", err)
	}

	return &llamaServerRunner{
		port:        port,
		cmd:         fakeRunningCmd(),
		sem:         semaphore.NewWeighted(1),
		options:     api.Options{Runner: api.Runner{NumCtx: 2048}},
		mediaMarker: mediaMarker,
		launch:      llamaServerLaunchConfig{projectors: []string{"vision"}},
	}
}

// embeddingCaptured returns the prompt string and the media payloads from the
// captured request.
func embeddingCaptured(t *testing.T, captured map[string]any) (string, []string) {
	t.Helper()

	promptObj, ok := captured["input"].(map[string]any)
	if !ok {
		t.Fatalf("input = %T, want multimodal prompt object", captured["input"])
	}
	promptStr, _ := promptObj["prompt_string"].(string)

	var data []string
	if raw, ok := promptObj["multimodal_data"].([]any); ok {
		for _, v := range raw {
			s, _ := v.(string)
			data = append(data, s)
		}
	}

	return promptStr, data
}

// A prompt that names the first image: the marker belongs where the
// placeholder was, not after the text.
func TestLlamaServerEmbeddingMediaMarkerPlacement(t *testing.T) {
	const mediaMarker = "<__ollama_media_test__>"

	var captured map[string]any
	runner := embeddingTestRunner(t, &captured, mediaMarker)

	if _, _, err := runner.Embedding(t.Context(), "a [img-0] b", [][]byte{[]byte("one")}); err != nil {
		t.Fatalf("Embedding error: %v", err)
	}

	promptStr, data := embeddingCaptured(t, captured)

	if want := "a " + mediaMarker + " b"; promptStr != want {
		t.Errorf("prompt_string = %q, want %q", promptStr, want)
	}
	if len(data) != 1 {
		t.Fatalf("multimodal_data len = %d, want 1", len(data))
	}
	if want := base64.StdEncoding.EncodeToString([]byte("one")); data[0] != want {
		t.Errorf("multimodal_data[0] = %q, want %q", data[0], want)
	}
}

// A prompt that names no image still needs one marker per image, or
// llama-server rejects the request for a count mismatch.
func TestLlamaServerEmbeddingMediaMarkerCountMatchesImages(t *testing.T) {
	const mediaMarker = "<__ollama_media_test__>"

	var captured map[string]any
	runner := embeddingTestRunner(t, &captured, mediaMarker)

	media := [][]byte{[]byte("one"), []byte("two"), []byte("three")}
	if _, _, err := runner.Embedding(t.Context(), "some text", media); err != nil {
		t.Fatalf("Embedding error: %v", err)
	}

	promptStr, data := embeddingCaptured(t, captured)

	if got := strings.Count(promptStr, mediaMarker); got != len(media) {
		t.Errorf("marker count = %d, want %d", got, len(media))
	}
	if len(data) != len(media) {
		t.Errorf("multimodal_data len = %d, want %d", len(data), len(media))
	}
}

// The data has to be in the order the markers appear in the prompt, not in
// the order the images were given in.
func TestLlamaServerEmbeddingMediaMarkerOrderFollowsThePrompt(t *testing.T) {
	const mediaMarker = "<__ollama_media_test__>"

	var captured map[string]any
	runner := embeddingTestRunner(t, &captured, mediaMarker)

	// The prompt names the images out of order: image 1 first, then image 0.
	if _, _, err := runner.Embedding(t.Context(), "x [img-1] y [img-0]", [][]byte{[]byte("zero"), []byte("one")}); err != nil {
		t.Fatalf("Embedding error: %v", err)
	}

	promptStr, data := embeddingCaptured(t, captured)

	if want := "x " + mediaMarker + " y " + mediaMarker; promptStr != want {
		t.Errorf("prompt_string = %q, want %q", promptStr, want)
	}
	if len(data) != 2 {
		t.Fatalf("multimodal_data len = %d, want 2", len(data))
	}
	// The first marker in the prompt stands for image 1, so the data has to
	// start with image 1.
	if want := base64.StdEncoding.EncodeToString([]byte("one")); data[0] != want {
		t.Errorf("multimodal_data[0] = %q, want %q (image 1)", data[0], want)
	}
	if want := base64.StdEncoding.EncodeToString([]byte("zero")); data[1] != want {
		t.Errorf("multimodal_data[1] = %q, want %q (image 0)", data[1], want)
	}
}

// An image the prompt does not name has no marker of its own, so it goes
// first, ahead of the images that are named.
func TestLlamaServerEmbeddingUnnamedImageComesFirst(t *testing.T) {
	const mediaMarker = "<__ollama_media_test__>"

	var captured map[string]any
	runner := embeddingTestRunner(t, &captured, mediaMarker)

	// Only the first image is named, and the second one is not.
	if _, _, err := runner.Embedding(t.Context(), "a [img-0] b", [][]byte{[]byte("zero"), []byte("one")}); err != nil {
		t.Fatalf("Embedding error: %v", err)
	}

	promptStr, data := embeddingCaptured(t, captured)

	if want := mediaMarker + "a " + mediaMarker + " b"; promptStr != want {
		t.Errorf("prompt_string = %q, want %q", promptStr, want)
	}
	if len(data) != 2 {
		t.Fatalf("multimodal_data len = %d, want 2", len(data))
	}
	// The unnamed image is first, since its marker was prepended.
	if want := base64.StdEncoding.EncodeToString([]byte("one")); data[0] != want {
		t.Errorf("multimodal_data[0] = %q, want %q (the unnamed image)", data[0], want)
	}
	if want := base64.StdEncoding.EncodeToString([]byte("zero")); data[1] != want {
		t.Errorf("multimodal_data[1] = %q, want %q (image 0)", data[1], want)
	}
}
