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
// against, so the ordering it can honour is the order the images were given
// in. A prompt that names the images with the "[img-N]" placeholders the chat
// path uses is honoured when it is there.
//
// This drives the runner end to end against a fake llama-server, so it pins
// the request that is actually sent.
func TestLlamaServerEmbeddingMediaMarkerPlacement(t *testing.T) {
	const mediaMarker = "<__ollama_media_test__>"

	var capturedInput map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/v1/embeddings":
			if err := json.NewDecoder(r.Body).Decode(&capturedInput); err != nil {
				t.Errorf("invalid request body: %v", err)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"data":[{"embedding":[0.1],"tokens_evaluated":3}],"usage":{"prompt_tokens":3}}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	port, err := strconv.Atoi(srv.URL[strings.LastIndex(srv.URL, ":")+1:])
	if err != nil {
		t.Fatalf("parse port: %v", err)
	}

	runner := &llamaServerRunner{
		port:        port,
		cmd:         fakeRunningCmd(),
		sem:         semaphore.NewWeighted(1),
		options:     api.Options{Runner: api.Runner{NumCtx: 2048}},
		mediaMarker: mediaMarker,
		launch:      llamaServerLaunchConfig{projectors: []string{"vision"}},
	}

	// A prompt that names the first image: the marker belongs where the
	// placeholder was, not after the text.
	if _, _, err := runner.Embedding(t.Context(), "a [img-0] b", [][]byte{[]byte("one")}); err != nil {
		t.Fatalf("Embedding error: %v", err)
	}

	promptObj, ok := capturedInput["input"].(map[string]any)
	if !ok {
		t.Fatalf("input = %T, want multimodal prompt object", capturedInput["input"])
	}
	if got, want := promptObj["prompt_string"], "a "+mediaMarker+" b"; got != want {
		t.Errorf("prompt_string = %q, want %q", got, want)
	}

	data, ok := promptObj["multimodal_data"].([]any)
	if !ok {
		t.Fatalf("multimodal_data = %T, want array", promptObj["multimodal_data"])
	}
	if len(data) != 1 {
		t.Fatalf("multimodal_data len = %d, want 1", len(data))
	}
	if got, want := data[0], base64.StdEncoding.EncodeToString([]byte("one")); got != want {
		t.Errorf("multimodal_data[0] = %q, want %q", got, want)
	}
}

// A prompt that names no image still needs one marker per image, or
// llama-server rejects the request for a count mismatch.
func TestLlamaServerEmbeddingMediaMarkerCountMatchesImages(t *testing.T) {
	const mediaMarker = "<__ollama_media_test__>"

	var capturedInput map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/v1/embeddings":
			if err := json.NewDecoder(r.Body).Decode(&capturedInput); err != nil {
				t.Errorf("invalid request body: %v", err)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"data":[{"embedding":[0.1],"tokens_evaluated":3}],"usage":{"prompt_tokens":3}}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	port, err := strconv.Atoi(srv.URL[strings.LastIndex(srv.URL, ":")+1:])
	if err != nil {
		t.Fatalf("parse port: %v", err)
	}

	runner := &llamaServerRunner{
		port:        port,
		cmd:         fakeRunningCmd(),
		sem:         semaphore.NewWeighted(1),
		options:     api.Options{Runner: api.Runner{NumCtx: 2048}},
		mediaMarker: mediaMarker,
		launch:      llamaServerLaunchConfig{projectors: []string{"vision"}},
	}

	media := [][]byte{[]byte("one"), []byte("two"), []byte("three")}
	if _, _, err := runner.Embedding(t.Context(), "some text", media); err != nil {
		t.Fatalf("Embedding error: %v", err)
	}

	promptObj, ok := capturedInput["input"].(map[string]any)
	if !ok {
		t.Fatalf("input = %T, want multimodal prompt object", capturedInput["input"])
	}

	promptStr, _ := promptObj["prompt_string"].(string)
	if got := strings.Count(promptStr, mediaMarker); got != len(media) {
		t.Errorf("marker count = %d, want %d", got, len(media))
	}

	data, _ := promptObj["multimodal_data"].([]any)
	if len(data) != len(media) {
		t.Errorf("multimodal_data len = %d, want %d", len(data), len(media))
	}
}
