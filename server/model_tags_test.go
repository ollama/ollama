package server

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestModelTagsURL(t *testing.T) {
	n := model.ParseName("ministral-3:missing")
	if got, want := modelTagsURL(n).String(), "https://ollama.com/library/ministral-3/tags"; got != want {
		t.Fatalf("modelTagsURL() = %q, want %q", got, want)
	}
}

func TestPullModelSuggestsAvailableTags(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	var manifestRequests, tagRequests int
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/library/ministral-3/manifests/missing":
			manifestRequests++
			http.NotFound(w, r)
		case "/library/ministral-3/tags":
			tagRequests++
			if got := r.Header.Get("Accept"); got != "application/json" {
				t.Errorf("Accept header = %q, want application/json", got)
			}
			fmt.Fprint(w, `{"tags":["cloud","259m","3b","8b","24b","3b-cloud"]}`)
		default:
			t.Errorf("unexpected request path %q", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer registry.Close()

	name := registry.URL + "/library/ministral-3:missing"
	err := PullModel(t.Context(), name, &registryOptions{Insecure: true}, func(_ api.ProgressResponse) {})
	if err == nil {
		t.Fatal("PullModel() returned nil, want an error")
	}
	if !strings.Contains(err.Error(), "pull model manifest: file does not exist\n\nTry one of these models:\n  ") {
		t.Fatalf("PullModel() error has unexpected layout: %q", err)
	}

	for _, suggestion := range []string{
		"ministral-3:cloud",
		"ministral-3:259m",
		"ministral-3:3b",
		"ministral-3:8b",
		"ministral-3:24b",
		"ministral-3:3b-cloud",
	} {
		if !strings.Contains(err.Error(), suggestion) {
			t.Errorf("PullModel() error %q does not contain %q", err, suggestion)
		}
	}

	if manifestRequests != 1 {
		t.Errorf("manifest requests = %d, want 1", manifestRequests)
	}
	if tagRequests != 1 {
		t.Errorf("tag requests = %d, want 1", tagRequests)
	}
}

func TestPullModelPreservesNotFoundWhenTagsUnavailable(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	var tagRequests int
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/library/unknown/manifests/latest":
			http.NotFound(w, r)
		case "/library/unknown/tags":
			tagRequests++
			http.NotFound(w, r)
		default:
			t.Errorf("unexpected request path %q", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer registry.Close()

	name := registry.URL + "/library/unknown:latest"
	err := PullModel(t.Context(), name, &registryOptions{Insecure: true}, func(_ api.ProgressResponse) {})
	if err == nil {
		t.Fatal("PullModel() returned nil, want an error")
	}
	if got, want := err.Error(), "pull model manifest: file does not exist"; got != want {
		t.Fatalf("PullModel() error = %q, want %q", got, want)
	}
	if tagRequests != 1 {
		t.Errorf("tag requests = %d, want 1", tagRequests)
	}
}
