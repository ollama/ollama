package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestImageDownloaderDownloadImageHTTPAllowed(t *testing.T) {
	t.Setenv("OLLAMA_IMAGE_URL_ALLOWED_HOSTS", "")
	t.Setenv("OLLAMA_IMAGE_URL_CACHE_DIR", t.TempDir())
	t.Setenv("OLLAMA_IMAGE_URL_TIMEOUT", "5")
	t.Setenv("OLLAMA_IMAGE_URL_MAX_SIZE", "1048576")

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	}))
	defer ts.Close()

	d := NewImageDownloader()
	data, err := d.DownloadImage(context.Background(), api.ImageURL{
		URL:       ts.URL + "/img.png",
		AllowHTTP: true,
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("expected non-empty image data")
	}
}

func TestImageDownloaderDownloadImageHTTPDisallowed(t *testing.T) {
	t.Setenv("OLLAMA_IMAGE_URL_ALLOWED_HOSTS", "")
	t.Setenv("OLLAMA_IMAGE_URL_CACHE_DIR", t.TempDir())
	t.Setenv("OLLAMA_IMAGE_URL_TIMEOUT", "5")
	t.Setenv("OLLAMA_IMAGE_URL_MAX_SIZE", "1048576")

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	}))
	defer ts.Close()

	d := NewImageDownloader()
	_, err := d.DownloadImage(context.Background(), api.ImageURL{
		URL:       ts.URL + "/img.png",
		AllowHTTP: false,
	})
	if err == nil {
		t.Fatalf("expected error for disallowed http URL")
	}
}
