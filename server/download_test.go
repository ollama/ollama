package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func BenchmarkDownloadChunkCompletion(b *testing.B) {
	data := make([]byte, 1024*1024)
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(data)
	}))
	b.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		b.Fatal(err)
	}
	downloadPath := filepath.Join(b.TempDir(), "blob")

	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		download := &blobDownload{Name: downloadPath, Digest: digest}
		part := &blobDownloadPart{Size: int64(len(data)), blobDownload: download}
		if err := download.downloadChunk(b.Context(), requestURL, io.Discard, part); err != nil {
			b.Fatal(err)
		}
	}
}

func TestDownloadChunkReturnsWhenTransferCompletes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte{0})
	}))
	t.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	download := &blobDownload{
		Name:   filepath.Join(t.TempDir(), "blob"),
		Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
	}
	part := &blobDownloadPart{Size: 1, blobDownload: download}
	ctx, cancel := context.WithTimeout(t.Context(), 250*time.Millisecond)
	defer cancel()

	if err := download.downloadChunk(ctx, requestURL, io.Discard, part); err != nil {
		t.Fatalf("downloadChunk() error = %v, want nil", err)
	}
}

func TestDownloadChunkDetectsStallBeforeFirstByte(t *testing.T) {
	requestStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(requestStarted)
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	download := &blobDownload{Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
	part := &blobDownloadPart{Size: 1, blobDownload: download}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()

	originalStallTimeout := downloadStallTimeout
	downloadStallTimeout = 50 * time.Millisecond
	t.Cleanup(func() {
		downloadStallTimeout = originalStallTimeout
	})

	started := time.Now()
	err = download.downloadChunk(ctx, requestURL, io.Discard, part)
	elapsed := time.Since(started)

	select {
	case <-requestStarted:
	default:
		t.Fatal("download request did not start")
	}
	if !errors.Is(err, errPartStalled) {
		t.Fatalf("downloadChunk() error = %v after %v, want %v", err, elapsed, errPartStalled)
	}
	if elapsed >= 5*downloadStallTimeout {
		t.Fatalf("downloadChunk() detected the stall after %v, want less than %v", elapsed, 5*downloadStallTimeout)
	}
}

// TestDownloadBlobRedownloadsCorruptedExisting reproduces
// https://github.com/ollama/ollama/issues/17520: a blob on disk whose
// content doesn't match its digest, but whose size happens to match, must
// not be trusted as a cache hit just because a same-size file exists at its
// path.
func TestDownloadBlobRedownloadsCorruptedExisting(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	goodData := []byte("the quick brown fox jumps over the lazy dog")
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(goodData))

	fp, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	// Pre-populate the blob store with a same-size but corrupted blob, as a
	// stalled write or disk error would leave behind.
	if err := os.WriteFile(fp, make([]byte, len(goodData)), 0o644); err != nil {
		t.Fatal(err)
	}

	var directRequests atomic.Int32
	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead:
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(goodData)))
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/direct/"):
			directRequests.Add(1)
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(goodData)))
			w.WriteHeader(http.StatusPartialContent)
			w.Write(goodData)
		case r.Method == http.MethodGet:
			// Resolution request: redirect to the direct blob URL, mirroring
			// a registry handing off to a CDN.
			w.Header().Set("Location", serverURL+"/direct/blob")
			w.WriteHeader(http.StatusOK)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	n := model.ParseName("test/model:latest")
	n.ProtocolScheme = "http"
	n.Host = strings.TrimPrefix(server.URL, "http://")

	cacheHit, err := downloadBlob(t.Context(), downloadOpts{
		n:       n,
		digest:  digest,
		regOpts: &registryOptions{},
		fn:      func(api.ProgressResponse) {},
	})
	if err != nil {
		t.Fatalf("downloadBlob failed: %v", err)
	}
	if cacheHit {
		t.Error("cacheHit = true for a corrupted blob; want a fresh download")
	}
	if directRequests.Load() == 0 {
		t.Error("no request made to fetch blob content; corrupted same-size blob was trusted instead of re-downloaded")
	}

	got, err := os.ReadFile(fp)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, goodData) {
		t.Error("blob on disk does not match expected content after re-download")
	}
}
