package server

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"testing"
	"time"
)

// TestDownloadChunkStallWatchdogFiresWithoutProgress verifies that the
// chunk watchdog fires when the server sends response headers but never
// writes any body bytes. Previously the watchdog's IsZero guard caused
// it to permanently skip stall detection in this case, leaving the
// reader goroutine blocked on a dead TCP stream.
func TestDownloadChunkStallWatchdogFiresWithoutProgress(t *testing.T) {
	origStall := stallDuration
	stallDuration = 200 * time.Millisecond
	t.Cleanup(func() { stallDuration = origStall })

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1024")
		w.Header().Set("Content-Range", "bytes 0-1023/1024")
		w.WriteHeader(http.StatusPartialContent)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)

	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}

	b := &blobDownload{
		Name:   filepath.Join(t.TempDir(), "blob"),
		Digest: "sha256:deadbeef1234567890abcdef",
	}
	part := &blobDownloadPart{
		blobDownload: b,
		N:            0,
		Offset:       0,
		Size:         1024,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	start := time.Now()
	err = b.downloadChunk(ctx, u, io.Discard, part)
	elapsed := time.Since(start)

	if !errors.Is(err, errPartStalled) {
		t.Fatalf("want errPartStalled, got %v (elapsed %v)", err, elapsed)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("watchdog took too long to fire: %v (want ~stallDuration=%v)", elapsed, stallDuration)
	}
}
