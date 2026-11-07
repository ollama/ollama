//go:build unix

package transfer

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"syscall"
	"testing"
)

// Regression for ollama/ollama#18644: a full disk must fail the blob write
// itself. Previously the write error was discarded, the blob was treated as
// downloaded, and "no space left" only appeared later while writing the
// manifest — after the whole payload had been read.
func TestDownloadNoSpace(t *testing.T) {
	if _, err := os.Stat("/dev/full"); err != nil {
		t.Skip("/dev/full not available")
	}

	payload := []byte("blob-bytes")
	sum := sha256.Sum256(payload)
	digest := fmt.Sprintf("sha256:%x", sum)

	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	}))
	t.Cleanup(server.Close)

	clientDir := t.TempDir()
	tmp := filepath.Join(clientDir, digestToPath(digest)+".tmp")
	if err := os.MkdirAll(filepath.Dir(tmp), 0o755); err != nil {
		t.Fatal(err)
	}
	// /dev/full accepts the open and returns ENOSPC on write.
	if err := os.Symlink("/dev/full", tmp); err != nil {
		t.Fatal(err)
	}

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{{Digest: digest, Size: int64(len(payload))}},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	if !errors.Is(err, syscall.ENOSPC) {
		t.Fatalf("Download error = %v, want ENOSPC", err)
	}
	// resolve + one body GET. A retried disk-full error would hit the server again.
	if got := hits.Load(); got != 2 {
		t.Fatalf("server hits = %d, want 2 (one attempt)", got)
	}
}
