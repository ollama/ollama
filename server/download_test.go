package server

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
)

func TestVerifyBlobFile(t *testing.T) {
	dir := t.TempDir()

	content := []byte("hello world blob content for testing")
	h := sha256.Sum256(content)
	expectedDigest := fmt.Sprintf("sha256:%x", h)

	t.Run("matching digest", func(t *testing.T) {
		path := filepath.Join(dir, "good-blob")
		if err := os.WriteFile(path, content, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := verifyBlobFile(path, expectedDigest); err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
	})

	t.Run("mismatched digest", func(t *testing.T) {
		path := filepath.Join(dir, "bad-blob")
		if err := os.WriteFile(path, []byte("corrupted data"), 0o644); err != nil {
			t.Fatal(err)
		}
		err := verifyBlobFile(path, expectedDigest)
		if err == nil {
			t.Fatal("expected digest mismatch error, got nil")
		}
		if !errorIs(err, errDigestMismatch) {
			t.Fatalf("expected errDigestMismatch, got: %v", err)
		}
	})

	t.Run("missing file", func(t *testing.T) {
		err := verifyBlobFile(filepath.Join(dir, "nonexistent"), expectedDigest)
		if err == nil {
			t.Fatal("expected error for missing file, got nil")
		}
	})
}

// errorIs is a helper to check errors.Is without importing errors in the test
// (errDigestMismatch is already in the server package).
func errorIs(err, target error) bool {
	for err != nil {
		if err.Error() == target.Error() || err == target {
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

func TestDownloadChunkRejectsNon206(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		wantErr    bool
	}{
		{"206 Partial Content", http.StatusPartialContent, false},
		{"200 OK", http.StatusOK, false},
		{"403 Forbidden", http.StatusForbidden, true},
		{"502 Bad Gateway", http.StatusBadGateway, true},
		{"301 Moved", http.StatusMovedPermanently, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a test server that returns the specified status code
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				w.Write([]byte("test data padding for the request"))
			}))
			defer srv.Close()

			// Set up minimal blobDownload and part structures
			b := &blobDownload{}
			part := &blobDownloadPart{
				blobDownload: b,
				Size:         10,
			}

			tmpFile, err := os.CreateTemp(t.TempDir(), "chunk-test-*")
			if err != nil {
				t.Fatal(err)
			}
			defer tmpFile.Close()

			reqURL := mustParseURL(srv.URL)
			err = b.downloadChunk(t.Context(), reqURL, tmpFile, part)

			if tt.wantErr && err == nil {
				t.Errorf("expected error for status %d, got nil", tt.statusCode)
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error for status %d: %v", tt.statusCode, err)
			}
		})
	}
}

func mustParseURL(raw string) *url.URL {
	u, _ := url.Parse(raw)
	return u
}
