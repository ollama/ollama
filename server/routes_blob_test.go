package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/manifest"
)

// postBlob uploads body to the blob endpoint for the digest given in the URL.
func postBlob(t *testing.T, s *Server, digest string, body []byte) *responseRecorder {
	t.Helper()

	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = &http.Request{
		Method: http.MethodPost,
		Body:   io.NopCloser(bytes.NewReader(body)),
	}
	c.Params = gin.Params{{Key: "digest", Value: digest}}
	s.CreateBlobHandler(c)
	c.Writer.WriteHeaderNow()
	return w
}

func blobsIn(t *testing.T) []string {
	t.Helper()

	p, err := manifest.BlobsPath("")
	if err != nil {
		t.Fatal(err)
	}

	entries, err := os.ReadDir(p)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}

	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

func TestCreateBlobHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	payload := bytes.Repeat([]byte("ollama"), 200_000)
	sum := sha256.Sum256(payload)
	digest := fmt.Sprintf("sha256:%x", sum)

	t.Run("accepts a matching upload", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		var s Server

		w := postBlob(t, &s, digest, payload)
		if w.Code != http.StatusCreated {
			t.Fatalf("first upload status = %d, want %d", w.Code, http.StatusCreated)
		}

		// second upload of the same blob takes the "already exists" shortcut
		w = postBlob(t, &s, digest, payload)
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
		}
	})

	t.Run("rejects a mismatched upload without storing it", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		var s Server

		wrong := "sha256:" + strings.Repeat("0", 64)
		w := postBlob(t, &s, wrong, payload)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want %d", w.Code, http.StatusBadRequest)
		}

		var resp struct {
			Error string `json:"error"`
		}
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(resp.Error, digest) {
			t.Errorf("error = %q, want it to mention the actual digest %s", resp.Error, digest)
		}

		// the bytes were rejected, so neither the real digest nor a temp file may
		// be left in blobs/
		if got := blobsIn(t); len(got) != 0 {
			t.Fatalf("blobs/ = %v, want empty", got)
		}

		// and a later upload with the right digest still works
		w = postBlob(t, &s, digest, payload)
		if w.Code != http.StatusCreated {
			t.Fatalf("status = %d, want %d", w.Code, http.StatusCreated)
		}

		b, err := os.ReadFile(filepath.Join(filepath.Dir(mustBlobPath(t, digest)), filepath.Base(mustBlobPath(t, digest))))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b, payload) {
			t.Fatalf("stored blob has %d bytes, want %d", len(b), len(payload))
		}
	})

	t.Run("rejects an unreadable body", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		var s Server

		w := NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = &http.Request{Method: http.MethodPost, Body: io.NopCloser(errReader{})}
		c.Params = gin.Params{{Key: "digest", Value: digest}}
		s.CreateBlobHandler(c)
		c.Writer.WriteHeaderNow()

		if w.Code != http.StatusInternalServerError {
			t.Fatalf("status = %d, want %d", w.Code, http.StatusInternalServerError)
		}
		if got := blobsIn(t); len(got) != 0 {
			t.Fatalf("blobs/ = %v, want empty", got)
		}
	})
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }

func mustBlobPath(t *testing.T, digest string) string {
	t.Helper()
	p, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
