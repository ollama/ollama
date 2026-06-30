package transfer

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// chunkedSession tracks accumulated PATCH body bytes for an upload session.
// Tests that mock the registry use it to handle the GGUF-style POST → PATCH →
// PUT-finalize flow without each test reimplementing the bookkeeping.
type chunkedSession struct {
	mu       sync.Mutex
	sessions map[string]*bytes.Buffer
}

func newChunkedSession() *chunkedSession {
	return &chunkedSession{sessions: make(map[string]*bytes.Buffer)}
}

// recordPatch reads the request body into the session buffer and writes a
// 202 Accepted response with Docker-Upload-Location pointing at the same
// session URL. Use this from a mock handler's PATCH branch.
func (c *chunkedSession) recordPatch(w http.ResponseWriter, r *http.Request) {
	c.mu.Lock()
	buf, ok := c.sessions[r.URL.Path]
	if !ok {
		buf = &bytes.Buffer{}
		c.sessions[r.URL.Path] = buf
	}
	c.mu.Unlock()
	io.Copy(buf, r.Body)
	w.Header().Set("Docker-Upload-Location", r.URL.Path)
	w.WriteHeader(http.StatusAccepted)
}

// finalize returns the bytes accumulated for the given session URL path.
// Use it from the PUT-finalize branch of a mock handler.
func (c *chunkedSession) finalize(sessionPath string) []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	if buf, ok := c.sessions[sessionPath]; ok {
		return buf.Bytes()
	}
	return nil
}

// createTestBlob creates a blob with deterministic content and returns its digest
func createTestBlob(t *testing.T, dir string, size int) (Blob, []byte) {
	t.Helper()

	// Create deterministic content
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i % 256)
	}

	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)

	// Write to file
	path := filepath.Join(dir, digestToPath(digest))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	return Blob{Digest: digest, Size: int64(size)}, data
}

func TestDownload(t *testing.T) {
	// Create test blobs on "server"
	serverDir := t.TempDir()
	blob1, data1 := createTestBlob(t, serverDir, 1024)
	blob2, data2 := createTestBlob(t, serverDir, 2048)
	blob3, data3 := createTestBlob(t, serverDir, 512)

	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Extract digest from URL: /v2/library/_/blobs/sha256:...
		digest := filepath.Base(r.URL.Path)

		path := filepath.Join(serverDir, digestToPath(digest))
		data, err := os.ReadFile(path)
		if err != nil {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	// Download to client dir
	clientDir := t.TempDir()

	var progressCalls atomic.Int32
	var lastCompleted, lastTotal atomic.Int64

	err := Download(context.Background(), DownloadOptions{
		Blobs:       []Blob{blob1, blob2, blob3},
		BaseURL:     server.URL,
		DestDir:     clientDir,
		Concurrency: 2,
		Progress: func(completed, total int64) {
			progressCalls.Add(1)
			lastCompleted.Store(completed)
			lastTotal.Store(total)
		},
	})
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Verify files
	verifyBlob(t, clientDir, blob1, data1)
	verifyBlob(t, clientDir, blob2, data2)
	verifyBlob(t, clientDir, blob3, data3)

	// Verify progress was called
	if progressCalls.Load() == 0 {
		t.Error("Progress callback never called")
	}
	if lastTotal.Load() != blob1.Size+blob2.Size+blob3.Size {
		t.Errorf("Wrong total: got %d, want %d", lastTotal.Load(), blob1.Size+blob2.Size+blob3.Size)
	}
}

func TestDownloadWithRedirect(t *testing.T) {
	// Create test blob on "CDN"
	cdnDir := t.TempDir()
	blob, data := createTestBlob(t, cdnDir, 1024)

	// CDN server (the redirect target)
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Serve the blob content
		digest := filepath.Base(r.URL.Path)
		path := filepath.Join(cdnDir, digestToPath(digest))
		blobData, err := os.ReadFile(path)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(blobData)))
		w.WriteHeader(http.StatusOK)
		w.Write(blobData)
	}))
	defer cdn.Close()

	// Registry server (redirects to CDN)
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Redirect to CDN
		cdnURL := cdn.URL + r.URL.Path
		http.Redirect(w, r, cdnURL, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()

	clientDir := t.TempDir()

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: registry.URL,
		DestDir: clientDir,
	})
	if err != nil {
		t.Fatalf("Download with redirect failed: %v", err)
	}

	verifyBlob(t, clientDir, blob, data)
}

func TestDownloadWithRetry(t *testing.T) {
	// Create test blob
	serverDir := t.TempDir()
	blob, data := createTestBlob(t, serverDir, 1024)

	var requestCount atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := requestCount.Add(1)

		// Fail first 2 attempts, succeed on 3rd
		if count < 3 {
			http.Error(w, "temporary error", http.StatusServiceUnavailable)
			return
		}

		digest := filepath.Base(r.URL.Path)
		path := filepath.Join(serverDir, digestToPath(digest))
		blobData, err := os.ReadFile(path)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(blobData)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	if err != nil {
		t.Fatalf("Download with retry failed: %v", err)
	}

	verifyBlob(t, clientDir, blob, data)

	// Should have made 3 requests (2 failures + 1 success)
	if requestCount.Load() < 3 {
		t.Errorf("Expected at least 3 requests for retry, got %d", requestCount.Load())
	}
}

func TestDownloadWithAuth(t *testing.T) {
	serverDir := t.TempDir()
	blob, data := createTestBlob(t, serverDir, 1024)

	var authCalled atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Require auth
		auth := r.Header.Get("Authorization")
		if auth != "Bearer valid-token" {
			w.Header().Set("WWW-Authenticate", `Bearer realm="https://auth.example.com",service="registry",scope="repository:library:pull"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		digest := filepath.Base(r.URL.Path)
		path := filepath.Join(serverDir, digestToPath(digest))
		blobData, err := os.ReadFile(path)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(blobData)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: clientDir,
		GetToken: func(ctx context.Context, challenge AuthChallenge) (string, error) {
			authCalled.Store(true)
			if challenge.Realm != "https://auth.example.com" {
				t.Errorf("Wrong realm: %s", challenge.Realm)
			}
			if challenge.Service != "registry" {
				t.Errorf("Wrong service: %s", challenge.Service)
			}
			return "valid-token", nil
		},
	})
	if err != nil {
		t.Fatalf("Download with auth failed: %v", err)
	}

	if !authCalled.Load() {
		t.Error("GetToken was never called")
	}

	verifyBlob(t, clientDir, blob, data)
}

func TestDownloadSkipsExisting(t *testing.T) {
	serverDir := t.TempDir()
	blob1, data1 := createTestBlob(t, serverDir, 1024)

	// Pre-populate client dir
	clientDir := t.TempDir()
	path := filepath.Join(clientDir, digestToPath(blob1.Digest))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data1, 0o644); err != nil {
		t.Fatal(err)
	}

	var requestCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)
		http.NotFound(w, r)
	}))
	defer server.Close()

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob1},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Should not have made any requests (blob already exists)
	if requestCount.Load() != 0 {
		t.Errorf("Made %d requests, expected 0 (blob should be skipped)", requestCount.Load())
	}
}

func TestDownloadResumeProgressTotal(t *testing.T) {
	// Test that when resuming a download with some blobs already present:
	// 1. Total reflects ALL blob sizes (not just remaining)
	// 2. Completed starts at the size of already-downloaded blobs
	serverDir := t.TempDir()
	blob1, data1 := createTestBlob(t, serverDir, 1000)
	blob2, data2 := createTestBlob(t, serverDir, 2000)
	blob3, data3 := createTestBlob(t, serverDir, 3000)

	// Pre-populate client with blob1 and blob2 (simulating partial download)
	clientDir := t.TempDir()
	for _, b := range []struct {
		blob Blob
		data []byte
	}{{blob1, data1}, {blob2, data2}} {
		path := filepath.Join(clientDir, digestToPath(b.blob.Digest))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, b.data, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		digest := filepath.Base(r.URL.Path)
		path := filepath.Join(serverDir, digestToPath(digest))
		data, err := os.ReadFile(path)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	var firstCompleted, firstTotal int64
	var gotFirstProgress bool
	var mu sync.Mutex

	err := Download(context.Background(), DownloadOptions{
		Blobs:       []Blob{blob1, blob2, blob3},
		BaseURL:     server.URL,
		DestDir:     clientDir,
		Concurrency: 1,
		Progress: func(completed, total int64) {
			mu.Lock()
			defer mu.Unlock()
			if !gotFirstProgress {
				firstCompleted = completed
				firstTotal = total
				gotFirstProgress = true
			}
		},
	})
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Total should be sum of ALL blobs, not just blob3
	expectedTotal := blob1.Size + blob2.Size + blob3.Size
	if firstTotal != expectedTotal {
		t.Errorf("Total = %d, want %d (should include all blobs)", firstTotal, expectedTotal)
	}

	// First progress call should show already-completed bytes from blob1+blob2
	expectedCompleted := blob1.Size + blob2.Size
	if firstCompleted < expectedCompleted {
		t.Errorf("First completed = %d, want >= %d (should include already-downloaded blobs)", firstCompleted, expectedCompleted)
	}

	// Verify blob3 was downloaded
	verifyBlob(t, clientDir, blob3, data3)
}

func TestDownloadDigestMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return wrong data
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("wrong data"))
	}))
	defer server.Close()

	clientDir := t.TempDir()

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{{Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000", Size: 10}},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	if err == nil {
		t.Fatal("Expected error for digest mismatch")
	}
}

func TestUpload(t *testing.T) {
	// Create test blobs
	clientDir := t.TempDir()
	blob1, _ := createTestBlob(t, clientDir, 1024)
	blob2, _ := createTestBlob(t, clientDir, 2048)

	var uploadedBlobs sync.Map
	var uploadID atomic.Int32
	session := newChunkedSession()

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)

		case r.Method == http.MethodPost && r.URL.Path == "/v2/library/_/blobs/uploads/":
			id := uploadID.Add(1)
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/%d", serverURL, id))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPatch:
			session.recordPatch(w, r)

		case r.Method == http.MethodPut:
			digest := r.URL.Query().Get("digest")
			uploadedBlobs.Store(digest, session.finalize(r.URL.Path))
			w.WriteHeader(http.StatusCreated)

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	var progressCalls atomic.Int32
	err := Upload(context.Background(), UploadOptions{
		Blobs:       []Blob{blob1, blob2},
		BaseURL:     server.URL,
		SrcDir:      clientDir,
		Concurrency: 2,
		Progress: func(completed, total int64) {
			progressCalls.Add(1)
		},
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	// Verify both blobs were uploaded
	if _, ok := uploadedBlobs.Load(blob1.Digest); !ok {
		t.Error("Blob 1 not uploaded")
	}
	if _, ok := uploadedBlobs.Load(blob2.Digest); !ok {
		t.Error("Blob 2 not uploaded")
	}

	if progressCalls.Load() == 0 {
		t.Error("Progress callback never called")
	}
}

func TestUploadWithRedirect(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 1024)

	var uploadedBlobs sync.Map
	var cdnCalled atomic.Bool

	// CDN server (PATCH redirect target). PATCH is redirected to a PUT here,
	// matching production: server issues 307 + Location to a presigned CDN URL,
	// and the client re-uploads the part body via PUT.
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cdnCalled.Store(true)
		if r.Method == http.MethodPut {
			data, _ := io.ReadAll(r.Body)
			// Stash the body keyed by the path so the main server can pick it
			// up at finalize time.
			uploadedBlobs.Store(r.URL.Path, data)
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer cdn.Close()

	var serverURL string
	var uploadID atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)

		case http.MethodPost:
			id := uploadID.Add(1)
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/%d", serverURL, id))
			w.WriteHeader(http.StatusAccepted)

		case http.MethodPatch:
			// Redirect PATCH body to CDN, mirroring server behavior
			cdnURL := cdn.URL + r.URL.Path
			w.Header().Set("Docker-Upload-Location", r.URL.Path)
			http.Redirect(w, r, cdnURL, http.StatusTemporaryRedirect)

		case http.MethodPut:
			// Finalize: copy body the CDN received under this session path
			// to the uploadedBlobs map keyed by digest.
			digest := r.URL.Query().Get("digest")
			if v, ok := uploadedBlobs.Load(r.URL.Path); ok {
				uploadedBlobs.Store(digest, v)
				uploadedBlobs.Delete(r.URL.Path)
			}
			w.WriteHeader(http.StatusCreated)

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if err != nil {
		t.Fatalf("Upload with redirect failed: %v", err)
	}

	if !cdnCalled.Load() {
		t.Error("CDN was never called (redirect not followed)")
	}

	if _, ok := uploadedBlobs.Load(blob.Digest); !ok {
		t.Error("Blob not uploaded to CDN")
	}
}

func TestUploadWithAuth(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 1024)

	var uploadedBlobs sync.Map
	var authCalled atomic.Bool
	var uploadID atomic.Int32
	session := newChunkedSession()

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Require auth for all requests
		auth := r.Header.Get("Authorization")
		if auth != "Bearer valid-token" {
			w.Header().Set("WWW-Authenticate", `Bearer realm="https://auth.example.com",service="registry",scope="repository:library:push"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)

		case http.MethodPost:
			id := uploadID.Add(1)
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/%d", serverURL, id))
			w.WriteHeader(http.StatusAccepted)

		case http.MethodPatch:
			session.recordPatch(w, r)

		case http.MethodPut:
			digest := r.URL.Query().Get("digest")
			uploadedBlobs.Store(digest, session.finalize(r.URL.Path))
			w.WriteHeader(http.StatusCreated)

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
		GetToken: func(ctx context.Context, challenge AuthChallenge) (string, error) {
			authCalled.Store(true)
			return "valid-token", nil
		},
	})
	if err != nil {
		t.Fatalf("Upload with auth failed: %v", err)
	}

	if !authCalled.Load() {
		t.Error("GetToken was never called")
	}

	if _, ok := uploadedBlobs.Load(blob.Digest); !ok {
		t.Error("Blob not uploaded")
	}
}

func TestUploadSkipsExisting(t *testing.T) {
	clientDir := t.TempDir()
	blob1, _ := createTestBlob(t, clientDir, 1024)

	var headChecked atomic.Bool
	var putCalled atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			// HEAD check for blob existence - return 200 OK to indicate blob exists
			headChecked.Store(true)
			w.WriteHeader(http.StatusOK)
		case http.MethodPost:
			http.NotFound(w, r)
		case http.MethodPut:
			putCalled.Store(true)
			w.WriteHeader(http.StatusCreated)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob1},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	// Verify HEAD check was used
	if !headChecked.Load() {
		t.Error("HEAD check was never made")
	}

	// Should not have attempted PUT (blob already exists)
	if putCalled.Load() {
		t.Error("PUT was called even though blob exists (HEAD returned 200)")
	}

	t.Log("HEAD-based existence check verified")
}

// TestUploadWithCustomRepository verifies that custom repository paths are used
func TestUploadWithCustomRepository(t *testing.T) {
	clientDir := t.TempDir()
	blob1, _ := createTestBlob(t, clientDir, 1024)

	var headPath, postPath string
	var mu sync.Mutex
	session := newChunkedSession()

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			mu.Lock()
			headPath = r.URL.Path
			mu.Unlock()
			w.WriteHeader(http.StatusNotFound) // Blob doesn't exist
		case http.MethodPost:
			mu.Lock()
			postPath = r.URL.Path
			mu.Unlock()
			w.Header().Set("Location", fmt.Sprintf("%s/v2/myorg/mymodel/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPatch:
			session.recordPatch(w, r)
		case http.MethodPut:
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:      []Blob{blob1},
		BaseURL:    server.URL,
		SrcDir:     clientDir,
		Repository: "myorg/mymodel", // Custom repository
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	// Verify HEAD used custom repository path
	expectedHeadPath := fmt.Sprintf("/v2/myorg/mymodel/blobs/%s", blob1.Digest)
	if headPath != expectedHeadPath {
		t.Errorf("HEAD path mismatch: got %s, want %s", headPath, expectedHeadPath)
	}

	// Verify POST used custom repository path
	expectedPostPath := "/v2/myorg/mymodel/blobs/uploads/"
	if postPath != expectedPostPath {
		t.Errorf("POST path mismatch: got %s, want %s", postPath, expectedPostPath)
	}

	t.Logf("Custom repository paths verified: HEAD=%s, POST=%s", headPath, postPath)
}

// TestDownloadWithCustomRepository verifies that custom repository paths are used
func TestDownloadWithCustomRepository(t *testing.T) {
	serverDir := t.TempDir()
	blob, data := createTestBlob(t, serverDir, 1024)

	var requestPath string
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requestPath = r.URL.Path
		mu.Unlock()

		// Serve blob from any path
		digest := filepath.Base(r.URL.Path)
		path := filepath.Join(serverDir, digestToPath(digest))
		blobData, err := os.ReadFile(path)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(blobData)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	err := Download(context.Background(), DownloadOptions{
		Blobs:      []Blob{blob},
		BaseURL:    server.URL,
		DestDir:    clientDir,
		Repository: "myorg/mymodel", // Custom repository
	})
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	verifyBlob(t, clientDir, blob, data)

	mu.Lock()
	defer mu.Unlock()

	// Verify request used custom repository path
	expectedPath := fmt.Sprintf("/v2/myorg/mymodel/blobs/%s", blob.Digest)
	if requestPath != expectedPath {
		t.Errorf("Request path mismatch: got %s, want %s", requestPath, expectedPath)
	}

	t.Logf("Custom repository path verified: %s", requestPath)
}

func TestDigestToPath(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"sha256:abc123", "sha256-abc123"},
		{"sha256-abc123", "sha256-abc123"},
		{"other", "other"},
	}

	for _, tt := range tests {
		got := digestToPath(tt.input)
		if got != tt.want {
			t.Errorf("digestToPath(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestParseAuthChallenge(t *testing.T) {
	tests := []struct {
		input string
		want  AuthChallenge
	}{
		{
			input: `Bearer realm="https://auth.example.com/token",service="registry",scope="repository:library/test:pull"`,
			want: AuthChallenge{
				Realm:   "https://auth.example.com/token",
				Service: "registry",
				Scope:   "repository:library/test:pull",
			},
		},
		{
			input: `Bearer realm="https://auth.example.com"`,
			want: AuthChallenge{
				Realm: "https://auth.example.com",
			},
		},
	}

	for _, tt := range tests {
		got := parseAuthChallenge(tt.input)
		if got.Realm != tt.want.Realm {
			t.Errorf("parseAuthChallenge(%q).Realm = %q, want %q", tt.input, got.Realm, tt.want.Realm)
		}
		if got.Service != tt.want.Service {
			t.Errorf("parseAuthChallenge(%q).Service = %q, want %q", tt.input, got.Service, tt.want.Service)
		}
		if got.Scope != tt.want.Scope {
			t.Errorf("parseAuthChallenge(%q).Scope = %q, want %q", tt.input, got.Scope, tt.want.Scope)
		}
	}
}

func verifyBlob(t *testing.T, dir string, blob Blob, expected []byte) {
	t.Helper()

	path := filepath.Join(dir, digestToPath(blob.Digest))
	data, err := os.ReadFile(path)
	if err != nil {
		t.Errorf("Failed to read %s: %v", blob.Digest[:19], err)
		return
	}

	if len(data) != len(expected) {
		t.Errorf("Size mismatch for %s: got %d, want %d", blob.Digest[:19], len(data), len(expected))
		return
	}

	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	if digest != blob.Digest {
		t.Errorf("Digest mismatch for %s: got %s", blob.Digest[:19], digest[:19])
	}
}

// ==================== Parallelism Tests ====================

func TestDownloadParallelism(t *testing.T) {
	// Create many blobs to test parallelism
	serverDir := t.TempDir()
	numBlobs := 10
	blobs := make([]Blob, numBlobs)
	blobData := make([][]byte, numBlobs)

	for i := range numBlobs {
		blobs[i], blobData[i] = createTestBlob(t, serverDir, 1024+i*100)
	}

	var activeRequests atomic.Int32
	var maxConcurrent atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		current := activeRequests.Add(1)
		defer activeRequests.Add(-1)

		// Track max concurrent requests
		for {
			old := maxConcurrent.Load()
			if current <= old || maxConcurrent.CompareAndSwap(old, current) {
				break
			}
		}

		// Simulate network latency to ensure parallelism is visible
		time.Sleep(50 * time.Millisecond)

		digest := filepath.Base(r.URL.Path)
		path := filepath.Join(serverDir, digestToPath(digest))
		data, err := os.ReadFile(path)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	start := time.Now()
	err := Download(context.Background(), DownloadOptions{
		Blobs:           blobs,
		BaseURL:         server.URL,
		DestDir:         clientDir,
		Concurrency:     4,
		BodyConcurrency: 4,
	})
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Verify all blobs downloaded
	for i, blob := range blobs {
		verifyBlob(t, clientDir, blob, blobData[i])
	}

	// Verify parallelism was used
	if maxConcurrent.Load() < 2 {
		t.Errorf("Max concurrent requests was %d, expected at least 2 for parallelism", maxConcurrent.Load())
	}

	// With 10 blobs at 50ms each, sequential would take ~500ms
	// Parallel with 4 workers should take ~150ms (relax to 1s for CI variance)
	if elapsed > time.Second {
		t.Errorf("Downloads took %v, expected faster with parallelism", elapsed)
	}

	t.Logf("Downloaded %d blobs in %v with max %d concurrent requests", numBlobs, elapsed, maxConcurrent.Load())
}

func TestUploadParallelism(t *testing.T) {
	clientDir := t.TempDir()
	numBlobs := 10
	blobs := make([]Blob, numBlobs)

	for i := range numBlobs {
		blobs[i], _ = createTestBlob(t, clientDir, 1024+i*100)
	}

	var activeRequests atomic.Int32
	var maxConcurrent atomic.Int32
	var uploadedBlobs sync.Map
	var uploadID atomic.Int32
	session := newChunkedSession()

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		current := activeRequests.Add(1)
		defer activeRequests.Add(-1)

		// Track max concurrent
		for {
			old := maxConcurrent.Load()
			if current <= old || maxConcurrent.CompareAndSwap(old, current) {
				break
			}
		}

		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)

		case http.MethodPost:
			id := uploadID.Add(1)
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/%d", serverURL, id))
			w.WriteHeader(http.StatusAccepted)

		case http.MethodPatch:
			time.Sleep(50 * time.Millisecond) // simulate upload time on body chunk
			session.recordPatch(w, r)

		case http.MethodPut:
			digest := r.URL.Query().Get("digest")
			uploadedBlobs.Store(digest, session.finalize(r.URL.Path))
			w.WriteHeader(http.StatusCreated)

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	start := time.Now()
	err := Upload(context.Background(), UploadOptions{
		Blobs:       blobs,
		BaseURL:     server.URL,
		SrcDir:      clientDir,
		Concurrency: 4,
	})
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	// Verify all blobs uploaded
	for _, blob := range blobs {
		if _, ok := uploadedBlobs.Load(blob.Digest); !ok {
			t.Errorf("Blob %s not uploaded", blob.Digest[:19])
		}
	}

	if maxConcurrent.Load() < 2 {
		t.Errorf("Max concurrent requests was %d, expected at least 2", maxConcurrent.Load())
	}

	t.Logf("Uploaded %d blobs in %v with max %d concurrent requests", numBlobs, elapsed, maxConcurrent.Load())
}

// ==================== Stall Detection Test ====================

func TestDownloadStallDetection(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stall detection test in short mode")
	}

	serverDir := t.TempDir()
	blob, _ := createTestBlob(t, serverDir, 10*1024) // 10KB

	var requestCount atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := requestCount.Add(1)

		digest := filepath.Base(r.URL.Path)
		path := filepath.Join(serverDir, digestToPath(digest))
		data, err := os.ReadFile(path)
		if err != nil {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
		w.WriteHeader(http.StatusOK)

		if count == 1 {
			// First request: send partial data then stall
			w.Write(data[:1024]) // Send first 1KB
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			// Stall for longer than stall timeout (test uses 200ms)
			time.Sleep(500 * time.Millisecond)
			return
		}

		// Subsequent requests: send full data
		w.Write(data)
	}))
	defer func() {
		server.CloseClientConnections()
		server.Close()
	}()

	clientDir := t.TempDir()

	start := time.Now()
	err := Download(context.Background(), DownloadOptions{
		Blobs:        []Blob{blob},
		BaseURL:      server.URL,
		DestDir:      clientDir,
		StallTimeout: 200 * time.Millisecond, // Short timeout for testing
	})
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Should have retried after stall detection
	if requestCount.Load() < 2 {
		t.Errorf("Expected at least 2 requests (stall + retry), got %d", requestCount.Load())
	}

	// Should complete quickly with short stall timeout
	if elapsed > 3*time.Second {
		t.Errorf("Download took %v, stall detection should have triggered faster", elapsed)
	}

	t.Logf("Stall detection worked: %d requests in %v", requestCount.Load(), elapsed)
}

// ==================== Context Cancellation Tests ====================

func TestDownloadCancellation(t *testing.T) {
	serverDir := t.TempDir()
	blob, _ := createTestBlob(t, serverDir, 100*1024) // 100KB (smaller for faster test)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		digest := filepath.Base(r.URL.Path)
		path := filepath.Join(serverDir, digestToPath(digest))
		data, _ := os.ReadFile(path)

		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
		w.WriteHeader(http.StatusOK)

		// Send data slowly
		for i := 0; i < len(data); i += 1024 {
			end := i + 1024
			if end > len(data) {
				end = len(data)
			}
			w.Write(data[i:end])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer func() {
		server.CloseClientConnections()
		server.Close()
	}()

	clientDir := t.TempDir()

	ctx, cancel := context.WithCancel(context.Background())

	// Cancel after 50ms
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	err := Download(ctx, DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Expected error from cancellation")
	}

	if !errors.Is(err, context.Canceled) {
		t.Errorf("Expected context.Canceled error, got: %v", err)
	}

	// Should cancel quickly, not wait for full download
	if elapsed > 500*time.Millisecond {
		t.Errorf("Cancellation took %v, expected faster response", elapsed)
	}

	t.Logf("Cancellation worked in %v", elapsed)
}

func TestUploadCancellation(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 100*1024) // 100KB (smaller for faster test)

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)

		case http.MethodPost:
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)

		case http.MethodPatch:
			// Read slowly so the cancellation has time to interrupt the body upload.
			buf := make([]byte, 1024)
			for {
				_, err := r.Body.Read(buf)
				if err != nil {
					break
				}
				time.Sleep(5 * time.Millisecond)
			}
			w.Header().Set("Docker-Upload-Location", r.URL.Path)
			w.WriteHeader(http.StatusAccepted)

		case http.MethodPut:
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer func() {
		server.CloseClientConnections()
		server.Close()
	}()
	serverURL = server.URL

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	err := Upload(ctx, UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Expected error from cancellation")
	}

	if elapsed > 500*time.Millisecond {
		t.Errorf("Cancellation took %v, expected faster", elapsed)
	}

	t.Logf("Upload cancellation worked in %v", elapsed)
}

// ==================== Progress Tracking Tests ====================

func TestProgressTracking(t *testing.T) {
	serverDir := t.TempDir()
	blob1, data1 := createTestBlob(t, serverDir, 5000)
	blob2, data2 := createTestBlob(t, serverDir, 3000)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		digest := filepath.Base(r.URL.Path)
		path := filepath.Join(serverDir, digestToPath(digest))
		data, _ := os.ReadFile(path)
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	var progressHistory []struct{ completed, total int64 }
	var mu sync.Mutex

	err := Download(context.Background(), DownloadOptions{
		Blobs:       []Blob{blob1, blob2},
		BaseURL:     server.URL,
		DestDir:     clientDir,
		Concurrency: 1, // Sequential to make progress predictable
		Progress: func(completed, total int64) {
			mu.Lock()
			progressHistory = append(progressHistory, struct{ completed, total int64 }{completed, total})
			mu.Unlock()
		},
	})
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	verifyBlob(t, clientDir, blob1, data1)
	verifyBlob(t, clientDir, blob2, data2)

	mu.Lock()
	defer mu.Unlock()

	if len(progressHistory) == 0 {
		t.Fatal("No progress callbacks received")
	}

	// Total should always be sum of blob sizes
	expectedTotal := blob1.Size + blob2.Size
	for _, p := range progressHistory {
		if p.total != expectedTotal {
			t.Errorf("Total changed during download: got %d, want %d", p.total, expectedTotal)
		}
	}

	// Completed should be monotonically increasing
	var lastCompleted int64
	for _, p := range progressHistory {
		if p.completed < lastCompleted {
			t.Errorf("Progress went backwards: %d -> %d", lastCompleted, p.completed)
		}
		lastCompleted = p.completed
	}

	// Final completed should equal total
	final := progressHistory[len(progressHistory)-1]
	if final.completed != expectedTotal {
		t.Errorf("Final completed %d != total %d", final.completed, expectedTotal)
	}

	t.Logf("Progress tracked correctly: %d callbacks, final %d/%d", len(progressHistory), final.completed, final.total)
}

// ==================== Edge Cases ====================

func TestDownloadEmptyBlobList(t *testing.T) {
	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{},
		BaseURL: "http://unused",
		DestDir: t.TempDir(),
	})
	if err != nil {
		t.Errorf("Expected no error for empty blob list, got: %v", err)
	}
}

func TestUploadEmptyBlobList(t *testing.T) {
	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{},
		BaseURL: "http://unused",
		SrcDir:  t.TempDir(),
	})
	if err != nil {
		t.Errorf("Expected no error for empty blob list, got: %v", err)
	}
}

func TestUploadRetryOnFailure(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 1024)

	var patchCount atomic.Int32
	var uploadedBlobs sync.Map
	session := newChunkedSession()

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)

		case http.MethodPost:
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)

		case http.MethodPatch:
			count := patchCount.Add(1)
			if count < 3 {
				// Fail first 2 PATCH attempts to exercise the retry path
				io.Copy(io.Discard, r.Body)
				http.Error(w, "server error", http.StatusInternalServerError)
				return
			}
			session.recordPatch(w, r)

		case http.MethodPut:
			digest := r.URL.Query().Get("digest")
			uploadedBlobs.Store(digest, session.finalize(r.URL.Path))
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if err != nil {
		t.Fatalf("Upload with retry failed: %v", err)
	}

	if _, ok := uploadedBlobs.Load(blob.Digest); !ok {
		t.Error("Blob not uploaded after retry")
	}

	if patchCount.Load() < 3 {
		t.Errorf("Expected at least 3 PATCH attempts, got %d", patchCount.Load())
	}
}

// TestProgressRollback verifies that progress is rolled back on retry
func TestProgressRollback(t *testing.T) {
	content := []byte("test content for rollback test")
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(content))
	blob := Blob{Digest: digest, Size: int64(len(content))}

	clientDir := t.TempDir()
	path := filepath.Join(clientDir, digestToPath(digest))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatal(err)
	}

	var patchCount atomic.Int32
	var progressValues []int64
	var mu sync.Mutex
	session := newChunkedSession()

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)

		case http.MethodPost:
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)

		case http.MethodPatch:
			// Read some bytes (so the client reports progress) before failing,
			// to exercise the rollback-on-retry path.
			count := patchCount.Add(1)
			if count < 3 {
				io.CopyN(io.Discard, r.Body, 10)
				io.Copy(io.Discard, r.Body)
				http.Error(w, "server error", http.StatusInternalServerError)
				return
			}
			session.recordPatch(w, r)

		case http.MethodPut:
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
		Progress: func(completed, total int64) {
			mu.Lock()
			progressValues = append(progressValues, completed)
			mu.Unlock()
		},
	})
	if err != nil {
		t.Fatalf("Upload with retry failed: %v", err)
	}

	// Check that progress was rolled back (should have negative values or drops)
	mu.Lock()
	defer mu.Unlock()

	// Final progress should equal blob size
	if len(progressValues) > 0 {
		final := progressValues[len(progressValues)-1]
		if final != blob.Size {
			t.Errorf("Final progress %d != blob size %d", final, blob.Size)
		}
	}

	t.Logf("Progress rollback test: %d progress callbacks", len(progressValues))
}

// TestUserAgentHeader verifies User-Agent header is set on requests
func TestUserAgentHeader(t *testing.T) {
	content := []byte("test content")
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(content))
	blob := Blob{Digest: digest, Size: int64(len(content))}

	destDir := t.TempDir()
	var userAgents []string
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		ua := r.Header.Get("User-Agent")
		userAgents = append(userAgents, ua)
		mu.Unlock()

		if r.Method == http.MethodGet {
			w.Write(content)
		}
	}))
	defer server.Close()

	// Test with custom User-Agent
	customUA := "test-agent/1.0"
	err := Download(context.Background(), DownloadOptions{
		Blobs:     []Blob{blob},
		BaseURL:   server.URL,
		DestDir:   destDir,
		UserAgent: customUA,
	})
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	// Verify custom User-Agent was used
	for _, ua := range userAgents {
		if ua != customUA {
			t.Errorf("User-Agent %q != expected %q", ua, customUA)
		}
	}
	t.Logf("User-Agent header test: %d requests with correct User-Agent", len(userAgents))
}

// TestDefaultUserAgent verifies default User-Agent is used when not specified
func TestDefaultUserAgent(t *testing.T) {
	content := []byte("test content")
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(content))
	blob := Blob{Digest: digest, Size: int64(len(content))}

	destDir := t.TempDir()
	var userAgent string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		userAgent = r.Header.Get("User-Agent")
		if r.Method == http.MethodGet {
			w.Write(content)
		}
	}))
	defer server.Close()

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: destDir,
		// No UserAgent specified - should use default
	})
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	if userAgent == "" {
		t.Error("User-Agent header was empty")
	}
	if userAgent != defaultUserAgent {
		t.Errorf("Default User-Agent %q != expected %q", userAgent, defaultUserAgent)
	}
}

// TestManifestPush verifies that manifest is pushed after blobs
func TestManifestPush(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 1000)

	testManifest := []byte(`{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json"}`)
	testRepo := "library/test-model"
	testRef := "latest"

	var manifestReceived []byte
	var manifestPath string
	var manifestContentType string
	var serverURL string

	session := newChunkedSession()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)

		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/blobs/uploads"):
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPatch:
			session.recordPatch(w, r)

		case r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/blobs/"):
			// Finalize the chunked blob upload
			w.WriteHeader(http.StatusCreated)

		case r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/manifests/"):
			manifestPath = r.URL.Path
			manifestContentType = r.Header.Get("Content-Type")
			manifestReceived, _ = io.ReadAll(r.Body)
			w.WriteHeader(http.StatusCreated)

		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:       []Blob{blob},
		BaseURL:     server.URL,
		SrcDir:      clientDir,
		Manifest:    testManifest,
		ManifestRef: testRef,
		Repository:  testRepo,
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	// Verify manifest was pushed
	if manifestReceived == nil {
		t.Fatal("Manifest was not received by server")
	}

	if !bytes.Equal(manifestReceived, testManifest) {
		t.Errorf("Manifest content mismatch: got %s, want %s", manifestReceived, testManifest)
	}

	expectedPath := fmt.Sprintf("/v2/%s/manifests/%s", testRepo, testRef)
	if manifestPath != expectedPath {
		t.Errorf("Manifest path mismatch: got %s, want %s", manifestPath, expectedPath)
	}

	if manifestContentType != "application/vnd.docker.distribution.manifest.v2+json" {
		t.Errorf("Manifest content type mismatch: got %s", manifestContentType)
	}

	t.Logf("Manifest push test passed: received %d bytes at %s", len(manifestReceived), manifestPath)
}

// ==================== Throughput Benchmarks ====================

func BenchmarkDownloadThroughput(b *testing.B) {
	// Create test data - 1MB blob
	data := make([]byte, 1024*1024)
	for i := range data {
		data[i] = byte(i % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(len(data))}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	b.SetBytes(int64(len(data)))
	b.ResetTimer()

	for range b.N {
		clientDir := b.TempDir()
		err := Download(context.Background(), DownloadOptions{
			Blobs:       []Blob{blob},
			BaseURL:     server.URL,
			DestDir:     clientDir,
			Concurrency: 1,
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUploadThroughput(b *testing.B) {
	// Create test data - 1MB blob
	data := make([]byte, 1024*1024)
	for i := range data {
		data[i] = byte(i % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(len(data))}

	// Create source file once
	srcDir := b.TempDir()
	path := filepath.Join(srcDir, digestToPath(digest))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		b.Fatal(err)
	}

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)
		case http.MethodPost:
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPut:
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	b.SetBytes(int64(len(data)))
	b.ResetTimer()

	for range b.N {
		err := Upload(context.Background(), UploadOptions{
			Blobs:       []Blob{blob},
			BaseURL:     server.URL,
			SrcDir:      srcDir,
			Concurrency: 1,
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

// ==================== Resume Tests ====================

func TestResumeFromPartialFile(t *testing.T) {
	// Create a blob large enough for resume (>= resumeThreshold)
	blobSize := resumeThreshold + 1024
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte((i * 13) % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	var rangeHeader string
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
			w.WriteHeader(http.StatusOK)
			return
		}

		mu.Lock()
		rangeHeader = r.Header.Get("Range")
		mu.Unlock()

		rng := r.Header.Get("Range")
		if rng != "" {
			// Parse "bytes=N-"
			var start int64
			fmt.Sscanf(rng, "bytes=%d-", &start)
			if start > 0 && start < int64(blobSize) {
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, blobSize-1, blobSize))
				w.WriteHeader(http.StatusPartialContent)
				w.Write(data[start:])
				return
			}
		}

		w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	// Pre-create a partial .tmp file (first half)
	partialSize := blobSize / 2
	dest := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(dest), 0o755)
	os.WriteFile(dest+".tmp", data[:partialSize], 0o644)

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	if err != nil {
		t.Fatalf("Resume download failed: %v", err)
	}

	// Verify Range header was sent
	mu.Lock()
	if rangeHeader == "" {
		t.Error("Expected Range header for resume, got none")
	} else {
		expected := fmt.Sprintf("bytes=%d-", partialSize)
		if rangeHeader != expected {
			t.Errorf("Range header = %q, want %q", rangeHeader, expected)
		}
	}
	mu.Unlock()

	// Verify final file is correct
	finalData, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("Failed to read final file: %v", err)
	}
	if len(finalData) != blobSize {
		t.Errorf("Final file size = %d, want %d", len(finalData), blobSize)
	}
	finalHash := sha256.Sum256(finalData)
	if fmt.Sprintf("sha256:%x", finalHash) != digest {
		t.Error("Final file hash mismatch")
	}
}

func TestResumeCorruptPartialFile(t *testing.T) {
	blobSize := resumeThreshold + 1024
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte((i * 13) % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
			w.WriteHeader(http.StatusOK)
			return
		}

		rng := r.Header.Get("Range")
		if rng != "" {
			var start int64
			fmt.Sscanf(rng, "bytes=%d-", &start)
			if start > 0 && start < int64(blobSize) {
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, blobSize-1, blobSize))
				w.WriteHeader(http.StatusPartialContent)
				w.Write(data[start:])
				return
			}
		}

		w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	// Pre-create a partial .tmp file with CORRUPT data
	partialSize := blobSize / 2
	corruptData := make([]byte, partialSize)
	for i := range corruptData {
		corruptData[i] = 0xFF // All 0xFF — definitely wrong
	}
	dest := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(dest), 0o755)
	os.WriteFile(dest+".tmp", corruptData, 0o644)

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	// First attempt resumes with corrupt data → hash mismatch → retry.
	// Retry should clean up .tmp and re-download fully.
	if err != nil {
		t.Fatalf("Download with corrupt partial file failed: %v", err)
	}

	// Verify final file is correct
	finalData, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("Failed to read final file: %v", err)
	}
	finalHash := sha256.Sum256(finalData)
	if fmt.Sprintf("sha256:%x", finalHash) != digest {
		t.Error("Final file hash mismatch after corrupt resume recovery")
	}
}

func TestResumePartialFileLargerThanBlob(t *testing.T) {
	blobSize := resumeThreshold + 1024
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte((i * 13) % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	// Pre-create .tmp file LARGER than expected blob
	oversizedData := make([]byte, blobSize+1000)
	dest := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(dest), 0o755)
	os.WriteFile(dest+".tmp", oversizedData, 0o644)

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	if err != nil {
		t.Fatalf("Download with oversized .tmp failed: %v", err)
	}

	// Verify final file is correct
	finalData, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("Failed to read final file: %v", err)
	}
	finalHash := sha256.Sum256(finalData)
	if fmt.Sprintf("sha256:%x", finalHash) != digest {
		t.Error("Final file hash mismatch")
	}
}

func TestResumeBelowThreshold(t *testing.T) {
	// Blob below resume threshold should NOT attempt resume
	blobSize := 1024 // Well below resumeThreshold
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte(i % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	var gotRange atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Header.Get("Range") != "" {
			gotRange.Store(true)
		}
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	// Pre-create a partial .tmp file
	dest := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(dest), 0o755)
	os.WriteFile(dest+".tmp", data[:blobSize/2], 0o644)

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	if gotRange.Load() {
		t.Error("Range header sent for blob below resume threshold — should not attempt resume")
	}

	// Verify final file
	finalData, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("Failed to read final file: %v", err)
	}
	finalHash := sha256.Sum256(finalData)
	if fmt.Sprintf("sha256:%x", finalHash) != digest {
		t.Error("Final file hash mismatch")
	}
}

func TestResumeServerDoesNotSupportRange(t *testing.T) {
	blobSize := resumeThreshold + 1024
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte((i * 13) % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
			w.WriteHeader(http.StatusOK)
			return
		}
		// Ignore Range header — always return full content with 200
		w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	// Pre-create partial .tmp file
	dest := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(dest), 0o755)
	os.WriteFile(dest+".tmp", data[:blobSize/2], 0o644)

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	if err != nil {
		t.Fatalf("Download failed when server doesn't support Range: %v", err)
	}

	// Verify final file is correct
	finalData, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("Failed to read final file: %v", err)
	}
	finalHash := sha256.Sum256(finalData)
	if fmt.Sprintf("sha256:%x", finalHash) != digest {
		t.Error("Final file hash mismatch")
	}
}

func TestResumePartialFileExactSize(t *testing.T) {
	blobSize := resumeThreshold + 1024
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte((i * 13) % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	var requestCount atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
			w.WriteHeader(http.StatusOK)
			return
		}
		requestCount.Add(1)

		rng := r.Header.Get("Range")
		if rng != "" {
			var start int64
			fmt.Sscanf(rng, "bytes=%d-", &start)
			if start >= int64(blobSize) {
				// Nothing to send
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			if start > 0 {
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, blobSize-1, blobSize))
				w.WriteHeader(http.StatusPartialContent)
				w.Write(data[start:])
				return
			}
		}

		w.Header().Set("Content-Length", fmt.Sprintf("%d", blobSize))
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	}))
	defer server.Close()

	clientDir := t.TempDir()

	// Pre-create .tmp file with exact correct content (full size)
	// This simulates a download that completed but wasn't renamed
	dest := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(dest), 0o755)
	os.WriteFile(dest+".tmp", data, 0o644)

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		DestDir: clientDir,
	})
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Verify final file is correct
	finalData, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("Failed to read final file: %v", err)
	}
	resumeHash := sha256.Sum256(finalData)
	if fmt.Sprintf("sha256:%x", resumeHash) != digest {
		t.Error("Final file hash mismatch")
	}
}

// ==================== Chunked Upload Tests ====================

// chunkedUploadServer creates a test server that implements the OCI chunked
// upload protocol: POST → PATCH* (with Content-Range) → PUT (finalize).
type chunkedUploadServer struct {
	t             *testing.T
	mu            sync.Mutex
	parts         map[int][]byte // part offset -> received data
	patchCount    int
	finalized     bool
	finalDigest   string
	finalEtag     string
	patchHandler  func(w http.ResponseWriter, r *http.Request) // optional override
	uploadCounter int
	serverURL     *string
}

func newChunkedUploadServer(t *testing.T) *chunkedUploadServer {
	return &chunkedUploadServer{
		t:     t,
		parts: make(map[int][]byte),
	}
}

func (s *chunkedUploadServer) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)

		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/uploads"):
			s.mu.Lock()
			s.uploadCounter++
			id := s.uploadCounter
			s.mu.Unlock()
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/%d", *s.serverURL, id))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPatch:
			if s.patchHandler != nil {
				s.patchHandler(w, r)
				return
			}
			s.defaultPatchHandler(w, r)

		case r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/uploads"):
			s.mu.Lock()
			s.finalized = true
			s.finalDigest = r.URL.Query().Get("digest")
			s.finalEtag = r.URL.Query().Get("etag")
			s.mu.Unlock()
			w.WriteHeader(http.StatusCreated)

		default:
			http.NotFound(w, r)
		}
	}
}

func (s *chunkedUploadServer) defaultPatchHandler(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.patchCount++
	patchNum := s.patchCount
	s.mu.Unlock()

	cr := r.Header.Get("Content-Range")
	if cr == "" {
		http.Error(w, "missing Content-Range", http.StatusBadRequest)
		return
	}

	var start, end int64
	fmt.Sscanf(cr, "%d-%d", &start, &end)

	data, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	s.mu.Lock()
	s.parts[int(start)] = data
	s.mu.Unlock()

	w.Header().Set("Docker-Upload-Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/session-%d", *s.serverURL, patchNum+1))
	w.WriteHeader(http.StatusAccepted)
}

func (s *chunkedUploadServer) reassemble(totalSize int) []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	result := make([]byte, totalSize)
	for offset, data := range s.parts {
		copy(result[offset:], data)
	}
	return result
}

func TestComputeParts(t *testing.T) {
	tests := []struct {
		name      string
		totalSize int64
		wantParts int
		wantFirst int64
	}{
		{
			name:      "1GB blob — clamped to min part size",
			totalSize: 1 << 30,
			wantParts: int((1<<30 + minUploadPartSize - 1) / minUploadPartSize),
			wantFirst: minUploadPartSize,
		},
		{
			name:      "5GB blob — 16 parts",
			totalSize: 5 << 30,
			wantParts: 16,
			wantFirst: 5 << 30 / 16,
		},
		{
			name:      "20GB blob — clamped to max part size",
			totalSize: 20 << 30,
			wantParts: int((20<<30 + maxUploadPartSize - 1) / maxUploadPartSize),
			wantFirst: maxUploadPartSize,
		},
		{
			name:      "exactly min part size",
			totalSize: minUploadPartSize,
			wantParts: 1,
			wantFirst: minUploadPartSize,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parts := computeParts(tt.totalSize)
			if len(parts) != tt.wantParts {
				t.Errorf("computeParts(%d) = %d parts, want %d", tt.totalSize, len(parts), tt.wantParts)
			}
			if len(parts) > 0 && parts[0].size != tt.wantFirst {
				t.Errorf("first part size = %d, want %d", parts[0].size, tt.wantFirst)
			}

			// Verify parts cover entire blob with no gaps
			var total int64
			for i, p := range parts {
				if p.offset != total {
					t.Errorf("part %d offset = %d, want %d", i, p.offset, total)
				}
				if p.n != i {
					t.Errorf("part %d n = %d, want %d", i, p.n, i)
				}
				total += p.size
			}
			if total != tt.totalSize {
				t.Errorf("total part sizes = %d, want %d", total, tt.totalSize)
			}
		})
	}
}

func TestChunkedUploadBasic(t *testing.T) {
	blobSize := resumeThreshold + 1024
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte((i * 7) % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	clientDir := t.TempDir()
	path := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, data, 0o644)

	srv := newChunkedUploadServer(t)
	var serverURL string
	srv.serverURL = &serverURL
	server := httptest.NewServer(srv.handler())
	defer server.Close()
	serverURL = server.URL

	var progressCalls atomic.Int32
	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
		Progress: func(completed, total int64) {
			progressCalls.Add(1)
		},
	})
	if err != nil {
		t.Fatalf("Chunked upload failed: %v", err)
	}

	reassembled := srv.reassemble(blobSize)
	reassembledHash := sha256.Sum256(reassembled)
	if fmt.Sprintf("sha256:%x", reassembledHash) != digest {
		t.Error("Reassembled data hash mismatch")
	}

	srv.mu.Lock()
	if !srv.finalized {
		t.Error("Finalize PUT was not called")
	}
	if srv.finalDigest != digest {
		t.Errorf("Finalize digest = %s, want %s", srv.finalDigest, digest)
	}
	if srv.finalEtag == "" {
		t.Error("Finalize etag is empty")
	}
	if srv.patchCount == 0 {
		t.Error("No PATCH requests were sent")
	}
	srv.mu.Unlock()

	if progressCalls.Load() == 0 {
		t.Error("Progress callback never called")
	}
}

// TestSmallBlobUsesChunkedFlow verifies that even small blobs go through the
// PATCH+finalize chunked flow. Server-side redirect logic is gated on PATCH,
// so a single-PUT path could never trigger CDN redirection — every blob must
// use PATCH so the server has the chance to redirect.
func TestChunkedUploadCDNRedirect(t *testing.T) {
	blobSize := resumeThreshold + 1024
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte((i * 7) % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	clientDir := t.TempDir()
	path := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, data, 0o644)

	cdnParts := make(map[string][]byte)
	var cdnMu sync.Mutex
	var cdnGotAuth atomic.Bool

	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			cdnGotAuth.Store(true)
		}
		cdnData, _ := io.ReadAll(r.Body)
		cdnMu.Lock()
		cdnParts[r.URL.Path] = cdnData
		cdnMu.Unlock()
		w.WriteHeader(http.StatusCreated)
	}))
	defer cdn.Close()

	var serverURL string
	var patchCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)
		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/uploads"):
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)
		case r.Method == http.MethodPatch:
			n := patchCount.Add(1)
			w.Header().Set("Docker-Upload-Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/session-%d", serverURL, n+1))
			cdnPath := fmt.Sprintf("/cdn/part-%d", n)
			http.Redirect(w, r, cdn.URL+cdnPath, http.StatusTemporaryRedirect)
		case r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/uploads"):
			w.WriteHeader(http.StatusCreated)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if err != nil {
		t.Fatalf("Upload with CDN redirect failed: %v", err)
	}

	cdnMu.Lock()
	totalCDNBytes := 0
	for _, d := range cdnParts {
		totalCDNBytes += len(d)
	}
	cdnMu.Unlock()

	if totalCDNBytes != blobSize {
		t.Errorf("CDN received %d bytes, want %d", totalCDNBytes, blobSize)
	}

	if cdnGotAuth.Load() {
		t.Error("CDN received Authorization header — should not be sent to CDN")
	}
}

func TestChunkedUploadPartRetry(t *testing.T) {
	blobSize := resumeThreshold + 1024
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte(i % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	clientDir := t.TempDir()
	path := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, data, 0o644)

	var patchAttempts atomic.Int32

	srv := newChunkedUploadServer(t)
	srv.patchHandler = func(w http.ResponseWriter, r *http.Request) {
		attempt := patchAttempts.Add(1)
		if attempt == 1 {
			http.Error(w, "server error", http.StatusInternalServerError)
			return
		}
		srv.defaultPatchHandler(w, r)
	}
	var serverURL string
	srv.serverURL = &serverURL
	server := httptest.NewServer(srv.handler())
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if err != nil {
		t.Fatalf("Upload with retry failed: %v", err)
	}

	if patchAttempts.Load() < 2 {
		t.Errorf("Expected at least 2 PATCH attempts, got %d", patchAttempts.Load())
	}

	reassembled := srv.reassemble(blobSize)
	reassembledHash := sha256.Sum256(reassembled)
	if fmt.Sprintf("sha256:%x", reassembledHash) != digest {
		t.Error("Data integrity failed after retry")
	}
}

// multiPartTestHelper creates an uploader with small part sizes so multi-part
// behavior can be tested with small blobs. Returns the uploader and blob data.
func multiPartTestHelper(t *testing.T, blobSize int, partSize int64, serverURL string) (*uploader, Blob, []byte) {
	t.Helper()
	data := make([]byte, blobSize)
	for i := range data {
		data[i] = byte((i*7 + 13) % 256)
	}
	h := sha256.Sum256(data)
	digest := fmt.Sprintf("sha256:%x", h)
	blob := Blob{Digest: digest, Size: int64(blobSize)}

	clientDir := t.TempDir()
	path := filepath.Join(clientDir, digestToPath(digest))
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, data, 0o644)

	u := &uploader{
		client:    defaultClient,
		baseURL:   serverURL,
		srcDir:    clientDir,
		userAgent: defaultUserAgent,
		progress:  newProgressTracker(int64(blobSize), nil),
		makeParts: func(totalSize int64) []uploadPart {
			return computePartsWithLimits(totalSize, 16, partSize, partSize*10)
		},
	}
	return u, blob, data
}

func TestChunkedUploadMultiPartSessionURLChain(t *testing.T) {
	// Use 10KB blobs with 2KB parts → 5 parts, exercising the URL chain
	blobSize := 10240
	partSize := int64(2048)

	var patchURLs []string
	var mu sync.Mutex

	srv := newChunkedUploadServer(t)
	srv.patchHandler = func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		patchURLs = append(patchURLs, r.URL.Path)
		mu.Unlock()
		srv.defaultPatchHandler(w, r)
	}
	var serverURL string
	srv.serverURL = &serverURL
	server := httptest.NewServer(srv.handler())
	defer server.Close()
	serverURL = server.URL

	u, blob, data := multiPartTestHelper(t, blobSize, partSize, server.URL)

	f, err := os.Open(filepath.Join(u.srcDir, digestToPath(blob.Digest)))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	initURL := fmt.Sprintf("%s/v2/library/_/blobs/uploads/init-1", server.URL)
	n, err := u.putChunked(context.Background(), initURL, f, blob)
	if err != nil {
		t.Fatalf("putChunked failed: %v", err)
	}
	if n != int64(blobSize) {
		t.Errorf("bytes written = %d, want %d", n, blobSize)
	}

	// Verify data integrity
	reassembled := srv.reassemble(blobSize)
	if !bytes.Equal(reassembled, data) {
		t.Error("Reassembled data mismatch")
	}

	mu.Lock()
	defer mu.Unlock()

	// Should have 5 parts with distinct URLs
	if len(patchURLs) != 5 {
		t.Fatalf("Expected 5 PATCH requests, got %d", len(patchURLs))
	}

	// First PATCH uses the init URL
	if !strings.Contains(patchURLs[0], "init-1") {
		t.Errorf("First PATCH URL should contain init-1, got %s", patchURLs[0])
	}

	// Subsequent PATCHes should use session URLs from Docker-Upload-Location
	for i := 1; i < len(patchURLs); i++ {
		if patchURLs[i] == patchURLs[i-1] {
			t.Errorf("PATCH %d used same URL as PATCH %d — chain broken", i, i-1)
		}
		if !strings.Contains(patchURLs[i], "session-") {
			t.Errorf("PATCH %d URL should contain session-, got %s", i, patchURLs[i])
		}
	}
}

func TestChunkedUploadMultiPartDataIntegrity(t *testing.T) {
	// Non-evenly-divisible: 10001 bytes with 3000-byte parts → 4 parts (3000+3000+3000+1001)
	blobSize := 10001
	partSize := int64(3000)

	srv := newChunkedUploadServer(t)
	var serverURL string
	srv.serverURL = &serverURL
	server := httptest.NewServer(srv.handler())
	defer server.Close()
	serverURL = server.URL

	u, blob, data := multiPartTestHelper(t, blobSize, partSize, server.URL)

	f, err := os.Open(filepath.Join(u.srcDir, digestToPath(blob.Digest)))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	initURL := fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", server.URL)
	_, err = u.putChunked(context.Background(), initURL, f, blob)
	if err != nil {
		t.Fatalf("putChunked failed: %v", err)
	}

	reassembled := srv.reassemble(blobSize)
	if !bytes.Equal(reassembled, data) {
		t.Error("Reassembled data mismatch with non-evenly-divisible parts")
	}

	srv.mu.Lock()
	if srv.patchCount != 4 {
		t.Errorf("Expected 4 PATCH requests, got %d", srv.patchCount)
	}
	srv.mu.Unlock()
}

func TestChunkedUploadMultiPartProgressRollback(t *testing.T) {
	blobSize := 6000
	partSize := int64(2000) // 3 parts

	var patchAttempts atomic.Int32

	srv := newChunkedUploadServer(t)
	srv.patchHandler = func(w http.ResponseWriter, r *http.Request) {
		attempt := patchAttempts.Add(1)
		// Fail the second PATCH attempt (part 1, first try). Drain the body
		// before erroring so the server sends 100 Continue (under Expect:
		// 100-continue) and the client uploads the body — that's what makes
		// the progress rollback observable.
		if attempt == 2 {
			io.Copy(io.Discard, r.Body)
			http.Error(w, "server error", http.StatusInternalServerError)
			return
		}
		srv.defaultPatchHandler(w, r)
	}
	var serverURL string
	srv.serverURL = &serverURL
	server := httptest.NewServer(srv.handler())
	defer server.Close()
	serverURL = server.URL

	u, blob, data := multiPartTestHelper(t, blobSize, partSize, server.URL)
	// Track progress
	var progressValues []int64
	var mu sync.Mutex
	u.progress = newProgressTracker(int64(blobSize), func(completed, total int64) {
		mu.Lock()
		progressValues = append(progressValues, completed)
		mu.Unlock()
	})

	f, err := os.Open(filepath.Join(u.srcDir, digestToPath(blob.Digest)))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	initURL := fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", server.URL)
	_, err = u.putChunked(context.Background(), initURL, f, blob)
	if err != nil {
		t.Fatalf("putChunked failed: %v", err)
	}

	// Verify data integrity despite retry
	reassembled := srv.reassemble(blobSize)
	if !bytes.Equal(reassembled, data) {
		t.Error("Data mismatch after retry")
	}

	// Verify progress had a rollback (decrease) then recovered
	mu.Lock()
	defer mu.Unlock()

	hadDecrease := false
	for i := 1; i < len(progressValues); i++ {
		if progressValues[i] < progressValues[i-1] {
			hadDecrease = true
			break
		}
	}
	if !hadDecrease {
		t.Error("Expected progress to decrease (rollback) during retry, but it was monotonic")
	}

	// Final should equal blob size
	if len(progressValues) > 0 && progressValues[len(progressValues)-1] != int64(blobSize) {
		t.Errorf("Final progress = %d, want %d", progressValues[len(progressValues)-1], blobSize)
	}
}

// ==================== v2 direct-upload extension tests ====================

// TestV2InitRequestShape verifies the init POST advertises the v2 capability
// with the expected query parameter and headers, and that the request body
// is empty.
func TestV2InitRequestShape(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 4096)

	var sawDigestQuery, sawCapHeader, sawSizeHeader string
	var bodyLen int
	session := newChunkedSession()

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)
		case http.MethodPost:
			sawDigestQuery = r.URL.Query().Get("digest")
			sawCapHeader = r.Header.Get("X-Redirect-Uploads")
			sawSizeHeader = r.Header.Get("X-Content-Length")
			body, _ := io.ReadAll(r.Body)
			bodyLen = len(body)
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPatch:
			session.recordPatch(w, r)
		case http.MethodPut:
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	if sawDigestQuery != blob.Digest {
		t.Errorf("init POST ?digest= = %q, want %q", sawDigestQuery, blob.Digest)
	}
	if sawCapHeader != "2" {
		t.Errorf("init POST X-Redirect-Uploads = %q, want %q", sawCapHeader, "2")
	}
	if sawSizeHeader != fmt.Sprintf("%d", blob.Size) {
		t.Errorf("init POST X-Content-Length = %q, want %q", sawSizeHeader, fmt.Sprintf("%d", blob.Size))
	}
	if bodyLen != 0 {
		t.Errorf("init POST body length = %d, want 0", bodyLen)
	}
}

// TestV2DirectUpload verifies the v2 happy path: server returns
// X-Direct-Upload-URL + X-Signed-Header-X-Amz-Checksum-Sha256, the client
// PUTs body to the direct URL with the forwarded checksum header, then
// commits via a bodyless PUT to the session URL.
func TestV2DirectUpload(t *testing.T) {
	clientDir := t.TempDir()
	blob, data := createTestBlob(t, clientDir, 8192)

	var (
		cdnReceived       []byte
		cdnChecksumHeader string
		cdnAuthHeader     string
		cdnHits           atomic.Int32
	)

	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cdnHits.Add(1)
		if r.Method != http.MethodPut {
			http.Error(w, "want PUT", http.StatusMethodNotAllowed)
			return
		}
		cdnChecksumHeader = r.Header.Get("X-Amz-Checksum-Sha256")
		cdnAuthHeader = r.Header.Get("Authorization")
		cdnReceived, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer cdn.Close()

	// Compute the checksum the registry would sign — base64 of the SHA-256
	// binary digest. The mock just hands the value back to the client; the
	// client forwards it to the CDN.
	hexDigest := strings.TrimPrefix(blob.Digest, "sha256:")
	digestBytes := make([]byte, len(hexDigest)/2)
	for i := range digestBytes {
		fmt.Sscanf(hexDigest[i*2:i*2+2], "%02x", &digestBytes[i])
	}
	expectedChecksum := base64.StdEncoding.EncodeToString(digestBytes)

	var (
		commitDigest atomic.Value
		commitBody   atomic.Int32
	)

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)
		case http.MethodPost:
			directURL := cdn.URL + "/upload/" + blob.Digest
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.Header().Set("X-Direct-Upload-URL", directURL)
			w.Header().Set("X-Signed-Header-X-Amz-Checksum-Sha256", expectedChecksum)
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPut:
			commitDigest.Store(r.URL.Query().Get("digest"))
			body, _ := io.ReadAll(r.Body)
			commitBody.Store(int32(len(body)))
			w.WriteHeader(http.StatusCreated)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	if got := cdnHits.Load(); got != 1 {
		t.Errorf("CDN hits = %d, want 1", got)
	}
	if !bytes.Equal(cdnReceived, data) {
		t.Errorf("CDN body length = %d, want %d", len(cdnReceived), len(data))
	}
	if cdnChecksumHeader != expectedChecksum {
		t.Errorf("CDN x-amz-checksum-sha256 = %q, want %q", cdnChecksumHeader, expectedChecksum)
	}
	if cdnAuthHeader != "" {
		t.Errorf("CDN Authorization = %q, want empty (presigned URL shouldn't carry auth)", cdnAuthHeader)
	}
	if got, _ := commitDigest.Load().(string); got != blob.Digest {
		t.Errorf("commit ?digest= = %q, want %q", got, blob.Digest)
	}
	if commitBody.Load() != 0 {
		t.Errorf("commit body length = %d, want 0", commitBody.Load())
	}
}

// TestV2FallbackToChunked verifies that when the server returns a standard
// 202 without v2 extension headers, the client falls back to the chunked
// PATCH path. This exercises the vanilla Docker Registry compatibility.
func TestV2FallbackToChunked(t *testing.T) {
	clientDir := t.TempDir()
	blob, data := createTestBlob(t, clientDir, 8192)

	var (
		uploadedBlobs sync.Map
		patchHit      atomic.Int32
		commitHit     atomic.Int32
	)
	session := newChunkedSession()

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)
		case http.MethodPost:
			// vanilla: standard Location only, no v2 extension headers
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPatch:
			patchHit.Add(1)
			session.recordPatch(w, r)
		case http.MethodPut:
			commitHit.Add(1)
			digest := r.URL.Query().Get("digest")
			uploadedBlobs.Store(digest, session.finalize(r.URL.Path))
			w.WriteHeader(http.StatusCreated)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	if patchHit.Load() < 1 {
		t.Error("expected at least one PATCH (chunked fallback), got none")
	}
	if commitHit.Load() != 1 {
		t.Errorf("commit PUT hits = %d, want 1", commitHit.Load())
	}
	if got, ok := uploadedBlobs.Load(blob.Digest); !ok {
		t.Error("blob not uploaded")
	} else if !bytes.Equal(got.([]byte), data) {
		t.Errorf("uploaded body length = %d, want %d", len(got.([]byte)), len(data))
	}
}

// TestV2BlobAlreadyExists verifies that a 201 Created response from the init
// POST short-circuits the upload — the server has matched our ?digest=
// against existing storage and there's nothing to upload.
func TestV2BlobAlreadyExists(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 1024)

	var (
		postHits  atomic.Int32
		patchHits atomic.Int32
		putHits   atomic.Int32
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)
		case http.MethodPost:
			postHits.Add(1)
			// Server matched ?digest= against existing storage.
			w.WriteHeader(http.StatusCreated)
		case http.MethodPatch:
			patchHits.Add(1)
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPut:
			putHits.Add(1)
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer server.Close()

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	if postHits.Load() != 1 {
		t.Errorf("init POST hits = %d, want 1", postHits.Load())
	}
	if patchHits.Load() != 0 {
		t.Errorf("PATCH hits = %d, want 0 (blob existed; nothing to upload)", patchHits.Load())
	}
	if putHits.Load() != 0 {
		t.Errorf("PUT hits = %d, want 0 (blob existed; nothing to upload)", putHits.Load())
	}
}
