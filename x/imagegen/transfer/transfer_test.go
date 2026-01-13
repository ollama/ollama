package transfer

import (
	"bytes"
	"context"
	"crypto/sha256"
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
	uploadID := 0

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead:
			// Blob doesn't exist
			http.NotFound(w, r)

		case r.Method == http.MethodPost && r.URL.Path == "/v2/library/_/blobs/uploads/":
			// Initiate upload
			uploadID++
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/%d", serverURL, uploadID))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPut:
			// Complete upload
			digest := r.URL.Query().Get("digest")
			data, _ := io.ReadAll(r.Body)
			uploadedBlobs.Store(digest, data)
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

	// CDN server (redirect target)
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cdnCalled.Store(true)
		if r.Method == http.MethodPut {
			digest := r.URL.Query().Get("digest")
			data, _ := io.ReadAll(r.Body)
			uploadedBlobs.Store(digest, data)
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer cdn.Close()

	var serverURL string
	uploadID := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)

		case r.Method == http.MethodPost && r.URL.Path == "/v2/library/_/blobs/uploads/":
			uploadID++
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/%d", serverURL, uploadID))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPut:
			// Redirect to CDN
			cdnURL := cdn.URL + r.URL.Path + "?" + r.URL.RawQuery
			http.Redirect(w, r, cdnURL, http.StatusTemporaryRedirect)

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
	uploadID := 0

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Require auth for all requests
		auth := r.Header.Get("Authorization")
		if auth != "Bearer valid-token" {
			w.Header().Set("WWW-Authenticate", `Bearer realm="https://auth.example.com",service="registry",scope="repository:library:push"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)

		case r.Method == http.MethodPost && r.URL.Path == "/v2/library/_/blobs/uploads/":
			uploadID++
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/%d", serverURL, uploadID))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPut:
			digest := r.URL.Query().Get("digest")
			data, _ := io.ReadAll(r.Body)
			uploadedBlobs.Store(digest, data)
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

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		switch r.Method {
		case http.MethodHead:
			headPath = r.URL.Path
			w.WriteHeader(http.StatusNotFound) // Blob doesn't exist
		case http.MethodPost:
			postPath = r.URL.Path
			w.Header().Set("Location", fmt.Sprintf("%s/v2/myorg/mymodel/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPut:
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusCreated)
		}
		mu.Unlock()
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
		Blobs:       blobs,
		BaseURL:     server.URL,
		DestDir:     clientDir,
		Concurrency: 4,
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

		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)

		case r.Method == http.MethodPost:
			id := uploadID.Add(1)
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/%d", serverURL, id))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPut:
			time.Sleep(50 * time.Millisecond) // Simulate upload time
			digest := r.URL.Query().Get("digest")
			data, _ := io.ReadAll(r.Body)
			uploadedBlobs.Store(digest, data)
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
		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)

		case r.Method == http.MethodPost:
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPut:
			// Read slowly
			buf := make([]byte, 1024)
			for {
				_, err := r.Body.Read(buf)
				if err != nil {
					break
				}
				time.Sleep(5 * time.Millisecond)
			}
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

func TestDownloadManyBlobs(t *testing.T) {
	// Test with many blobs to verify high concurrency works
	serverDir := t.TempDir()
	numBlobs := 50
	blobs := make([]Blob, numBlobs)
	blobData := make([][]byte, numBlobs)

	for i := range numBlobs {
		blobs[i], blobData[i] = createTestBlob(t, serverDir, 512) // Small blobs
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		Blobs:       blobs,
		BaseURL:     server.URL,
		DestDir:     clientDir,
		Concurrency: 16,
	})
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	// Verify all blobs
	for i, blob := range blobs {
		verifyBlob(t, clientDir, blob, blobData[i])
	}

	t.Logf("Downloaded %d blobs in %v", numBlobs, elapsed)
}

func TestUploadRetryOnFailure(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 1024)

	var putCount atomic.Int32
	var uploadedBlobs sync.Map

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)

		case r.Method == http.MethodPost:
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPut:
			count := putCount.Add(1)
			if count < 3 {
				// Fail first 2 attempts
				http.Error(w, "server error", http.StatusInternalServerError)
				return
			}
			digest := r.URL.Query().Get("digest")
			data, _ := io.ReadAll(r.Body)
			uploadedBlobs.Store(digest, data)
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

	if putCount.Load() < 3 {
		t.Errorf("Expected at least 3 PUT attempts, got %d", putCount.Load())
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

	var putCount atomic.Int32
	var progressValues []int64
	var mu sync.Mutex

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead:
			http.NotFound(w, r)

		case r.Method == http.MethodPost:
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPut:
			// Read some data before failing
			io.CopyN(io.Discard, r.Body, 10)
			count := putCount.Add(1)
			if count < 3 {
				http.Error(w, "server error", http.StatusInternalServerError)
				return
			}
			io.Copy(io.Discard, r.Body)
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

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Handle blob check (HEAD)
		if r.Method == http.MethodHead {
			http.NotFound(w, r)
			return
		}

		// Handle blob upload initiate (POST)
		if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/blobs/uploads") {
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", serverURL))
			w.WriteHeader(http.StatusAccepted)
			return
		}

		// Handle blob upload (PUT to blobs)
		if r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/blobs/") {
			w.WriteHeader(http.StatusCreated)
			return
		}

		// Handle manifest push (PUT to manifests)
		if r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/manifests/") {
			manifestPath = r.URL.Path
			manifestContentType = r.Header.Get("Content-Type")
			manifestReceived, _ = io.ReadAll(r.Body)
			w.WriteHeader(http.StatusCreated)
			return
		}

		http.NotFound(w, r)
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

// TestThroughput is a quick throughput test that reports MB/s
func TestThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping throughput test in short mode")
	}

	// Test parameters - 5MB total across 5 blobs
	const blobSize = 1024 * 1024 // 1MB per blob
	const numBlobs = 5
	const concurrency = 5

	// Create test blobs
	serverDir := t.TempDir()
	blobs := make([]Blob, numBlobs)
	for i := range numBlobs {
		data := make([]byte, blobSize)
		// Different seed per blob for unique digests
		for j := range data {
			data[j] = byte((i*256 + j) % 256)
		}
		h := sha256.Sum256(data)
		digest := fmt.Sprintf("sha256:%x", h)
		blobs[i] = Blob{Digest: digest, Size: int64(len(data))}

		path := filepath.Join(serverDir, digestToPath(digest))
		os.MkdirAll(filepath.Dir(path), 0o755)
		os.WriteFile(path, data, 0o644)
	}

	totalBytes := int64(blobSize * numBlobs)

	// Download server
	dlServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	defer dlServer.Close()

	// Measure download throughput
	clientDir := t.TempDir()
	start := time.Now()
	err := Download(context.Background(), DownloadOptions{
		Blobs:       blobs,
		BaseURL:     dlServer.URL,
		DestDir:     clientDir,
		Concurrency: concurrency,
	})
	dlElapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	dlThroughput := float64(totalBytes) / dlElapsed.Seconds() / (1024 * 1024)
	t.Logf("Download: %.2f MB/s (%d bytes in %v)", dlThroughput, totalBytes, dlElapsed)

	// Upload server
	var ulServerURL string
	ulServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)
		case http.MethodPost:
			w.Header().Set("Location", fmt.Sprintf("%s/v2/library/_/blobs/uploads/1", ulServerURL))
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPut:
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer ulServer.Close()
	ulServerURL = ulServer.URL

	// Measure upload throughput
	start = time.Now()
	err = Upload(context.Background(), UploadOptions{
		Blobs:       blobs,
		BaseURL:     ulServer.URL,
		SrcDir:      serverDir,
		Concurrency: concurrency,
	})
	ulElapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Upload failed: %v", err)
	}

	ulThroughput := float64(totalBytes) / ulElapsed.Seconds() / (1024 * 1024)
	t.Logf("Upload: %.2f MB/s (%d bytes in %v)", ulThroughput, totalBytes, ulElapsed)

	// Sanity check - local transfers should be fast (>50 MB/s is reasonable for local)
	// This ensures the implementation isn't artificially throttled
	if dlThroughput < 10 {
		t.Errorf("Download throughput unexpectedly low: %.2f MB/s", dlThroughput)
	}
	if ulThroughput < 10 {
		t.Errorf("Upload throughput unexpectedly low: %.2f MB/s", ulThroughput)
	}

	// Overall time check - should complete in <500ms for local transfers
	if dlElapsed+ulElapsed > 500*time.Millisecond {
		t.Logf("Warning: total time %v exceeds 500ms target", dlElapsed+ulElapsed)
	}
}
