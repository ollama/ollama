package client

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
)

func init() {
	// Disable real sleeps in tests.
	backoffDuration = func(int) time.Duration { return 0 }
}

// newTestClient creates an api.Client pointing at the given test server.
func newTestClient(t *testing.T, s *httptest.Server) *api.Client {
	t.Helper()
	u, err := url.Parse(s.URL)
	if err != nil {
		t.Fatal(err)
	}
	return api.NewClient(u, s.Client())
}

func TestUploadBlob_Success(t *testing.T) {
	var headCalls, postCalls atomic.Int32

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			headCalls.Add(1)
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			postCalls.Add(1)
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer s.Close()

	client := newTestClient(t, s)
	var transferred atomic.Int64
	bar := progress.NewBar("test", 0, 100)

	data := strings.Repeat("x", 100)
	b := blob{
		name:   "test.weight",
		digest: "sha256:abc123",
		size:   int64(len(data)),
		reader: func() io.ReadCloser { return io.NopCloser(strings.NewReader(data)) },
	}

	var called bool
	err := uploadBlob(context.Background(), client, b, &transferred, bar, func() {
		called = true
	})
	if err != nil {
		t.Fatalf("uploadBlob failed: %v", err)
	}
	if !called {
		t.Error("onSuccess was not called")
	}
	if headCalls.Load() != 1 {
		t.Errorf("HEAD calls = %d, want 1", headCalls.Load())
	}
	if postCalls.Load() != 1 {
		t.Errorf("POST calls = %d, want 1", postCalls.Load())
	}
	if transferred.Load() != int64(len(data)) {
		t.Errorf("transferred = %d, want %d", transferred.Load(), len(data))
	}
}

func TestUploadBlob_AlreadyExists(t *testing.T) {
	var headCalls atomic.Int32

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			headCalls.Add(1)
			w.WriteHeader(http.StatusOK)
			return
		}
		t.Errorf("unexpected %s request — blob should exist", r.Method)
	}))
	defer s.Close()

	client := newTestClient(t, s)
	var transferred atomic.Int64
	bar := progress.NewBar("test", 0, 100)

	b := blob{
		name:   "test.weight",
		digest: "sha256:abc123",
		size:   50,
		reader: func() io.ReadCloser { return io.NopCloser(strings.NewReader("x")) },
	}

	var called bool
	err := uploadBlob(context.Background(), client, b, &transferred, bar, func() {
		called = true
	})
	if err != nil {
		t.Fatalf("uploadBlob failed: %v", err)
	}
	if !called {
		t.Error("onSuccess was not called")
	}
	if headCalls.Load() != 1 {
		t.Errorf("HEAD calls = %d, want 1", headCalls.Load())
	}
	// When blob exists, we credit b.size (not bytes read).
	if transferred.Load() != 50 {
		t.Errorf("transferred = %d, want 50", transferred.Load())
	}
}

func TestUploadBlob_RetryOnTransientFailure(t *testing.T) {
	var postCalls atomic.Int32

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			n := postCalls.Add(1)
			io.Copy(io.Discard, r.Body)
			if n <= 2 {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer s.Close()

	client := newTestClient(t, s)
	var transferred atomic.Int64
	bar := progress.NewBar("test", 0, 100)

	data := "hello world"
	b := blob{
		name:   "test.weight",
		digest: "sha256:abc123",
		size:   int64(len(data)),
		reader: func() io.ReadCloser { return io.NopCloser(strings.NewReader(data)) },
	}

	var called bool
	err := uploadBlob(context.Background(), client, b, &transferred, bar, func() {
		called = true
	})
	if err != nil {
		t.Fatalf("uploadBlob failed: %v", err)
	}
	if !called {
		t.Error("onSuccess was not called")
	}
	if postCalls.Load() != 3 {
		t.Errorf("POST calls = %d, want 3 (2 failures + 1 success)", postCalls.Load())
	}
	// Progress should reflect only the successful transfer after rollbacks.
	if transferred.Load() != int64(len(data)) {
		t.Errorf("transferred = %d, want %d", transferred.Load(), len(data))
	}
}

func TestUploadBlob_ProgressRollbackOnRetry(t *testing.T) {
	// Server reads partial data before failing, verifying progress rollback.
	var postCalls atomic.Int32

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			n := postCalls.Add(1)
			if n == 1 {
				// Read partial data then fail.
				buf := make([]byte, 5)
				io.ReadAtLeast(r.Body, buf, 5)
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer s.Close()

	client := newTestClient(t, s)
	var transferred atomic.Int64
	bar := progress.NewBar("test", 0, 100)

	data := "0123456789" // 10 bytes
	b := blob{
		name:   "test.weight",
		digest: "sha256:abc123",
		size:   int64(len(data)),
		reader: func() io.ReadCloser { return io.NopCloser(strings.NewReader(data)) },
	}

	err := uploadBlob(context.Background(), client, b, &transferred, bar, func() {})
	if err != nil {
		t.Fatalf("uploadBlob failed: %v", err)
	}

	// After successful retry, transferred should equal blob size exactly.
	if transferred.Load() != int64(len(data)) {
		t.Errorf("transferred = %d, want %d (rollback not working correctly)", transferred.Load(), len(data))
	}
}

func TestUploadBlob_ExhaustedRetries(t *testing.T) {
	var postCalls atomic.Int32

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			postCalls.Add(1)
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer s.Close()

	client := newTestClient(t, s)
	var transferred atomic.Int64
	bar := progress.NewBar("test", 0, 100)

	data := "x"
	b := blob{
		name:   "test.weight",
		digest: "sha256:abc123",
		size:   int64(len(data)),
		reader: func() io.ReadCloser { return io.NopCloser(strings.NewReader(data)) },
	}

	err := uploadBlob(context.Background(), client, b, &transferred, bar, func() {
		t.Error("onSuccess should not be called on failure")
	})
	if err == nil {
		t.Fatal("expected error after exhausted retries")
	}
	if !strings.Contains(err.Error(), fmt.Sprintf("after %d attempts", maxUploadRetries)) {
		t.Errorf("error should mention retry count, got: %v", err)
	}
	if postCalls.Load() != int32(maxUploadRetries) {
		t.Errorf("POST calls = %d, want %d", postCalls.Load(), maxUploadRetries)
	}
	// Progress should be zero after all rollbacks.
	if transferred.Load() != 0 {
		t.Errorf("transferred = %d, want 0 after all failures", transferred.Load())
	}
}

func TestUploadBlob_ContextCancellation(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer s.Close()

	client := newTestClient(t, s)
	var transferred atomic.Int64
	bar := progress.NewBar("test", 0, 100)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	b := blob{
		name:   "test.weight",
		digest: "sha256:abc123",
		size:   10,
		reader: func() io.ReadCloser { return io.NopCloser(strings.NewReader("x")) },
	}

	err := uploadBlob(ctx, client, b, &transferred, bar, func() {
		t.Error("onSuccess should not be called on cancellation")
	})
	if err == nil {
		t.Fatal("expected error on cancelled context")
	}
}

func TestUploadBlob_HeadFailRetry(t *testing.T) {
	// HEAD check fails transiently, then succeeds and finds blob exists.
	var headCalls atomic.Int32

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			n := headCalls.Add(1)
			if n == 1 {
				w.WriteHeader(http.StatusBadGateway)
				return
			}
			w.WriteHeader(http.StatusOK)
			return
		}
		t.Errorf("unexpected %s request — blob should exist", r.Method)
	}))
	defer s.Close()

	client := newTestClient(t, s)
	var transferred atomic.Int64
	bar := progress.NewBar("test", 0, 100)

	b := blob{
		name:   "test.weight",
		digest: "sha256:abc123",
		size:   42,
		reader: func() io.ReadCloser { return io.NopCloser(strings.NewReader("x")) },
	}

	var called bool
	err := uploadBlob(context.Background(), client, b, &transferred, bar, func() {
		called = true
	})
	if err != nil {
		t.Fatalf("uploadBlob failed: %v", err)
	}
	if !called {
		t.Error("onSuccess was not called")
	}
	if headCalls.Load() != 2 {
		t.Errorf("HEAD calls = %d, want 2 (1 failure + 1 success)", headCalls.Load())
	}
	// Blob already existed — credited b.size.
	if transferred.Load() != 42 {
		t.Errorf("transferred = %d, want 42", transferred.Load())
	}
}

func TestUploadBlobOnce_ProgressTracking(t *testing.T) {
	// Verify uploadBlobOnce tracks progress correctly for a successful upload.
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusCreated)
		}
	}))
	defer s.Close()

	client := newTestClient(t, s)
	var transferred atomic.Int64
	bar := progress.NewBar("test", 0, 1000)

	data := strings.Repeat("a", 256)
	b := blob{
		name:   "test.weight",
		digest: "sha256:abc123",
		size:   int64(len(data)),
		reader: func() io.ReadCloser { return io.NopCloser(strings.NewReader(data)) },
	}

	err := uploadBlobOnce(context.Background(), client, b, &transferred, bar)
	if err != nil {
		t.Fatalf("uploadBlobOnce failed: %v", err)
	}
	if transferred.Load() != int64(len(data)) {
		t.Errorf("transferred = %d, want %d", transferred.Load(), len(data))
	}
}

func TestUploadBlobOnce_RollbackOnFailure(t *testing.T) {
	// Verify progress is rolled back when upload fails.
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			// Read all data (progress gets incremented) then fail.
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer s.Close()

	client := newTestClient(t, s)

	// Start with some pre-existing progress from other blobs.
	var transferred atomic.Int64
	transferred.Store(500)
	bar := progress.NewBar("test", 0, 1000)

	data := strings.Repeat("b", 100)
	b := blob{
		name:   "test.weight",
		digest: "sha256:abc123",
		size:   int64(len(data)),
		reader: func() io.ReadCloser { return io.NopCloser(strings.NewReader(data)) },
	}

	err := uploadBlobOnce(context.Background(), client, b, &transferred, bar)
	if err == nil {
		t.Fatal("expected error on 500 response")
	}
	// Progress should be rolled back to pre-existing value.
	if transferred.Load() != 500 {
		t.Errorf("transferred = %d, want 500 (rollback failed)", transferred.Load())
	}
}
