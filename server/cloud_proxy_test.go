package server

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"
)

func TestCopyProxyRequestHeaders_StripsConnectionTokenHeaders(t *testing.T) {
	src := http.Header{}
	src.Add("Connection", "keep-alive, X-Trace-Hop, x-alt-hop")
	src.Add("X-Trace-Hop", "drop-me")
	src.Add("X-Alt-Hop", "drop-me-too")
	src.Add("Keep-Alive", "timeout=5")
	src.Add("X-End-To-End", "keep-me")

	dst := http.Header{}
	copyProxyRequestHeaders(dst, src)

	if got := dst.Get("Connection"); got != "" {
		t.Fatalf("expected Connection to be stripped, got %q", got)
	}
	if got := dst.Get("Keep-Alive"); got != "" {
		t.Fatalf("expected Keep-Alive to be stripped, got %q", got)
	}
	if got := dst.Get("X-Trace-Hop"); got != "" {
		t.Fatalf("expected X-Trace-Hop to be stripped via Connection token, got %q", got)
	}
	if got := dst.Get("X-Alt-Hop"); got != "" {
		t.Fatalf("expected X-Alt-Hop to be stripped via Connection token, got %q", got)
	}
	if got := dst.Get("X-End-To-End"); got != "keep-me" {
		t.Fatalf("expected X-End-To-End to be forwarded, got %q", got)
	}
}

func TestCopyProxyResponseHeaders_StripsConnectionTokenHeaders(t *testing.T) {
	src := http.Header{}
	src.Add("Connection", "X-Upstream-Hop")
	src.Add("X-Upstream-Hop", "drop-me")
	src.Add("Content-Type", "application/json")
	src.Add("X-Server-Trace", "keep-me")

	dst := http.Header{}
	copyProxyResponseHeaders(dst, src)

	if got := dst.Get("Connection"); got != "" {
		t.Fatalf("expected Connection to be stripped, got %q", got)
	}
	if got := dst.Get("X-Upstream-Hop"); got != "" {
		t.Fatalf("expected X-Upstream-Hop to be stripped via Connection token, got %q", got)
	}
	if got := dst.Get("Content-Type"); got != "application/json" {
		t.Fatalf("expected Content-Type to be forwarded, got %q", got)
	}
	if got := dst.Get("X-Server-Trace"); got != "keep-me" {
		t.Fatalf("expected X-Server-Trace to be forwarded, got %q", got)
	}
}

func TestResolveCloudProxyBaseURL_Default(t *testing.T) {
	baseURL, signingHost, overridden, err := resolveCloudProxyBaseURL("", gin.ReleaseMode)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if overridden {
		t.Fatal("expected override=false for empty input")
	}
	if baseURL != defaultCloudProxyBaseURL {
		t.Fatalf("expected default base URL %q, got %q", defaultCloudProxyBaseURL, baseURL)
	}
	if signingHost != defaultCloudProxySigningHost {
		t.Fatalf("expected default signing host %q, got %q", defaultCloudProxySigningHost, signingHost)
	}
}

func TestResolveCloudProxyBaseURL_ReleaseAllowsLoopback(t *testing.T) {
	baseURL, signingHost, overridden, err := resolveCloudProxyBaseURL("http://localhost:8080", gin.ReleaseMode)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !overridden {
		t.Fatal("expected override=true")
	}
	if baseURL != "http://localhost:8080" {
		t.Fatalf("unexpected base URL: %q", baseURL)
	}
	if signingHost != "localhost" {
		t.Fatalf("unexpected signing host: %q", signingHost)
	}
}

func TestResolveCloudProxyBaseURL_ReleaseRejectsNonLoopback(t *testing.T) {
	_, _, _, err := resolveCloudProxyBaseURL("https://example.com", gin.ReleaseMode)
	if err == nil {
		t.Fatal("expected error for non-loopback override in release mode")
	}
}

func TestResolveCloudProxyBaseURL_DevAllowsNonLoopbackHTTPS(t *testing.T) {
	baseURL, signingHost, overridden, err := resolveCloudProxyBaseURL("https://example.com:8443", gin.DebugMode)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !overridden {
		t.Fatal("expected override=true")
	}
	if baseURL != "https://example.com:8443" {
		t.Fatalf("unexpected base URL: %q", baseURL)
	}
	if signingHost != "example.com" {
		t.Fatalf("unexpected signing host: %q", signingHost)
	}
}

func TestResolveCloudProxyBaseURL_DevRejectsNonLoopbackHTTP(t *testing.T) {
	_, _, _, err := resolveCloudProxyBaseURL("http://example.com", gin.DebugMode)
	if err == nil {
		t.Fatal("expected error for non-loopback http override in dev mode")
	}
}

func TestBuildCloudSignatureChallengeIncludesExistingQuery(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://ollama.com/v1/messages?beta=true&foo=bar", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}

	got := buildCloudSignatureChallenge(req, "123")
	want := "POST,/v1/messages?beta=true&foo=bar&ts=123"
	if got != want {
		t.Fatalf("challenge mismatch: got %q want %q", got, want)
	}
	if req.URL.RawQuery != "beta=true&foo=bar&ts=123" {
		t.Fatalf("unexpected signed query: %q", req.URL.RawQuery)
	}
}

func TestCloudPassthroughMiddleware_ZstdBody(t *testing.T) {
	gin.SetMode(gin.TestMode)

	plainBody := []byte(`{"model":"test-model:cloud","messages":[{"role":"user","content":"hi"}]}`)
	var compressed bytes.Buffer
	w, err := zstd.NewWriter(&compressed)
	if err != nil {
		t.Fatalf("zstd writer: %v", err)
	}
	if _, err := w.Write(plainBody); err != nil {
		t.Fatalf("zstd write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("zstd close: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(compressed.Bytes()))
	req.Header.Set("Content-Encoding", "zstd")
	rec := httptest.NewRecorder()

	// Track whether the middleware detected the cloud model by checking
	// if c.Next() was called (non-cloud path) vs c.Abort() (cloud path).
	nextCalled := false

	r := gin.New()
	r.POST("/v1/responses", cloudPassthroughMiddleware("test"), func(c *gin.Context) {
		nextCalled = true
		// Verify the body is decompressed and Content-Encoding is removed.
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		model, ok := extractModelField(body)
		if !ok {
			t.Fatal("expected to extract model from decompressed body")
		}
		if model != "test-model:cloud" {
			t.Fatalf("expected model %q, got %q", "test-model:cloud", model)
		}
		if enc := c.GetHeader("Content-Encoding"); enc != "" {
			t.Fatalf("expected Content-Encoding to be removed, got %q", enc)
		}
		c.Status(http.StatusOK)
	})
	r.ServeHTTP(rec, req)

	// The cloud passthrough middleware should detect the cloud model and
	// proxy (abort), so the next handler should NOT be called.
	// However, since there's no actual cloud server to proxy to, the
	// middleware will attempt to proxy and fail. We just verify it didn't
	// fall through to c.Next() due to failure to read the compressed body.
	if nextCalled {
		t.Fatal("expected cloud passthrough to detect cloud model from zstd body, but it fell through to next handler")
	}
}

func TestCloudPassthroughMiddleware_ZstdBodyTooLarge(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Create a body that exceeds the 20MB limit
	oversized := make([]byte, maxDecompressedBodySize+1024)
	for i := range oversized {
		oversized[i] = 'A'
	}

	var compressed bytes.Buffer
	w, err := zstd.NewWriter(&compressed)
	if err != nil {
		t.Fatalf("zstd writer: %v", err)
	}
	if _, err := w.Write(oversized); err != nil {
		t.Fatalf("zstd write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("zstd close: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(compressed.Bytes()))
	req.Header.Set("Content-Encoding", "zstd")
	rec := httptest.NewRecorder()

	r := gin.New()
	r.POST("/v1/responses", cloudPassthroughMiddleware("test"), func(c *gin.Context) {
		t.Fatal("handler should not be reached for oversized body")
	})
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", rec.Code)
	}
}

func TestBuildCloudSignatureChallengeOverwritesExistingTimestamp(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://ollama.com/v1/messages?beta=true&ts=999", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}

	got := buildCloudSignatureChallenge(req, "123")
	want := "POST,/v1/messages?beta=true&ts=123"
	if got != want {
		t.Fatalf("challenge mismatch: got %q want %q", got, want)
	}
	if req.URL.RawQuery != "beta=true&ts=123" {
		t.Fatalf("unexpected signed query: %q", req.URL.RawQuery)
	}
}

func TestJSONLFramingResponseWriter_SplitsCoalescedLines(t *testing.T) {
	rec := &chunkRecorder{header: http.Header{}}
	w := &jsonlFramingResponseWriter{ResponseWriter: rec}

	payload := []byte("{\"a\":1}\n{\"b\":2}\n")
	if n, err := w.Write(payload); err != nil {
		t.Fatalf("write failed: %v", err)
	} else if n != len(payload) {
		t.Fatalf("write byte count mismatch: got %d want %d", n, len(payload))
	}

	if err := w.FlushPending(); err != nil {
		t.Fatalf("FlushPending failed: %v", err)
	}

	if len(rec.chunks) != 2 {
		t.Fatalf("expected 2 framed writes, got %d", len(rec.chunks))
	}
	if got := string(rec.chunks[0]); got != `{"a":1}` {
		t.Fatalf("first chunk mismatch: got %q", got)
	}
	if got := string(rec.chunks[1]); got != `{"b":2}` {
		t.Fatalf("second chunk mismatch: got %q", got)
	}
}

func TestJSONLFramingResponseWriter_FlushPendingWritesTrailingLine(t *testing.T) {
	rec := &chunkRecorder{header: http.Header{}}
	w := &jsonlFramingResponseWriter{ResponseWriter: rec}

	if _, err := w.Write([]byte("{\"a\":1")); err != nil {
		t.Fatalf("write failed: %v", err)
	}
	if len(rec.chunks) != 0 {
		t.Fatalf("expected no writes before newline/flush, got %d", len(rec.chunks))
	}

	if err := w.FlushPending(); err != nil {
		t.Fatalf("FlushPending failed: %v", err)
	}
	if len(rec.chunks) != 1 {
		t.Fatalf("expected 1 write after FlushPending, got %d", len(rec.chunks))
	}
	if got := string(rec.chunks[0]); got != `{"a":1` {
		t.Fatalf("trailing chunk mismatch: got %q", got)
	}
}

type chunkRecorder struct {
	header http.Header
	status int
	chunks [][]byte
}

func (r *chunkRecorder) Header() http.Header {
	return r.header
}

func (r *chunkRecorder) WriteHeader(statusCode int) {
	r.status = statusCode
}

func (r *chunkRecorder) Write(p []byte) (int, error) {
	cp := append([]byte(nil), p...)
	r.chunks = append(r.chunks, cp)
	return len(p), nil
}
