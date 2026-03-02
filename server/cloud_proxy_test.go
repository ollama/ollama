package server

import (
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"
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
