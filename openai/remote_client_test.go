package openai

import (
	"net/http"
	"net/url"
	"testing"
	"time"
)

func TestNewRemoteClient_DefensiveCopyHeaders(t *testing.T) {
	base, err := url.Parse("https://example.com/v1")
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}

	headers := map[string]string{"X-Test": "A"}
	c := NewRemoteClient(base, "sk-test", headers)

	// Mutate the input map after construction; client should not observe the change.
	headers["X-Test"] = "B"
	headers["X-New"] = "C"

	if got := c.headers["X-Test"]; got != "A" {
		t.Fatalf("expected defensive copy to keep original value 'A', got %q", got)
	}
	if _, ok := c.headers["X-New"]; ok {
		t.Fatalf("expected defensive copy to not contain new key added after construction")
	}
}

func TestNewRemoteClient_NilHeaders(t *testing.T) {
	base, err := url.Parse("https://example.com/v1")
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}

	c := NewRemoteClient(base, "sk-test", nil)
	if c.headers != nil {
		t.Fatalf("expected nil headers to remain nil, got %#v", c.headers)
	}
}

func TestNewRemoteClient_UsesPhaseTimeoutTransport(t *testing.T) {
	base, err := url.Parse("https://example.com/v1")
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}

	c := NewRemoteClient(base, "sk-test", nil)
	if c.http == nil {
		t.Fatalf("expected http client to be set")
	}
	if c.http == http.DefaultClient {
		t.Fatalf("expected NewRemoteClient to not use http.DefaultClient")
	}
	if c.http.Timeout != 0 {
		t.Fatalf("expected http.Client.Timeout to be 0 (overall lifetime controlled by context), got %v", c.http.Timeout)
	}

	tr, ok := c.http.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("expected Transport to be *http.Transport, got %T", c.http.Transport)
	}
	if tr.ResponseHeaderTimeout != 60*time.Second {
		t.Fatalf("expected ResponseHeaderTimeout 60s, got %v", tr.ResponseHeaderTimeout)
	}
	if tr.TLSHandshakeTimeout != 10*time.Second {
		t.Fatalf("expected TLSHandshakeTimeout 10s, got %v", tr.TLSHandshakeTimeout)
	}
	if tr.DialContext == nil {
		t.Fatalf("expected DialContext to be set")
	}
}
