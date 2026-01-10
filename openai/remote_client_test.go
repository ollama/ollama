package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
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

type errReadCloser struct{}

func (errReadCloser) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }
func (errReadCloser) Close() error             { return nil }

func TestStreamChatCompletion_ReadAllErrorIsReturned(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer ts.Close()

	base, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}

	c := NewRemoteClient(base, "sk-test", nil)
	// Swap transport to inject a response body that errors on Read.
	c.http.Transport = roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadRequest,
			Body:       errReadCloser{},
			Header:     make(http.Header),
			Request:    &http.Request{Method: http.MethodPost},
		}, nil
	})

	err = c.StreamChatCompletion(context.Background(), ChatCompletionRequest{}, func(ChatCompletionChunk) error { return nil })
	if err == nil {
		t.Fatalf("expected error")
	}
	if got := err.Error(); got == "" || !containsAll(got, []string{"upstream error (400)", "read response body"}) {
		t.Fatalf("expected error to mention status and read failure, got %q", got)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func containsAll(s string, subs []string) bool {
	for _, sub := range subs {
		if !strings.Contains(s, sub) {
			return false
		}
	}
	return true
}
