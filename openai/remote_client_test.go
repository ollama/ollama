package openai

import (
	"net/url"
	"testing"
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
