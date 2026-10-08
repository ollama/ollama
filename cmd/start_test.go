//go:build darwin || windows

package cmd

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

type startWaitRoundTripper func(*http.Request) (*http.Response, error)

func (f startWaitRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestWaitForServerReturnsWhenContextIsCanceled(t *testing.T) {
	client := api.NewClient(
		&url.URL{Scheme: "http", Host: "127.0.0.1:11434"},
		&http.Client{Transport: startWaitRoundTripper(func(r *http.Request) (*http.Response, error) {
			return nil, r.Context().Err()
		})},
	)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	start := time.Now()
	err := waitForServer(ctx, client)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("waitForServer error = %v, want context.Canceled", err)
	}
	if elapsed := time.Since(start); elapsed >= 250*time.Millisecond {
		t.Fatalf("waitForServer took %s to observe cancellation", elapsed)
	}
}
