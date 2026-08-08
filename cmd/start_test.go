//go:build darwin || windows

package cmd

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestWaitForServerReturnsWhenContextIsCanceled(t *testing.T) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	done := make(chan error, 1)
	go func() {
		done <- waitForServer(ctx, client)
	}()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("waitForServer() error = %v, want context.Canceled", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("waitForServer did not return after context cancellation")
	}
}
