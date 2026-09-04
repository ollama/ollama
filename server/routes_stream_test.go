package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestSendStreamValueReturnsAfterContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	done := make(chan bool)
	go func() {
		done <- sendStreamValue(ctx, make(chan any), api.ProgressResponse{Status: "pulling"})
	}()

	select {
	case ok := <-done:
		if ok {
			t.Fatal("sendStreamValue succeeded after context cancellation")
		}
	case <-time.After(time.Second):
		t.Fatal("sendStreamValue blocked after context cancellation")
	}
}
