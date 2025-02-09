package main

// Test generated using Keploy
import (
    "context"
    "testing"
)

func TestRunChat_NilClient(t *testing.T) {
    ctx := context.Background()

    err := runChat(ctx, nil)
    if err == nil {
        t.Errorf("Expected error when client is nil, got nil")
    }
}
