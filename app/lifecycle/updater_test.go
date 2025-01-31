package lifecycle

import (
    "context"
    "testing"
    "time"
)


// Test generated using Keploy
func TestStartBackgroundUpdaterChecker_ContextCancel(t *testing.T) {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    callbackCalled := false
    cb := func(version string) error {
        callbackCalled = true
        return nil
    }

    go StartBackgroundUpdaterChecker(ctx, cb)

    // Cancel the context and wait briefly to ensure the goroutine stops
    cancel()
    time.Sleep(100 * time.Millisecond)

    if callbackCalled {
        t.Errorf("Expected callback not to be called after context cancellation")
    }
}
