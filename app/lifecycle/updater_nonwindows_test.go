//go:build !windows
package lifecycle

// Test generated using Keploy
import (
    "testing"
    "context"
    "errors"
)

func TestDoUpgrade_NilCancelFunction(t *testing.T) {
    var cancel context.CancelFunc = nil
    done := make(chan int)
    defer close(done)

    err := DoUpgrade(cancel, done)
    expectedError := errors.New("not implemented")

    if err == nil || err.Error() != expectedError.Error() {
        t.Errorf("Expected error: %v, got: %v", expectedError, err)
    }
}
