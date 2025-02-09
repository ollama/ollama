//go:build !windows

package lifecycle

// Test generated using Keploy
import (
    "testing"
    "errors"
)

func TestGetStarted_ReturnsNotImplementedError(t *testing.T) {
    err := GetStarted()
    expectedError := errors.New("not implemented")

    if err == nil {
        t.Fatalf("Expected an error, but got nil")
    }

    if err.Error() != expectedError.Error() {
        t.Errorf("Expected error message '%v', but got '%v'", expectedError.Error(), err.Error())
    }
}
