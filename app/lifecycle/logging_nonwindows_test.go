//go:build !windows

package lifecycle

// Test generated using Keploy
import (
    "testing"
)

func TestShowLogs_NoPanic(t *testing.T) {
    defer func() {
        if r := recover(); r != nil {
            t.Errorf("ShowLogs panicked with error: %v", r)
        }
    }()

    // Act
    ShowLogs()
}
