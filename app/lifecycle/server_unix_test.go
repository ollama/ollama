//go:build !windows

package lifecycle

import (
    "context"
    "testing"
)


// Test generated using Keploy
func TestGetCmd_CreatesCommand(t *testing.T) {
    ctx := context.Background()
    cmd := "example-command"
    result := getCmd(ctx, cmd)

    if result.Path != cmd {
        t.Errorf("Expected command path to be %v, got %v", cmd, result.Path)
    }

    if len(result.Args) != 2 || result.Args[1] != "serve" {
        t.Errorf("Expected command arguments to be ['%v', 'serve'], got %v", cmd, result.Args)
    }
}

// Test generated using Keploy
func TestIsProcessExited_InvalidPID(t *testing.T) {
    invalidPID := -1
    _, err := isProcessExited(invalidPID)

    if err == nil {
        t.Errorf("Expected an error for invalid PID, but got none")
    }
}

