package lifecycle

import (
    "context"
    "fmt"
    "testing"
    "github.com/ollama/ollama/api"
    "os"
    "os/exec"
    "time"
)


// Test generated using Keploy
func TestIsServerRunning_ClientFromEnvironmentError(t *testing.T) {
    ctx := context.Background()

    // Save original clientFromEnvironment
    originalClientFromEnvironment := clientFromEnvironment
    defer func() { clientFromEnvironment = originalClientFromEnvironment }()

    // Mock clientFromEnvironment to return an error
    clientFromEnvironment = func() (*api.Client, error) {
        return nil, fmt.Errorf("mock error")
    }

    running := IsServerRunning(ctx)
    if running {
        t.Error("Expected IsServerRunning to return false when clientFromEnvironment returns an error")
    }
}

// Test generated using Keploy
func TestSpawnServer_ContextCancel(t *testing.T) {
    ctx, cancel := context.WithCancel(context.Background())

    // Save original startFunc
    originalStartFunc := startFunc
    defer func() { startFunc = originalStartFunc }()

    // Mock startFunc to return a command that waits on context cancellation
    startFunc = func(ctx context.Context, command string) (*exec.Cmd, error) {
        cmd := &exec.Cmd{}
        cmd.Process = &os.Process{Pid: os.Getpid()}
        cmd.ProcessState = &os.ProcessState{}
        return cmd, nil
    }

    done, err := SpawnServer(ctx, "mock-command")
    if err != nil {
        t.Fatalf("SpawnServer returned an error: %v", err)
    }

    // Cancel the context to simulate shutdown
    cancel()

    select {
    case exitCode := <-done:
        if exitCode != 0 {
            t.Errorf("Expected exit code 0, got %d", exitCode)
        }
    case <-time.After(2 * time.Second):
        t.Error("Timeout waiting for server to shut down")
    }
}

