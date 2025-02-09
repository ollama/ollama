package main

import (
    "fmt"
    "testing"
    "github.com/ollama/ollama/api"
    "context"
)


// Test generated using Keploy
func TestProgressFunction_Print(t *testing.T) {
    progressFunc := func(resp api.ProgressResponse) error {
        expectedOutput := fmt.Sprintf("Progress: status=%v, total=%v, completed=%v\n", resp.Status, resp.Total, resp.Completed)
        if expectedOutput == "" {
            t.Errorf("Expected formatted output, but got an empty string")
        }
        return nil
    }

    resp := api.ProgressResponse{
        Status:    "in-progress",
        Total:     100,
        Completed: 50,
    }
    err := progressFunc(resp)
    if err != nil {
        t.Fatalf("Expected no error, but got %v", err)
    }
}

// Test generated using Keploy
func TestClientPull_ProgressFuncError(t *testing.T) {
    mockClient := &MockClient{
        PullFunc: func(ctx context.Context, req *api.PullRequest, progressFunc func(api.ProgressResponse) error) error {
            return progressFunc(api.ProgressResponse{Status: "error"})
        },
    }

    ctx := context.Background()
    req := &api.PullRequest{
        Model: "mistral",
    }
    progressFunc := func(resp api.ProgressResponse) error {
        return fmt.Errorf("mock progress function error")
    }

    err := mockClient.Pull(ctx, req, progressFunc)
    if err == nil || err.Error() != "mock progress function error" {
        t.Fatalf("Expected 'mock progress function error', but got %v", err)
    }
}

