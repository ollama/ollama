package main

import (
    "context"
    "fmt"
    "log"

    "github.com/ollama/ollama/api"
)

// Refactored main function to improve testability by injecting dependencies.
func main() {
    client, err := api.ClientFromEnvironment()
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()
    req := &api.PullRequest{
        Model: "mistral",
    }

    progressFunc := createProgressFunc()

    err = client.Pull(ctx, req, progressFunc)
    if err != nil {
        log.Fatal(err)
    }
}

// createProgressFunc is extracted to allow mocking and easier testing.
func createProgressFunc() func(api.ProgressResponse) error {
    return func(resp api.ProgressResponse) error {
        fmt.Printf("Progress: status=%v, total=%v, completed=%v\n", resp.Status, resp.Total, resp.Completed)
        return nil
    }
}

// MockClient is introduced for testing purposes.
type MockClient struct {
    PullFunc func(ctx context.Context, req *api.PullRequest, progressFunc func(api.ProgressResponse) error) error
}

func (m *MockClient) Pull(ctx context.Context, req *api.PullRequest, progressFunc func(api.ProgressResponse) error) error {
    return m.PullFunc(ctx, req, progressFunc)
}
