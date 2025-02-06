package main

import (
    "context"
    "fmt"
    "log"

    "github.com/ollama/ollama/api"
)

// Refactored to inject the client dependency for better testability.
func main() {
    client, err := api.ClientFromEnvironment()
    if err != nil {
        log.Fatal(err)
    }

    runPullProcess(client)
}

func runPullProcess(client *api.Client) {
    ctx := context.Background()

    req := &api.PullRequest{
        Model: "mistral",
    }
    progressFunc := func(resp api.ProgressResponse) error {
        fmt.Printf("Progress: status=%v, total=%v, completed=%v\n", resp.Status, resp.Total, resp.Completed)
        return nil
    }

    err := client.Pull(ctx, req, progressFunc)
    if err != nil {
        log.Fatal(err)
    }
}
