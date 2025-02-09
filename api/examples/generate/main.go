package main

import (
    "context"
    "fmt"
    "log"

    "github.com/ollama/ollama/api"
)

type GenerateClient interface {
    Generate(ctx context.Context, req *api.GenerateRequest, respFunc func(api.GenerateResponse) error) error
}

func main() {
    client, err := api.ClientFromEnvironment()
    if err != nil {
        log.Fatal(err)
    }

    req := &api.GenerateRequest{
        Model:  "gemma2",
        Prompt: "how many planets are there?",
        Stream: new(bool),
    }

    ctx := context.Background()
    respFunc := func(resp api.GenerateResponse) error {
        fmt.Println(resp.Response)
        return nil
    }

    if err := client.Generate(ctx, req, respFunc); err != nil {
        log.Fatal(err)
    }
}

// MockClient is a mock implementation of the GenerateClient interface for testing purposes.
type MockClient struct {
    GenerateFunc func(ctx context.Context, req *api.GenerateRequest, respFunc func(api.GenerateResponse) error) error
}

func (m *MockClient) Generate(ctx context.Context, req *api.GenerateRequest, respFunc func(api.GenerateResponse) error) error {
    return m.GenerateFunc(ctx, req, respFunc)
}
