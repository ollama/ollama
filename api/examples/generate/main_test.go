package main

import (
    "context"
    "testing"
    "github.com/ollama/ollama/api"
)


// Test generated using Keploy
func TestGenerate_SuccessfulResponse(t *testing.T) {
    mockClient := &MockClient{}
    mockClient.GenerateFunc = func(ctx context.Context, req *api.GenerateRequest, respFunc func(api.GenerateResponse) error) error {
        response := api.GenerateResponse{
            Response: "There are 8 planets.",
        }
        return respFunc(response)
    }

    req := &api.GenerateRequest{
        Model:  "gemma2",
        Prompt: "how many planets are there?",
        Stream: new(bool),
    }

    ctx := context.Background()
    respFunc := func(resp api.GenerateResponse) error {
        if resp.Response != "There are 8 planets." {
            t.Errorf("Expected response 'There are 8 planets.', got '%s'", resp.Response)
        }
        return nil
    }

    err := mockClient.Generate(ctx, req, respFunc)
    if err != nil {
        t.Errorf("Expected no error, got %v", err)
    }
}
