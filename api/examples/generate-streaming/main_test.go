package main

import (
    "bytes"
    "context"
    "io"
    "os"
    "testing"
    "github.com/ollama/ollama/api"
)


// Test generated using Keploy
func TestRun_Success(t *testing.T) {
    // Mock APIClient
    mockClient := &MockClient{
        GenerateFunc: func(ctx context.Context, req *api.GenerateRequest, respFunc api.GenerateResponseFunc) error {
            // Simulate the GenerateResponse
            resp := api.GenerateResponse{
                Response: "There are eight planets in the solar system.",
            }
            // Call the response function with the simulated response
            return respFunc(resp)
        },
    }

    // Capture stdout
    oldStdout := os.Stdout
    r, w, _ := os.Pipe()
    os.Stdout = w

    // Run the function
    err := run(mockClient)
    if err != nil {
        t.Fatalf("Expected no error, got %v", err)
    }

    // Restore stdout
    w.Close()
    os.Stdout = oldStdout

    // Read captured output
    var buf bytes.Buffer
    io.Copy(&buf, r)
    output := buf.String()

    expectedOutput := "There are eight planets in the solar system.\n"
    if output != expectedOutput {
        t.Errorf("Expected output %q, got %q", expectedOutput, output)
    }
}

// MockClient implements the APIClient interface for testing purposes
type MockClient struct {
    GenerateFunc func(ctx context.Context, req *api.GenerateRequest, respFunc api.GenerateResponseFunc) error
}

func (m *MockClient) Generate(ctx context.Context, req *api.GenerateRequest, respFunc api.GenerateResponseFunc) error {
    return m.GenerateFunc(ctx, req, respFunc)
}
