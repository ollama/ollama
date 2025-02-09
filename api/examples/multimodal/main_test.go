package main

import (
    "testing"
    "fmt"
    "context"
    "github.com/ollama/ollama/api"
)


// Test generated using Keploy
func TestRun_FileNotFound(t *testing.T) {
    originalReadFile := readFile
    defer func() { readFile = originalReadFile }()

    readFile = func(name string) ([]byte, error) {
        return nil, fmt.Errorf("file not found")
    }

    err := run([]string{"program", "missingfile.jpg"})
    if err == nil {
        t.Fatal("Expected error when file cannot be read")
    }
    if err.Error() != "file not found" {
        t.Fatalf("Expected error 'file not found', got '%s'", err.Error())
    }
}

// Test generated using Keploy
func TestRun_ClientFromEnvironmentError(t *testing.T) {
    originalClientFromEnvironment := clientFromEnvironment
    defer func() { clientFromEnvironment = originalClientFromEnvironment }()

    clientFromEnvironment = func() (APIClient, error) {
        return nil, fmt.Errorf("client error")
    }

    originalReadFile := readFile
    defer func() { readFile = originalReadFile }()

    readFile = func(name string) ([]byte, error) {
        return []byte("image data"), nil
    }

    err := run([]string{"program", "image.jpg"})
    if err == nil {
        t.Fatal("Expected error when clientFromEnvironment fails")
    }
    if err.Error() != "client error" {
        t.Fatalf("Expected error 'client error', got '%s'", err.Error())
    }
}


// Test generated using Keploy
func TestRun_NoArguments(t *testing.T) {
    err := run([]string{"program"})
    if err == nil {
        t.Fatal("Expected error when no arguments are provided")
    }
    expectedError := "usage: <image name>"
    if err.Error() != expectedError {
        t.Fatalf("Expected error '%s', got '%s'", expectedError, err.Error())
    }
}


// Test generated using Keploy
type mockAPIClient struct {
    GenerateFunc func(ctx context.Context, req *api.GenerateRequest, respFunc api.GenerateResponseFunc) error
}

func (m *mockAPIClient) Generate(ctx context.Context, req *api.GenerateRequest, respFunc api.GenerateResponseFunc) error {
    return m.GenerateFunc(ctx, req, respFunc)
}

func TestRun_ClientGenerateError(t *testing.T) {
    originalClientFromEnvironment := clientFromEnvironment
    defer func() { clientFromEnvironment = originalClientFromEnvironment }()

    mockClient := &mockAPIClient{
        GenerateFunc: func(ctx context.Context, req *api.GenerateRequest, respFunc api.GenerateResponseFunc) error {
            return fmt.Errorf("generate error")
        },
    }

    clientFromEnvironment = func() (APIClient, error) {
        return mockClient, nil
    }

    originalReadFile := readFile
    defer func() { readFile = originalReadFile }()

    readFile = func(name string) ([]byte, error) {
        return []byte("image data"), nil
    }

    err := run([]string{"program", "image.jpg"})
    if err == nil {
        t.Fatal("Expected error when Generate fails")
    }
    if err.Error() != "generate error" {
        t.Fatalf("Expected error 'generate error', got '%s'", err.Error())
    }
}

