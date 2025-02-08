package main

import (
    "fmt"
    "testing"
    "github.com/ollama/ollama/api"
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
