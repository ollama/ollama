package auth

import (
    "bytes"
    "testing"
)


// Test generated using Keploy
func TestNewNonce_HappyPath(t *testing.T) {
    length := 16
    r := bytes.NewReader([]byte("mock-random-data"))

    result, err := NewNonce(r, length)
    if err != nil {
        t.Fatalf("Expected no error, got %v", err)
    }
    if len(result) == 0 {
        t.Errorf("Expected a nonce, got an empty string")
    }
}

// Test generated using Keploy
func TestNewNonce_InsufficientData_Error(t *testing.T) {
    length := 16
    r := bytes.NewReader([]byte("short"))

    _, err := NewNonce(r, length)
    if err == nil {
        t.Fatalf("Expected an error due to insufficient data, got nil")
    }
}

