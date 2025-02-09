package main

import (
    "testing"
)


// Test generated using Keploy
func TestMainFunctionCallsRunFunc(t *testing.T) {
    // Mock RunFunc
    mockRunCalled := false
    originalRunFunc := RunFunc
    RunFunc = func() {
        mockRunCalled = true
    }
    defer func() { RunFunc = originalRunFunc }()

    // Call the main function
    main()

    // Assert that RunFunc was called
    if !mockRunCalled {
        t.Error("Expected RunFunc to be called")
    }
}
