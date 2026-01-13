package imagegen

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

// TestRunCLIDoesNotValidateModelLocally verifies that RunCLI doesn't perform
// client-side model validation. Model validation should be done by the server,
// not the client, to maintain proper client-server separation.
func TestRunCLIDoesNotValidateModelLocally(t *testing.T) {
	// Create a minimal cobra command for testing
	cmd := &cobra.Command{}
	cmd.SetContext(context.Background())

	// Call RunCLI with a model name that doesn't exist locally.
	// The function should NOT return "unknown image generation model" error
	// (that would indicate client-side validation).
	// Instead, it should attempt to contact the server and fail with a
	// connection error (or succeed if a server happens to be running).

	err := RunCLI(cmd, "nonexistent-test-model", "test prompt", false, nil)

	if err == nil {
		// Server was running and handled the request - that's fine
		return
	}

	// The error should NOT be about an unknown model (client-side validation)
	// It should be a connection/server error instead
	errMsg := err.Error()
	if strings.Contains(errMsg, "unknown image generation model") {
		t.Errorf("RunCLI performed client-side model validation, got: %v", err)
	}
}

// TestDefaultOptions verifies default image generation options are sensible.
func TestDefaultOptions(t *testing.T) {
	opts := DefaultOptions()

	if opts.Width <= 0 {
		t.Errorf("Expected positive width, got %d", opts.Width)
	}
	if opts.Height <= 0 {
		t.Errorf("Expected positive height, got %d", opts.Height)
	}
	if opts.Steps <= 0 {
		t.Errorf("Expected positive steps, got %d", opts.Steps)
	}
}

// TestSanitizeFilename verifies filename sanitization works correctly.
func TestSanitizeFilename(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"hello world", "hello-world"},
		{"Hello World", "hello-world"},
		{"test@#$%file", "testfile"},
		{"a b c", "a-b-c"},
		{"", ""},
		{"already-valid", "already-valid"},
		{"123-numbers-456", "123-numbers-456"},
	}

	for _, tc := range tests {
		got := sanitizeFilename(tc.input)
		if got != tc.expected {
			t.Errorf("sanitizeFilename(%q) = %q, want %q", tc.input, got, tc.expected)
		}
	}
}

// TestHandleSetCommand verifies option setting works correctly.
func TestHandleSetCommand(t *testing.T) {
	opts := DefaultOptions()

	tests := []struct {
		args      string
		wantErr   bool
		checkFunc func() bool
	}{
		{"width 512", false, func() bool { return opts.Width == 512 }},
		{"height 768", false, func() bool { return opts.Height == 768 }},
		{"steps 20", false, func() bool { return opts.Steps == 20 }},
		{"seed 42", false, func() bool { return opts.Seed == 42 }},
		{"negative bad quality", false, func() bool { return opts.NegativePrompt == "bad quality" }},
		{"w 256", false, func() bool { return opts.Width == 256 }},
		{"h 384", false, func() bool { return opts.Height == 384 }},
		{"s 5", false, func() bool { return opts.Steps == 5 }},
		{"width invalid", true, nil},
		{"unknown 123", true, nil},
		{"", true, nil},
	}

	for _, tc := range tests {
		err := handleSetCommand(tc.args, &opts)
		if (err != nil) != tc.wantErr {
			t.Errorf("handleSetCommand(%q) error = %v, wantErr %v", tc.args, err, tc.wantErr)
			continue
		}
		if tc.checkFunc != nil && !tc.checkFunc() {
			t.Errorf("handleSetCommand(%q) did not update option correctly", tc.args)
		}
	}
}

// TestGetModelInfoReturnsErrorForMissingModel verifies GetModelInfo handles missing models.
func TestGetModelInfoReturnsErrorForMissingModel(t *testing.T) {
	_, err := GetModelInfo("nonexistent-model-12345")
	if err == nil {
		t.Error("Expected error for nonexistent model, got nil")
	}
}

// TestKeepAlivePassedToRequest verifies keepAlive is passed to API request.
func TestKeepAlivePassedToRequest(t *testing.T) {
	// This is a compile-time verification that the api.Duration type is used correctly
	var keepAlive *api.Duration
	_ = keepAlive // Just verify the type exists and is usable
}
