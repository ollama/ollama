package server

import (
	"errors"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRecoverCreatePanic(t *testing.T) {
	var sent any
	func() {
		defer recoverCreatePanic(func(resp any) bool {
			sent = resp
			return true
		})

		panic("boom")
	}()

	h, ok := sent.(gin.H)
	if !ok {
		t.Fatalf("sent response type = %T, want gin.H", sent)
	}

	if got, want := h["error"], "internal server error"; got != want {
		t.Fatalf("sent error = %q, want %q", got, want)
	}
}

func TestRecoverCreatePanicNoPanic(t *testing.T) {
	called := false
	func() {
		defer recoverCreatePanic(func(resp any) bool {
			called = true
			return true
		})
	}()

	if called {
		t.Fatal("recoverCreatePanic sent a response without a panic")
	}
}

func TestRemoteURL(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
		hasError bool
	}{
		{
			name:     "absolute path",
			input:    "/foo/bar",
			expected: "http://localhost:11434/foo/bar",
			hasError: false,
		},
		{
			name:     "absolute path with cleanup",
			input:    "/foo/../bar",
			expected: "http://localhost:11434/bar",
			hasError: false,
		},
		{
			name:     "root path",
			input:    "/",
			expected: "http://localhost:11434/",
			hasError: false,
		},
		{
			name:     "host without scheme",
			input:    "example.com",
			expected: "http://example.com:11434",
			hasError: false,
		},
		{
			name:     "host with port",
			input:    "example.com:8080",
			expected: "http://example.com:8080",
			hasError: false,
		},
		{
			name:     "full URL",
			input:    "https://example.com:8080/path",
			expected: "https://example.com:8080/path",
			hasError: false,
		},
		{
			name:     "full URL with path cleanup",
			input:    "https://example.com:8080/path/../other",
			expected: "https://example.com:8080/other",
			hasError: false,
		},
		{
			name:     "ollama.com special case",
			input:    "ollama.com",
			expected: "https://ollama.com:443",
			hasError: false,
		},
		{
			name:     "http ollama.com special case",
			input:    "http://ollama.com",
			expected: "https://ollama.com:443",
			hasError: false,
		},
		{
			name:     "URL with only host",
			input:    "http://example.com",
			expected: "http://example.com:11434",
			hasError: false,
		},
		{
			name:     "URL with root path cleaned",
			input:    "http://example.com/",
			expected: "http://example.com:11434",
			hasError: false,
		},
		{
			name:     "invalid URL",
			input:    "http://[::1]:namedport", // invalid port
			expected: "",
			hasError: true,
		},
		{
			name:     "empty string",
			input:    "",
			expected: "http://localhost:11434",
			hasError: false,
		},
		{
			name:     "host with scheme but no port",
			input:    "http://localhost",
			expected: "http://localhost:11434",
			hasError: false,
		},
		{
			name:     "complex path cleanup",
			input:    "/a/b/../../c/./d",
			expected: "http://localhost:11434/c/d",
			hasError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := remoteURL(tt.input)

			if tt.hasError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestRemoteURL_Idempotent(t *testing.T) {
	// Test that applying remoteURL twice gives the same result as applying it once
	testInputs := []string{
		"/foo/bar",
		"example.com",
		"https://example.com:8080/path",
		"ollama.com",
		"http://localhost:11434",
	}

	for _, input := range testInputs {
		t.Run(input, func(t *testing.T) {
			firstResult, err := remoteURL(input)
			if err != nil {
				t.Fatalf("first call failed: %v", err)
			}

			secondResult, err := remoteURL(firstResult)
			if err != nil {
				t.Fatalf("second call failed: %v", err)
			}

			if firstResult != secondResult {
				t.Errorf("function is not idempotent: first=%q, second=%q", firstResult, secondResult)
			}
		})
	}
}

func TestSetTemplate(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	t.Run("valid template", func(t *testing.T) {
		layers, err := setTemplate(nil, "{{ .Prompt }}")
		if err != nil {
			t.Fatalf("setTemplate returned error for valid template: %v", err)
		}

		if len(layers) != 1 {
			t.Fatalf("expected 1 layer, got %d", len(layers))
		}

		if got, want := layers[0].MediaType, "application/vnd.ollama.image.template"; got != want {
			t.Fatalf("unexpected media type: got %q, want %q", got, want)
		}
	})

	t.Run("invalid template", func(t *testing.T) {
		_, err := setTemplate(nil, "{{ if .Prompt }}")
		if err == nil {
			t.Fatal("expected error for invalid template, got nil")
		}

		if !errors.Is(err, errBadTemplate) {
			t.Fatalf("expected errBadTemplate, got %v", err)
		}
	})
}
