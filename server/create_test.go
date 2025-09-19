package server

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestConvertFromSafetensors(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	// Helper function to create a new layer and return its digest
	makeTemp := func(content string) string {
		l, err := NewLayer(strings.NewReader(content), "application/octet-stream")
		if err != nil {
			t.Fatalf("Failed to create layer: %v", err)
		}
		return l.Digest
	}

	// Create a safetensors compatible file with empty JSON content
	var buf bytes.Buffer
	headerSize := int64(len("{}"))
	binary.Write(&buf, binary.LittleEndian, headerSize)
	buf.WriteString("{}")

	model := makeTemp(buf.String())
	config := makeTemp(`{
		"architectures": ["LlamaForCausalLM"], 
		"vocab_size": 32000
	}`)
	tokenizer := makeTemp(`{
		"version": "1.0",
		"truncation": null,
		"padding": null,
		"added_tokens": [
			{
				"id": 0,
				"content": "<|endoftext|>",
				"single_word": false,
				"lstrip": false,
				"rstrip": false,
				"normalized": false,
				"special": true
			}
		]
	}`)

	tests := []struct {
		name     string
		filePath string
		wantErr  error
	}{
		// Invalid
		{
			name:     "InvalidRelativePathShallow",
			filePath: filepath.Join("..", "file.safetensors"),
			wantErr:  errFilePath,
		},
		{
			name:     "InvalidRelativePathDeep",
			filePath: filepath.Join("..", "..", "..", "..", "..", "..", "data", "file.txt"),
			wantErr:  errFilePath,
		},
		{
			name:     "InvalidNestedPath",
			filePath: filepath.Join("dir", "..", "..", "..", "..", "..", "other.safetensors"),
			wantErr:  errFilePath,
		},
		{
			name:     "AbsolutePathOutsideRoot",
			filePath: filepath.Join(os.TempDir(), "model.safetensors"),
			wantErr:  errFilePath, // Should fail since it's outside tmpDir
		},
		{
			name:     "ValidRelativePath",
			filePath: "model.safetensors",
			wantErr:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create the minimum required file map for convertFromSafetensors
			files := map[string]string{
				tt.filePath:      model,
				"config.json":    config,
				"tokenizer.json": tokenizer,
			}

			_, err := convertFromSafetensors(files, nil, false, func(resp api.ProgressResponse) {})

			if (tt.wantErr == nil && err != nil) ||
				(tt.wantErr != nil && err == nil) ||
				(tt.wantErr != nil && !errors.Is(err, tt.wantErr)) {
				t.Errorf("convertFromSafetensors() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
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
