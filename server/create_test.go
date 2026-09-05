package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/manifest"
)

func TestSplitGGUFTensorReaderRejectsShortSource(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tensor.bin")
	if err := os.WriteFile(path, []byte{1, 2, 3}, 0o600); err != nil {
		t.Fatal(err)
	}

	var dst bytes.Buffer
	n, err := (splitGGUFTensorReader{ctx: context.Background(), path: path, size: 4}).WriteTo(&dst)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("WriteTo() error = %v, want io.EOF", err)
	}
	if n != 3 {
		t.Fatalf("WriteTo() bytes = %d, want 3", n)
	}
}

type cancelAfterWrite struct {
	cancel context.CancelFunc
}

func (w cancelAfterWrite) Write(p []byte) (int, error) {
	w.cancel()
	return len(p), nil
}

func TestSplitGGUFTensorReaderStopsAfterCancellation(t *testing.T) {
	const size = 128 << 10
	path := filepath.Join(t.TempDir(), "tensor.bin")
	if err := os.WriteFile(path, make([]byte, size), 0o600); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	n, err := (splitGGUFTensorReader{ctx: ctx, path: path, size: size}).WriteTo(cancelAfterWrite{cancel: cancel})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("WriteTo() error = %v, want context.Canceled", err)
	}
	if n <= 0 || n >= size {
		t.Fatalf("WriteTo() bytes = %d, want partial copy", n)
	}
}

func TestValidateCreateFilePath(t *testing.T) {
	tests := []struct {
		name string
		path string
		want bool
	}{
		{name: "file", path: "model.safetensors", want: true},
		{name: "nested", path: "weights/model.safetensors", want: true},
		{name: "empty", path: ""},
		{name: "dot", path: "."},
		{name: "dot dot", path: ".."},
		{name: "trailing separator", path: "weights/"},
		{name: "absolute", path: "/model.safetensors"},
		{name: "drive relative", path: "c:model.safetensors"},
		{name: "unc", path: `\\server\share\model.safetensors`},
		{name: "device path", path: `\\?\c:\model.safetensors`},
		{name: "mixed separators", path: `weights\model.safetensors`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateCreateFilePath(tt.path)
			if got := err == nil; got != tt.want {
				t.Fatalf("validateCreateFilePath(%q) success = %v, want %v (error: %v)", tt.path, got, tt.want, err)
			}
		})
	}
}

func TestValidateCreateFilesRejectsTooManyFiles(t *testing.T) {
	files := make(map[string]string, maxCreateFiles+1)
	digest := "sha256:" + strings.Repeat("0", 64)
	for i := range maxCreateFiles + 1 {
		files[fmt.Sprintf("file-%04d.json", i)] = digest
	}

	err := validateCreateFiles(files)
	if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("exceeds maximum %d", maxCreateFiles)) {
		t.Fatalf("validateCreateFiles() error = %v, want file count limit", err)
	}
}

func TestStageSafetensorsSourceFilesRejectsOversizedMetadata(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	digest := "sha256:" + strings.Repeat("0", 64)
	blobPath, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(blobPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(maxSafetensorsMetadataSize + 1); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	dir, cleanup, err := stageSafetensorsSourceFiles(t.Context(), map[string]string{"config.json": digest})
	if cleanup != nil {
		cleanup()
	}
	if err == nil {
		t.Fatalf("stageSafetensorsSourceFiles() = %q, nil, want size error", dir)
	}
	if !strings.Contains(err.Error(), "exceeds maximum") {
		t.Fatalf("stageSafetensorsSourceFiles() error = %v, want size error", err)
	}
}

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
