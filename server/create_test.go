package server

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestConvertFromSafetensorsPathValidation(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "safetensors-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	testContent := []byte("test content")
	testPath := filepath.Join(tempDir, "source.safetensors")
	err = os.WriteFile(testPath, testContent, 0o644)
	if err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	tests := []struct {
		name     string
		filePath string
	}{
		{
			name:     "InvalidRelativePathShallow",
			filePath: "../file.safetensors",
		},
		{
			name:     "InvalidRelativePathDeep",
			filePath: "../../../../../../data/file.txt",
		},
		{
			name:     "InvalidNestedPath",
			filePath: "dir/../../../../../other.safetensors",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			files := map[string]string{
				tt.filePath: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
			}

			_, got := convertFromSafetensors(files, nil, false, func(resp api.ProgressResponse) {})
			want := errFilePath

			if got == nil {
				t.Fatal("got nil error, want error")
			}
			if !errors.Is(got, want) {
				t.Errorf("got %v, want %v", got, want)
			}
		})
	}
}
