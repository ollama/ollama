package convert

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"
)

func TestParseTorchResolvesFSPath(t *testing.T) {
	// Test with os.DirFS where files are outside the current working directory
	dir := t.TempDir()
	testFile := "pytorch_model.bin"
	if err := os.WriteFile(filepath.Join(dir, testFile), []byte("dummy invalid pickle"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := parseTorch(os.DirFS(dir), strings.NewReplacer(), testFile)
	if err == nil {
		t.Fatal("expected error parsing dummy pickle content, got nil")
	}
	// The file should be found on disk and opened, so it must not fail with "no such file or directory"
	if strings.Contains(err.Error(), "no such file") || strings.Contains(err.Error(), "The system cannot find the file specified") {
		t.Fatalf("expected file to be found in fs.FS, but got: %v", err)
	}
}

func TestParseTorchFallbackWithMapFS(t *testing.T) {
	// Test with in-memory fs.FS (MapFS) to verify temporary file copy fallback
	mapFS := fstest.MapFS{
		"pytorch_model.bin": &fstest.MapFile{
			Data: []byte("dummy invalid pickle"),
			Mode: 0o644,
		},
	}

	_, err := parseTorch(mapFS, strings.NewReplacer(), "pytorch_model.bin")
	if err == nil {
		t.Fatal("expected error parsing dummy pickle content, got nil")
	}
	if strings.Contains(err.Error(), "no such file") || strings.Contains(err.Error(), "The system cannot find the file specified") {
		t.Fatalf("expected file to be found in MapFS, but got: %v", err)
	}
}
