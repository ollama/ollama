package convert

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestParseTorchUsesSuppliedFS ensures parseTorch resolves the given path
// against fsys, not against the process's working directory. It chdirs
// somewhere the file doesn't exist, so if parseTorch ever regresses to
// passing the fs-relative path straight to gopickle, this fails with a
// "file does not exist" error instead of a pickle-decoding error.
func TestParseTorchUsesSuppliedFS(t *testing.T) {
	dir := t.TempDir()
	name := "pytorch_model.bin"
	if err := os.WriteFile(filepath.Join(dir, name), []byte("not a real pickle file"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Chdir(t.TempDir())

	_, err := parseTorch(os.DirFS(dir), strings.NewReplacer(), name)
	if err == nil {
		t.Fatal("expected an error decoding a bogus pickle file, got nil")
	}
	if strings.Contains(err.Error(), "no such file") {
		t.Fatalf("parseTorch resolved %q against the working directory instead of the supplied fs.FS: %v", name, err)
	}
}

func TestParseTorchMissingFile(t *testing.T) {
	dir := t.TempDir()

	_, err := parseTorch(os.DirFS(dir), strings.NewReplacer(), "missing.bin")
	if err == nil {
		t.Fatal("expected an error for a missing file, got nil")
	}
}
