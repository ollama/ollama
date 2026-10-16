//go:build windows

package server

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestResolvePathAddsExecutableExtension(t *testing.T) {
	t.Chdir(t.TempDir())

	name := "ollama-resolve-path-test"
	want := filepath.Join("dist", "windows-"+runtime.GOARCH, name+".exe")
	if err := os.MkdirAll(filepath.Dir(want), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(want, nil, 0o755); err != nil {
		t.Fatal(err)
	}

	if got := resolvePath(name); got != want {
		t.Fatalf("resolvePath(%q) = %q, want %q", name, got, want)
	}
}
