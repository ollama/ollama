package main

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func testTrashDirectory(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, ".Trash"), "ollama-startup-test-")
	if err != nil {
		t.Skipf("cannot create a temporary directory in Trash: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

func TestIsInTrash(t *testing.T) {
	trash := testTrashDirectory(t)
	normal := t.TempDir()
	for _, tc := range []struct {
		name string
		path string
		want bool
	}{
		{"trashed bundle", filepath.Join(trash, "Ollama.app"), true},
		{"nested backup", filepath.Join(trash, "uninstalled", "Ollama.app"), true},
		{"ordinary bundle", filepath.Join(normal, "Ollama.app"), false},
		{"unrelated Trash name", filepath.Join(normal, "Trash", "Ollama.app"), false},
		{"unrelated dot Trash name", filepath.Join(normal, ".Trash", "Ollama.app"), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := os.MkdirAll(tc.path, 0o700); err != nil {
				t.Fatal(err)
			}
			if got := isInTrash(tc.path); got != tc.want {
				t.Fatalf("isInTrash(%q) = %v, want %v", tc.path, got, tc.want)
			}
		})
	}
	link := filepath.Join(normal, "linked-app")
	if err := os.Symlink(filepath.Join(trash, "Ollama.app"), link); err != nil {
		t.Fatal(err)
	}
	if !isInTrash(link) {
		t.Fatal("symlink to a trashed bundle was not recognized")
	}
	if isInTrash("") {
		t.Fatal("empty path is in Trash")
	}
}

func TestTrashStartupHelper(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_TRASH_STARTUP") != "1" {
		return
	}
	exitIfRunningFromTrash()
	os.Exit(42) // Startup was allowed to continue.
}

func TestTrashedExecutableDoesNotStart(t *testing.T) {
	trash := testTrashDirectory(t)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{
		"Ollama.app/Contents/MacOS/Ollama",
		"backup/Ollama.app/Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel",
	} {
		t.Run(path, func(t *testing.T) {
			destination := filepath.Join(trash, path)
			if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
				t.Fatal(err)
			}
			src, err := os.Open(executable)
			if err != nil {
				t.Fatal(err)
			}
			defer src.Close()
			dst, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o700)
			if err != nil {
				t.Fatal(err)
			}
			_, copyErr := io.Copy(dst, src)
			closeErr := dst.Close()
			if copyErr != nil || closeErr != nil {
				t.Fatalf("copy test executable: %v, %v", copyErr, closeErr)
			}
			run := func(path string) ([]byte, error) {
				cmd := exec.Command(path, "-test.run=^TestTrashStartupHelper$")
				cmd.Env = append(os.Environ(), "OLLAMA_TEST_TRASH_STARTUP=1")
				return cmd.CombinedOutput()
			}
			output, err := run(destination)
			if err != nil || !strings.Contains(string(output), "not starting Ollama from Trash") {
				t.Fatalf("trashed executable did not exit cleanly: %v\n%s", err, output)
			}
			restored := filepath.Join(t.TempDir(), "Ollama")
			if err := os.Rename(destination, restored); err != nil {
				t.Fatal(err)
			}
			output, err = run(restored)
			if exit, ok := err.(*exec.ExitError); !ok || exit.ExitCode() != 42 {
				t.Fatalf("restored executable was not allowed to continue: %v\n%s", err, output)
			}
		})
	}
}
