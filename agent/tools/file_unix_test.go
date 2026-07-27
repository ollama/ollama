//go:build !windows

package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/ollama/ollama/agent"
)

func TestOpenRegularFileRejectsFIFO(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pipe")
	if err := syscall.Mkfifo(path, 0o600); err != nil {
		t.Skipf("mkfifo unavailable: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		file, _, err := openRegularFile(dir, "pipe", false)
		if file != nil {
			file.Close()
		}
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected FIFO to be rejected")
		}
		if !strings.Contains(err.Error(), "not a regular file") {
			t.Fatalf("err = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("openRegularFile blocked on FIFO")
	}
}

func TestEditPreservesModeDespiteUmask(t *testing.T) {
	oldUmask := syscall.Umask(0o077)
	defer syscall.Umask(oldUmask)

	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	if err := os.WriteFile(path, []byte("hello\n"), 0o666); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0o666); err != nil {
		t.Fatal(err)
	}

	_, err := (&Edit{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":     "note.txt",
		"old_text": "hello",
		"new_text": "hi",
	})
	if err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o666 {
		t.Fatalf("mode = %#o, want 0666", got)
	}
}

func TestReadRejectsSymlinkEscapingWorkingDir(t *testing.T) {
	root := t.TempDir()
	secret := filepath.Join(t.TempDir(), "secret.txt")
	if err := os.WriteFile(secret, []byte("top secret\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "notes")
	if err := os.Symlink(secret, link); err != nil {
		t.Fatal(err)
	}

	_, err := (&Read{}).Execute(context.Background(), agent.ToolContext{WorkingDir: root}, map[string]any{
		"path": "notes",
	})
	if err == nil {
		t.Fatal("expected symlink escaping working dir to be rejected")
	}
	if !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("err = %v, want symlink rejection", err)
	}
}

func TestReadRejectsSymlinkInsideWorkingDirToOutside(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "real.txt")
	if err := os.WriteFile(target, []byte("hello\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A symlink to a sibling file still resolves inside the root; Read must
	// reject it regardless, consistent with Edit's rejectFinalSymlink.
	link := filepath.Join(root, "alias")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}

	_, err := (&Read{}).Execute(context.Background(), agent.ToolContext{WorkingDir: root}, map[string]any{
		"path": "alias",
	})
	if err == nil {
		t.Fatal("expected symlink to be rejected even when target is inside root")
	}
	if !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("err = %v, want symlink rejection", err)
	}
}
