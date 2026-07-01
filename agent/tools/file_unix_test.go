//go:build !windows

package tools

import (
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestOpenRegularFileRejectsFIFO(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pipe")
	if err := syscall.Mkfifo(path, 0o600); err != nil {
		t.Skipf("mkfifo unavailable: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		file, _, err := openRegularFile(dir, "pipe")
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
