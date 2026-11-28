package cmd

import (
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestListenUnixSocket(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix domain socket test")
	}

	socketPath := filepath.Join(t.TempDir(), "ollama.sock")
	listener, err := listen(&url.URL{Scheme: "unix", Path: socketPath})
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	info, err := os.Stat(socketPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Fatalf("socket permissions = %04o, want 0600", got)
	}
}

func TestListenUnixSocketRequiresAbsolutePath(t *testing.T) {
	if _, err := listen(&url.URL{Scheme: "unix", Path: "ollama.sock"}); err == nil {
		t.Fatal("expected relative Unix socket path to fail")
	}
}
