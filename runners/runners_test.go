package runners

import (
	"log/slog"
	"os"
	"path"
	"runtime"
	"strings"
	"testing"
	"testing/fstest"
)

func TestRefreshRunners(t *testing.T) {
	slog.SetLogLoggerLevel(slog.LevelDebug)

	payloadFS := fstest.MapFS{
		path.Join(runtime.GOOS, runtime.GOARCH, "foo", "ollama_llama_server"): {Data: []byte("hello, world\n")},
	}
	tmpDir, err := os.MkdirTemp("", "testing")
	if err != nil {
		t.Fatalf("failed to make tmp dir %s", err)
	}
	t.Setenv("OLLAMA_TMPDIR", tmpDir)
	rDir, err := Refresh(payloadFS)
	if err != nil {
		t.Fatalf("failed to extract to %s %s", tmpDir, err)
	}
	if !strings.Contains(rDir, tmpDir) {
		t.Fatalf("runner dir %s was not in tmp dir %s", rDir, tmpDir)
	}

	// spot check results
	servers := GetAvailableServers(rDir)
	if len(servers) < 1 {
		t.Fatalf("expected at least 1 server")
	}

	// Refresh contents
	rDir, err = extractRunners(payloadFS)
	if err != nil {
		t.Fatalf("failed to extract to %s %s", tmpDir, err)
	}
	if !strings.Contains(rDir, tmpDir) {
		t.Fatalf("runner dir %s was not in tmp dir %s", rDir, tmpDir)
	}

	cleanupTmpDirs()

	Cleanup(payloadFS)
}
