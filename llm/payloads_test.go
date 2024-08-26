package llm

import (
	"os"
	"strings"
	"testing"
)

func TestExtractRunners(t *testing.T) {
	if !hasPayloads() {
		t.Skip("no payloads")
	}
	tmpDir, err := os.MkdirTemp("", "testing")
	if err != nil {
		t.Fatalf("failed to make tmp dir %s", err)
	}
	t.Setenv("OLLAMA_TMPDIR", tmpDir)
	rDir, err := RunnersDir()
	if err != nil {
		t.Fatalf("failed to extract to %s %s", tmpDir, err)
	}
	if !strings.Contains(rDir, tmpDir) {
		t.Fatalf("runner dir %s was not in tmp dir %s", rDir, tmpDir)
	}

	// spot check results
	servers := getAvailableServers(rDir)
	if len(servers) < 1 {
		t.Fatalf("expected at least 1 server")
	}

	// Refresh contents
	rDir, err = extractRunners()
	if err != nil {
		t.Fatalf("failed to extract to %s %s", tmpDir, err)
	}
	if !strings.Contains(rDir, tmpDir) {
		t.Fatalf("runner dir %s was not in tmp dir %s", rDir, tmpDir)
	}

	cleanupTmpDirs()

	Cleanup()
}
