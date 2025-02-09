package lifecycle

import (
    "os"
    "path/filepath"
    "strconv"
    "testing"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "io"
    "strings"
)

func TestRotateLogs(t *testing.T) {
	logDir := t.TempDir()
	logFile := filepath.Join(logDir, "testlog.log")

	// No log exists
	rotateLogs(logFile)

	require.NoError(t, os.WriteFile(logFile, []byte("1"), 0o644))
	assert.FileExists(t, logFile)
	// First rotation
	rotateLogs(logFile)
	assert.FileExists(t, filepath.Join(logDir, "testlog-1.log"))
	assert.NoFileExists(t, filepath.Join(logDir, "testlog-2.log"))
	assert.NoFileExists(t, logFile)

	// Should be a no-op without a new log
	rotateLogs(logFile)
	assert.FileExists(t, filepath.Join(logDir, "testlog-1.log"))
	assert.NoFileExists(t, filepath.Join(logDir, "testlog-2.log"))
	assert.NoFileExists(t, logFile)

	for i := 2; i <= LogRotationCount+1; i++ {
		require.NoError(t, os.WriteFile(logFile, []byte(strconv.Itoa(i)), 0o644))
		assert.FileExists(t, logFile)
		rotateLogs(logFile)
		assert.NoFileExists(t, logFile)
		for j := 1; j < i; j++ {
			assert.FileExists(t, filepath.Join(logDir, "testlog-"+strconv.Itoa(j)+".log"))
		}
		assert.NoFileExists(t, filepath.Join(logDir, "testlog-"+strconv.Itoa(i+1)+".log"))
	}
}

// Test generated using Keploy
func TestInitLogging_DebugMode(t *testing.T) {
    // Mock envconfigDebug to return true
    originalEnvconfigDebug := envconfigDebug
    envconfigDebug = func() bool { return true }
    defer func() { envconfigDebug = originalEnvconfigDebug }()

    // Redirect os.Stderr to capture logs
    originalStderr := os.Stderr
    r, w, _ := os.Pipe()
    os.Stderr = w

    // Call InitLogging
    InitLogging()

    // Close and restore os.Stderr
    w.Close()
    os.Stderr = originalStderr

    // Read captured logs
    var buf strings.Builder
    _, _ = io.Copy(&buf, r)

    // Assert that the startup message is present
    assert.Contains(t, buf.String(), "ollama app started")
}

