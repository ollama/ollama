package lifecycle

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
