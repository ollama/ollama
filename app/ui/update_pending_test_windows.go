//go:build windows

package ui

import (
	"testing"

	"github.com/ollama/ollama/app/updater"
)

func isolateNoPendingUpdateState(t *testing.T) {
	t.Helper()

	oldStageDir := updater.UpdateStageDir
	updater.UpdateStageDir = t.TempDir()
	t.Cleanup(func() {
		updater.UpdateStageDir = oldStageDir
	})
}
