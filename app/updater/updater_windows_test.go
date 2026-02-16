//go:build windows || darwin

package updater

import (
	"log/slog"
	"testing"
)

func TestIsInstallerRunning(t *testing.T) {
	slog.SetLogLoggerLevel(slog.LevelDebug)
	Installer = "go.exe"
	if !isInstallerRunning() {
		t.Fatal("not running")
	}
}
