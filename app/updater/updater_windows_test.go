//go:build windows

package updater

import (
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVerifyDownloadRejectsUnsignedWindowsInstaller(t *testing.T) {
	oldUpdateStageDir := UpdateStageDir
	defer func() {
		UpdateStageDir = oldUpdateStageDir
	}()

	t.Setenv("LOCALAPPDATA", t.TempDir())
	UpdateStageDir = t.TempDir()
	bundle := filepath.Join(UpdateStageDir, "etag", "OllamaSetup.exe")
	if err := os.MkdirAll(filepath.Dir(bundle), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(bundle, []byte("not a signed installer"), 0o755); err != nil {
		t.Fatal(err)
	}

	err := verifyDownload()
	if err == nil || !strings.Contains(err.Error(), "signature verification failed") {
		t.Fatalf("expected signature verification failure, got %v", err)
	}
}

func TestDoUpgradeAtStartupRejectsUnsignedWindowsInstaller(t *testing.T) {
	oldUpdateStageDir := UpdateStageDir
	oldRunningInstaller := runningInstaller
	oldUpgradeLogFile := UpgradeLogFile
	oldUpgradeMarkerFile := UpgradeMarkerFile
	oldVerifyDownload := VerifyDownload
	defer func() {
		UpdateStageDir = oldUpdateStageDir
		runningInstaller = oldRunningInstaller
		UpgradeLogFile = oldUpgradeLogFile
		UpgradeMarkerFile = oldUpgradeMarkerFile
		VerifyDownload = oldVerifyDownload
	}()

	t.Setenv("LOCALAPPDATA", t.TempDir())
	UpdateStageDir = t.TempDir()
	runDir := t.TempDir()
	runningInstaller = filepath.Join(runDir, "OllamaSetup.exe")
	UpgradeLogFile = filepath.Join(runDir, "upgrade.log")
	UpgradeMarkerFile = filepath.Join(runDir, "upgraded")
	VerifyDownload = verifyDownload

	bundle := filepath.Join(UpdateStageDir, "etag", "OllamaSetup.exe")
	if err := os.MkdirAll(filepath.Dir(bundle), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(bundle, []byte("not a signed installer"), 0o755); err != nil {
		t.Fatal(err)
	}

	err := DoUpgradeAtStartup()
	if err == nil || !strings.Contains(err.Error(), "signature verification failed") {
		t.Fatalf("expected signature verification failure, got %v", err)
	}
	if _, err := os.Stat(runningInstaller); !os.IsNotExist(err) {
		t.Fatalf("unsigned installer was moved before verification failed: %v", err)
	}
	if _, err := os.Stat(bundle); !os.IsNotExist(err) {
		t.Fatalf("unsigned staged installer was not removed after verification failure: %v", err)
	}
}

func TestIsInstallerRunning(t *testing.T) {
	oldInstaller := Installer
	defer func() {
		Installer = oldInstaller
	}()

	slog.SetLogLoggerLevel(slog.LevelDebug)
	Installer = "go.exe"
	if !isInstallerRunning() {
		t.Fatal("not running")
	}
}
