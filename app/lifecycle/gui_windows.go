package lifecycle

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
)

func ShowGui() {
	localAppData := os.Getenv("LOCALAPPDATA")
	AppDataDir = filepath.Join(localAppData, "Ollama")
	AppDir = filepath.Join(localAppData, "Programs", "Ollama")
	cmd_path := filepath.Join(AppDir, "DemoGUI.exe")
	slog.Debug(fmt.Sprintf("Launching DemoGUI as a separate process %s", AppDataDir))
	cmd := exec.Command(cmd_path, "/c", "start", AppDataDir)

	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	err := cmd.Start()
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to launch DemoGUI: %v", err))
	}
}
