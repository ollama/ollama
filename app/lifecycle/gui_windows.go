package lifecycle

import (
	"fmt"
	"log/slog"
	"os/exec"
	"syscall"
)

func ShowGui() {
	slog.Debug("Launching DemoGUI as a separate process")
	cmd := exec.Command("C:\\ollama-yontracks\\gui\\DemoGUI.exe")

	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	err := cmd.Start()
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to launch DemoGUI: %v", err))
	}
}
