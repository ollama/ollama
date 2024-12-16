package lifecycle

import (
	"fmt"
	"log/slog"
	"os/exec"
	"syscall"
)

func ShowSettings() {
	cmd_path := "c:\\Windows\\system32\\cmd.exe"
	slog.Debug("Opening Windows Environment Variables settings")
	cmd := exec.Command(cmd_path, "/c", "start", "SystemPropertiesAdvanced")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: false, CreationFlags: 0x08000000}
	err := cmd.Start()
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to open Environment Variables settings: %s", err))
	}
}
