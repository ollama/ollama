package lifecycle

import (
	"fmt"
	"log/slog"
	"os/exec"
	"syscall"
)

func ShowLogs() {
	cmd_path := "c:\\Windows\\system32\\cmd.exe"
	slog.Debug(fmt.Sprintf("XXX launching start %s", AppDir))
	cmd := exec.Command(cmd_path, "/c", "start", AppDir)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	err := cmd.Start()
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to open log dir: %s", err))
	}
}
