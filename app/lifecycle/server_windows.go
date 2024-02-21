package lifecycle

import (
	"context"
	"os/exec"
	"syscall"
)

func getCmd(ctx context.Context, exePath string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, exePath, "serve")
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
	return cmd
}
