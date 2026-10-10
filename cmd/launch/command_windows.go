package launch

import (
	"context"
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

func backgroundCommandContext(ctx context.Context, name string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windows.CREATE_NO_WINDOW}
	return cmd
}
