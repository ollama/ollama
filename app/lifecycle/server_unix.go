//go:build !windows

package lifecycle

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"syscall"
)

func getCmd(ctx context.Context, cmd string) *exec.Cmd {
	return exec.CommandContext(ctx, cmd, "serve")
}

func terminate(cmd *exec.Cmd) error {
	return cmd.Process.Signal(os.Interrupt)
}

func isProcessExited(pid int) (bool, error) {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false, fmt.Errorf("failed to find process: %v", err)
	}

	err = proc.Signal(syscall.Signal(0))
	if err != nil {
		if errors.Is(err, os.ErrProcessDone) || errors.Is(err, syscall.ESRCH) {
			return true, nil
		}

		return false, fmt.Errorf("error signaling process: %v", err)
	}

	return false, nil
}
