//go:build !windows

package llm

import (
	"syscall"
)

// killProcess sends SIGKILL directly to a pid, bypassing exec.Cmd. Only used
// after Process.Kill already failed, and it deliberately doesn't touch the Cmd.
func killProcess(pid int) {
	_ = syscall.Kill(pid, syscall.SIGKILL)
}
