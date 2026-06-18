//go:build windows

package tools

import "os/exec"

func configureBashCommand(*exec.Cmd) {}

func killBashCommand(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	_ = cmd.Process.Kill()
	return nil
}
