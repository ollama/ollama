//go:build !windows

package tools

import (
	"context"
	"os/exec"
	"syscall"
)

func shellToolName() string {
	return "bash"
}

func shellToolDescription() string {
	return "Execute a bash command on the system. Use this to inspect files, run tests, and perform development tasks."
}

func shellCommandDescription() string {
	return "The bash command to execute."
}

func newBashCommand(ctx context.Context, command, cwdPath string) *exec.Cmd {
	script := command + "\n__ollama_status=$?\npwd -P > " + shellQuote(cwdPath) + "\nexit $__ollama_status"
	cmd := exec.CommandContext(ctx, "bash", "-c", script)
	configureBashCommand(cmd)
	return cmd
}

func configureBashCommand(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func runBashCommand(cmd *exec.Cmd) error {
	return cmd.Run()
}

func killBashCommand(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	return nil
}
