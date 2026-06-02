package llm

import (
	"log/slog"
	"os"
	"os/exec"
	"sync"
)

var (
	caffeinateCmd *exec.Cmd
	caffeinateMu  sync.Mutex
)

func preventSleep() (func(), error) {
	caffeinateMu.Lock()
	defer caffeinateMu.Unlock()

	if caffeinateCmd != nil {
		return func() {}, nil
	}

	cmd := exec.Command("caffeinate", "-dimsu")
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		slog.Debug("failed to start caffeinate to prevent sleep", "error", err)
		return func() {}, nil
	}
	caffeinateCmd = cmd
	slog.Debug("sleep prevention enabled via caffeinate")

	return func() {
		caffeinateMu.Lock()
		defer caffeinateMu.Unlock()
		if caffeinateCmd != nil {
			if err := caffeinateCmd.Process.Kill(); err != nil {
				slog.Debug("failed to kill caffeinate", "error", err)
			}
			caffeinateCmd = nil
			slog.Debug("sleep prevention disabled")
		}
	}, nil
}
