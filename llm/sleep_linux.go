//go:build linux

package llm

import (
	"os/exec"
	"sync"
)

var (
	linuxInhibit struct {
		mu  sync.Mutex
		cmd *exec.Cmd
	}
)

func init() {
	inhibitSleep = func() {
		linuxInhibit.mu.Lock()
		defer linuxInhibit.mu.Unlock()

		if linuxInhibit.cmd != nil {
			return
		}

		path, err := exec.LookPath("systemd-inhibit")
		if err != nil {
			return
		}

		cmd := exec.Command(path,
			"--what=sleep:idle",
			"--who=ollama",
			"--why=Model inference in progress",
			"sleep", "infinity",
		)

		if err := cmd.Start(); err == nil {
			linuxInhibit.cmd = cmd
		}
	}

	uninhibitSleep = func() {
		linuxInhibit.mu.Lock()
		defer linuxInhibit.mu.Unlock()

		if linuxInhibit.cmd != nil {
			linuxInhibit.cmd.Process.Kill()
			linuxInhibit.cmd.Wait()
			linuxInhibit.cmd = nil
		}
	}
}
