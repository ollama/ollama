//go:build darwin

package wakelock

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
)

func init() {
	newAssertion = newCaffeinateAssertion
}

// caffeinateAssertion holds a running `caffeinate` subprocess. While the
// process is alive macOS treats the system as having a
// PreventUserIdleSystemSleep assertion (verifiable with
// `pmset -g assertions`). Killing the subprocess drops the assertion.
//
// We invoke caffeinate with `-w <our pid>` so the child exits automatically
// if the ollama process is killed without running release(); this avoids
// orphaning a caffeinate process that would otherwise keep the machine awake
// indefinitely.
//
// Using a subprocess (vs IOPMAssertionCreateWithName via cgo) keeps the
// server package free of cgo, matching the pattern already used elsewhere in
// server/ for platform integration (e.g. app/server/server_unix.go).
type caffeinateAssertion struct {
	cmd *exec.Cmd
}

func newCaffeinateAssertion(reason string) (assertion, error) {
	// -i prevents idle sleep (display and disk sleep remain allowed).
	// -w ties the assertion lifetime to the ollama process so a crashed
	//    server cannot leave the machine permanently awake.
	cmd := exec.Command("caffeinate", "-i", "-w", strconv.Itoa(os.Getpid()))
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start caffeinate: %w", err)
	}
	slog.Debug("acquired wake lock via caffeinate", "pid", cmd.Process.Pid, "reason", reason)
	// Reap the process so it does not become a zombie after release() kills
	// it (or after our PID exits and caffeinate self-terminates).
	go func() {
		_ = cmd.Wait()
	}()
	return &caffeinateAssertion{cmd: cmd}, nil
}

func (a *caffeinateAssertion) release() {
	if a == nil || a.cmd == nil || a.cmd.Process == nil {
		return
	}
	if err := a.cmd.Process.Kill(); err != nil {
		// Already exited is fine.
		slog.Debug("failed to release wake lock", "error", err)
	}
	a.cmd = nil
}
