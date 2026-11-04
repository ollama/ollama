//go:build linux

package server

import "os/exec"

// platformAcquirePowerAssertion implements the linux side of issue #4072 via
// systemd's inhibitor interface. It spawns `systemd-inhibit` blocking
// sleep+idle for the lifetime of the shared assertion (one child process no
// matter how many requests are concurrent). On non-systemd systems the helper
// is absent and this degrades to a no-op so inference is unaffected.
var platformAcquirePowerAssertion = func() (func(), bool) {
	if _, err := exec.LookPath("systemd-inhibit"); err != nil {
		return nil, false
	}
	cmd := exec.Command("systemd-inhibit",
		"--what=sleep:idle",
		"--who=ollama",
		"--why=local inference in progress",
		"--mode=block",
		"sleep", "infinity",
	)
	if err := cmd.Start(); err != nil {
		return nil, false
	}
	return func() {
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		cmd.Wait()
	}, true
}
