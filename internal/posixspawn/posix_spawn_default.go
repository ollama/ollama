//go:build !darwin

package posixspawn

import "os/exec"

// StartCmd calls cmd.Start() on non-macOS platforms.
func StartCmd(cmd *exec.Cmd) error {
	return cmd.Start()
}
