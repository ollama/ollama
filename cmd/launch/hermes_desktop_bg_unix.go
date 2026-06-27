//go:build !windows

package launch

import "syscall"

// hermesDesktopBackgroundSysProcAttr returns SysProcAttr for running Hermes
// Desktop in the background on Unix. Setpgid detaches the desktop process from
// the launcher's process group so it survives the launcher exiting and does not
// receive signals sent to the launcher terminal.
func hermesDesktopBackgroundSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{
		Setpgid: true,
	}
}
