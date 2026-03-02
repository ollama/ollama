//go:build !windows

package cmd

import "syscall"

// backgroundServerSysProcAttr returns SysProcAttr for running the server in the background on Unix.
// Setpgid prevents the server from being killed when the parent process exits.
func backgroundServerSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{
		Setpgid: true,
	}
}
