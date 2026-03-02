package cmd

import "syscall"

// backgroundServerSysProcAttr returns SysProcAttr for running the server in the background on Windows.
// CREATE_NO_WINDOW (0x08000000) prevents a console window from appearing.
func backgroundServerSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{
		CreationFlags: 0x08000000,
		HideWindow:    true,
	}
}
