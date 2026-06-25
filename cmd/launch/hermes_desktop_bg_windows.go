package launch

import "syscall"

// hermesDesktopBackgroundSysProcAttr returns SysProcAttr for running Hermes
// Desktop in the background on Windows. CREATE_NO_WINDOW (0x08000000) keeps a
// console window from flashing for the launcher-spawned process while the
// desktop app's own window opens normally.
func hermesDesktopBackgroundSysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{
		CreationFlags: 0x08000000,
		HideWindow:    true,
	}
}
