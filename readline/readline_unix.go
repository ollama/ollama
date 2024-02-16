//go:build !windows

package readline

import (
	"syscall"
)

func handleCharCtrlZ(fd int, termios any) (string, error) {
	t := termios.(*Termios)
	if err := UnsetRawMode(fd, t); err != nil {
		return "", err
	}

	_ = syscall.Kill(0, syscall.SIGSTOP)

	// on resume...
	return "", nil
}
