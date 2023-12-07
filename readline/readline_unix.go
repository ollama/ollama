//go:build !windows

package readline

import (
	"syscall"
)

func (i *Instance) handleCharCtrlZ(fd int, termios *Termios) (string, error) {
	if err := UnsetRawMode(fd, termios); err != nil {
		return "", err
	}

	syscall.Kill(0, syscall.SIGSTOP)

	// on resume...
	return "", nil
}
