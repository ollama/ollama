//go:build aix

package readline

import (
	"golang.org/x/sys/unix"
)

func getTermios(fd uintptr) (*Termios, error) {
	t, err := unix.IoctlGetTermios(int(fd), unix.TCGETS)
	if err != nil {
		return nil, err
	}
	termios := Termios(*t)
	return &termios, nil
}

func setTermios(fd uintptr, termios *Termios) error {
	t := unix.Termios(*termios)
	return unix.IoctlSetTermios(int(fd), unix.TCSETS, &t)
}
