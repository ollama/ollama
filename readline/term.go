//go:build aix || darwin || dragonfly || freebsd || (linux && !appengine) || netbsd || openbsd || os400 || solaris

package readline

import (
	"syscall"
)

type Termios syscall.Termios

func SetRawMode(fd uintptr) (*Termios, error) {
	termios, err := getTermios(fd)
	if err != nil {
		return nil, err
	}

	newTermios := *termios
	newTermios.Iflag &^= syscall.IGNBRK | syscall.BRKINT | syscall.PARMRK | syscall.ISTRIP | syscall.INLCR | syscall.IGNCR | syscall.ICRNL | syscall.IXON
	newTermios.Lflag &^= syscall.ECHO | syscall.ECHONL | syscall.ICANON | syscall.ISIG | syscall.IEXTEN
	newTermios.Cflag &^= syscall.CSIZE | syscall.PARENB
	newTermios.Cflag |= syscall.CS8
	newTermios.Cc[syscall.VMIN] = 1
	newTermios.Cc[syscall.VTIME] = 0

	return termios, setTermios(fd, &newTermios)
}

func UnsetRawMode(fd uintptr, termios any) error {
	t := termios.(*Termios)
	return setTermios(fd, t)
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd uintptr) bool {
	_, err := getTermios(fd)
	return err == nil
}
