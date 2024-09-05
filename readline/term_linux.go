//go:build linux || solaris

package readline

import (
	"syscall"
	"unsafe"
)

const (
	tcgets = 0x5401
	tcsets = 0x5402
)

func getTermios(fd uintptr) (*Termios, error) {
	termios := new(Termios)
	_, _, err := syscall.Syscall6(syscall.SYS_IOCTL, fd, tcgets, uintptr(unsafe.Pointer(termios)), 0, 0, 0)
	if err != 0 {
		return nil, err
	}
	return termios, nil
}

func setTermios(fd uintptr, termios *Termios) error {
	_, _, err := syscall.Syscall6(syscall.SYS_IOCTL, fd, tcsets, uintptr(unsafe.Pointer(termios)), 0, 0, 0)
	if err != 0 {
		return err
	}
	return nil
}
