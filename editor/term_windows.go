// +build windows
package editor

import (
	"syscall"
	"unsafe"
)

type State uint32

var kernel32 = syscall.NewLazyDLL("kernel32.dll")

var (
        procGetConsoleMode             = kernel32.NewProc("GetConsoleMode")
        procSetConsoleMode             = kernel32.NewProc("SetConsoleMode")
        procGetConsoleScreenBufferInfo = kernel32.NewProc("GetConsoleScreenBufferInfo")
)

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd int) bool {
        var st uint32
        r, _, e := syscall.Syscall(procGetConsoleMode.Addr(), 2, uintptr(fd), uintptr(unsafe.Pointer(&st)), 0)
        return r != 0 && e == 0
}

func SetRawMode(fd int) (State, err) {
        var state State
        _, _, e := syscall.Syscall(procGetConsoleMode.Addr(), 2, uintptr(fd), uintptr(unsafe.Pointer(&state)), 0)
        if e != 0 {
                return 0, error(e)
        }
        raw := state &^ (enableEchoInput | enableProcessedInput | enableLineInput | enableProcessedOutput)
        _, _, e = syscall.Syscall(procSetConsoleMode.Addr(), 2, uintptr(fd), uintptr(raw), 0)
        if e != 0 {
                return nil, error(e)
        }
        return state, nil
}

func UnsetRawMode(fd int, state State) error {
	_, _, err := syscall.Syscall(procSetConsoleMode.Addr(), 2, uintptr(fd), uintptr(state.mode), 0)
	return err
}
