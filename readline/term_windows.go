package readline

import (
	"golang.org/x/sys/windows"
)

type State struct {
	mode uint32
}

// IsTerminal checks if the given file descriptor is associated with a terminal
func IsTerminal(fd int) bool {
	var st uint32
	err := windows.GetConsoleMode(windows.Handle(fd), &st)
	return err == nil
}

func SetRawMode(fd int) (*State, error) {
	var st uint32
	if err := windows.GetConsoleMode(windows.Handle(fd), &st); err != nil {
		return nil, err
	}
	raw := st &^ (windows.ENABLE_ECHO_INPUT | windows.ENABLE_PROCESSED_INPUT | windows.ENABLE_LINE_INPUT | windows.ENABLE_PROCESSED_OUTPUT)
	raw |= windows.ENABLE_VIRTUAL_TERMINAL_INPUT
	if err := windows.SetConsoleMode(windows.Handle(fd), raw); err != nil {
		return nil, err
	}
	return &State{st}, nil
}

func UnsetRawMode(fd int, state any) error {
	s := state.(*State)
	return windows.SetConsoleMode(windows.Handle(fd), s.mode)
}
