package config

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/term"
)

// ANSI escape sequences for terminal formatting.
const (
	ansiBold  = "\033[1m"
	ansiReset = "\033[0m"
	ansiGray  = "\033[37m"
	ansiGreen = "\033[32m"
)

// ErrCancelled is returned when the user cancels a selection.
var ErrCancelled = errors.New("cancelled")

// errCancelled is kept as an alias for backward compatibility within the package.
var errCancelled = ErrCancelled

// DefaultConfirmPrompt provides a TUI-based confirmation prompt.
// When set, confirmPrompt delegates to it instead of using raw terminal I/O.
var DefaultConfirmPrompt func(prompt string) (bool, error)

func confirmPrompt(prompt string) (bool, error) {
	if DefaultConfirmPrompt != nil {
		return DefaultConfirmPrompt(prompt)
	}

	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return false, err
	}
	defer term.Restore(fd, oldState)

	fmt.Fprintf(os.Stderr, "%s (\033[1my\033[0m/n) ", prompt)

	buf := make([]byte, 1)
	for {
		if _, err := os.Stdin.Read(buf); err != nil {
			return false, err
		}

		switch buf[0] {
		case 'Y', 'y', 13:
			fmt.Fprintf(os.Stderr, "yes\r\n")
			return true, nil
		case 'N', 'n', 27, 3:
			fmt.Fprintf(os.Stderr, "no\r\n")
			return false, nil
		}
	}
}
