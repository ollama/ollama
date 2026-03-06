package launch

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/term"
)

// ANSI escape sequences for terminal formatting.
const (
	ansiBold   = "\033[1m"
	ansiReset  = "\033[0m"
	ansiGray   = "\033[37m"
	ansiGreen  = "\033[32m"
	ansiYellow = "\033[33m"
)

// ErrCancelled is returned when the user cancels a selection.
var ErrCancelled = errors.New("cancelled")

// errCancelled is kept as an internal alias for existing call sites.
var errCancelled = ErrCancelled

// DefaultConfirmPrompt provides a TUI-based confirmation prompt.
// When set, ConfirmPrompt delegates to it instead of using raw terminal I/O.
var DefaultConfirmPrompt func(prompt string) (bool, error)

// SingleSelector is a function type for single item selection.
// current is the name of the previously selected item to highlight; empty means no pre-selection.
type SingleSelector func(title string, items []ModelItem, current string) (string, error)

// MultiSelector is a function type for multi item selection.
type MultiSelector func(title string, items []ModelItem, preChecked []string) ([]string, error)

// DefaultSingleSelector is the default single-select implementation.
var DefaultSingleSelector SingleSelector

// DefaultMultiSelector is the default multi-select implementation.
var DefaultMultiSelector MultiSelector

// DefaultSignIn provides a TUI-based sign-in flow.
// When set, EnsureAuth uses it instead of plain text prompts.
// Returns the signed-in username or an error.
var DefaultSignIn func(modelName, signInURL string) (string, error)

// ConfirmPrompt asks the user to confirm an action using the configured prompt hook.
func ConfirmPrompt(prompt string) (bool, error) {
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
