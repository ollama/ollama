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
var DefaultConfirmPrompt func(prompt string, options ConfirmOptions) (bool, error)

// ConfirmOptions customizes labels for confirmation prompts.
type ConfirmOptions struct {
	YesLabel string
	NoLabel  string
}

// SingleSelector is a function type for single item selection.
// current is the name of the previously selected item to highlight; empty means no pre-selection.
type SingleSelector func(title string, items []SelectionItem, current string) (string, error)

// SingleSelectorWithUpdates is a single item selector that can receive refreshed item state while open.
type SingleSelectorWithUpdates func(title string, items []SelectionItem, current string, updates <-chan []SelectionItem) (string, error)

// MultiSelector is a function type for multi item selection.
type MultiSelector func(title string, items []SelectionItem, preChecked []string) ([]string, error)

// MultiSelectorWithUpdates is a multi item selector that can receive refreshed item state while open.
type MultiSelectorWithUpdates func(title string, items []SelectionItem, preChecked []string, updates <-chan []SelectionItem) ([]string, error)

// DefaultSingleSelector is the default single-select implementation.
var DefaultSingleSelector SingleSelector

// DefaultSingleSelectorWithUpdates is the default single-select implementation with live updates.
var DefaultSingleSelectorWithUpdates SingleSelectorWithUpdates

// DefaultMultiSelector is the default multi-select implementation.
var DefaultMultiSelector MultiSelector

// DefaultMultiSelectorWithUpdates is the default multi-select implementation with live updates.
var DefaultMultiSelectorWithUpdates MultiSelectorWithUpdates

// DefaultSignIn provides a TUI-based sign-in flow.
// When set, ensureAuth uses it instead of plain text prompts.
// Returns the signed-in username or an error.
var DefaultSignIn func(modelName, signInURL string) (string, error)

// DefaultUpgrade provides a TUI-based upgrade flow.
// Returns the updated plan or an error.
var DefaultUpgrade func(modelName, requiredPlan string) (string, error)

type launchConfirmPolicy struct {
	yes               bool
	requireYesMessage bool
}

var currentLaunchConfirmPolicy launchConfirmPolicy

func withLaunchConfirmPolicy(policy launchConfirmPolicy) func() {
	old := currentLaunchConfirmPolicy
	currentLaunchConfirmPolicy = policy
	return func() {
		currentLaunchConfirmPolicy = old
	}
}

// ConfirmPrompt is the shared confirmation gate for launch flows (integration
// edits, missing-model pulls, sign-in prompts, OpenClaw install/security, etc).
// Behavior is controlled by currentLaunchConfirmPolicy, typically scoped by
// withLaunchConfirmPolicy in LaunchCmd (e.g. auto-approve with --yes).
func ConfirmPrompt(prompt string) (bool, error) {
	return ConfirmPromptWithOptions(prompt, ConfirmOptions{})
}

// ConfirmPromptWithOptions is the shared confirmation gate for launch flows
// that need custom yes/no labels in interactive UIs.
func ConfirmPromptWithOptions(prompt string, options ConfirmOptions) (bool, error) {
	if currentLaunchConfirmPolicy.yes {
		return true, nil
	}
	if currentLaunchConfirmPolicy.requireYesMessage {
		return false, fmt.Errorf("%s requires confirmation; re-run with --yes to continue", prompt)
	}

	if DefaultConfirmPrompt != nil {
		return DefaultConfirmPrompt(prompt, options)
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
