package launch

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// SpinnerFrames are the braille spinner frames used by the bubbletea TUIs in
// this codebase (sign-in, upgrade). StartSpinner uses the same frames for its
// fallback so the restart spinner matches the look of those flows.
var SpinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// DefaultSpinner, when set, starts an animated spinner displaying message and
// returns a *Spinner. cmd/cmd.go registers a bubbletea implementation from
// cmd/tui; when unset (or when it returns nil, e.g. no TTY) StartSpinner falls
// back to a simple ANSI spinner using SpinnerFrames.
var DefaultSpinner func(message string) *Spinner

// Spinner is a handle on a running animated spinner. Stop halts the spinner
// and clears its line (it blocks until the spinner has fully stopped and is
// safe to call multiple times). Cancelled returns a channel that is closed if
// the user interrupts the spinner (e.g. with Ctrl+C); wait loops can select on
// it to abort early. For the non-interactive ANSI fallback the channel is never
// closed because Ctrl+C raises SIGINT and terminates the process directly.
type Spinner struct {
	stop      func()
	cancelled chan struct{}
}

// NewSpinner builds a Spinner from a stop function and a cancellation channel.
// It is intended for implementations of DefaultSpinner (e.g. the bubbletea
// spinner in cmd/tui). stop must be safe to call multiple times; cancelled is
// closed by the implementation when the user interrupts the spinner, or left
// open when interruption is handled another way (e.g. SIGINT).
func NewSpinner(stop func(), cancelled chan struct{}) *Spinner {
	return &Spinner{stop: stop, cancelled: cancelled}
}

// Stop halts the spinner and clears its line. It is a no-op when the spinner
// already stopped (for example after the user cancelled it).
func (s *Spinner) Stop() {
	if s != nil && s.stop != nil {
		s.stop()
	}
}

// Cancelled returns a channel that is closed when the user interrupts the
// spinner. Callers may select on it to abort a blocking wait.
func (s *Spinner) Cancelled() <-chan struct{} {
	if s == nil {
		return nil
	}
	return s.cancelled
}

// StartSpinner begins an animated spinner displaying message and returns a
// *Spinner handle. It uses DefaultSpinner when available, otherwise a simple
// ANSI fallback that renders SpinnerFrames to stderr without requiring a TTY.
func StartSpinner(message string) *Spinner {
	if DefaultSpinner != nil {
		if s := DefaultSpinner(message); s != nil {
			return s
		}
	}
	return defaultSpinner(message)
}

// defaultSpinner renders SpinnerFrames to stderr without requiring a TTY. It
// runs in its own goroutine so it can animate while a caller polls; Stop
// signals the goroutine to exit, waits for it, and clears the spinner line.
func defaultSpinner(message string) *Spinner {
	frames := SpinnerFrames
	frame := 0
	fmt.Fprintf(os.Stderr, "\r\033[90m%s %s\033[0m", message, frames[0])

	done := make(chan struct{})
	exited := make(chan struct{})
	var once sync.Once

	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				close(exited)
				return
			case <-ticker.C:
				frame++
				fmt.Fprintf(os.Stderr, "\r\033[90m%s %s\033[0m", message, frames[frame%len(frames)])
			}
		}
	}()

	stop := func() {
		once.Do(func() {
			close(done)
			<-exited
			fmt.Fprintf(os.Stderr, "\r\033[K")
		})
	}

	// Ctrl+C in non-raw mode raises SIGINT and terminates the process by
	// default (the launch flow installs no SIGINT handler), so this cancelled
	// channel is intentionally never closed.
	return &Spinner{stop: stop, cancelled: make(chan struct{})}
}
