package progress

import (
	"fmt"
	"os"
	"strings"
	"time"

	"golang.org/x/term"
)

type Spinner struct {
	message      string
	messageWidth int

	parts []string

	value int

	ticker  *time.Ticker
	started time.Time
	stopped time.Time
}

func NewSpinner(message string) *Spinner {
	s := &Spinner{
		message: message,
		parts: []string{
			"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
		},
		started: time.Now(),
	}
	go s.start()
	return s
}

func (s *Spinner) String() string {
	termWidth, _, err := term.GetSize(int(os.Stderr.Fd()))
	if err != nil {
		panic(err)
	}

	var pre strings.Builder
	if len(s.message) > 0 {
		message := strings.TrimSpace(s.message)
		if s.messageWidth > 0 && len(message) > s.messageWidth {
			message = message[:s.messageWidth]
		}

		fmt.Fprintf(&pre, "%s", message)
		if s.messageWidth-pre.Len() >= 0 {
			pre.WriteString(strings.Repeat(" ", s.messageWidth-pre.Len()))
		}

		pre.WriteString(" ")
	}

	var pad int
	if s.stopped.IsZero() {
		// spinner has a string length of 3 but a rune length of 1
		// in order to align correctly, we need to pad with (3 - 1) = 2 spaces
		spinner := s.parts[s.value]
		pre.WriteString(spinner)
		pad = len(spinner) - len([]rune(spinner))
	}

	var suf strings.Builder
	fmt.Fprintf(&suf, "(%s)", s.elapsed())

	var mid strings.Builder
	f := termWidth - pre.Len() - mid.Len() - suf.Len() + pad
	if f > 0 {
		mid.WriteString(strings.Repeat(" ", f))
	}

	return pre.String() + mid.String() + suf.String()
}

func (s *Spinner) start() {
	s.ticker = time.NewTicker(100 * time.Millisecond)
	for range s.ticker.C {
		s.value = (s.value + 1) % len(s.parts)
		if !s.stopped.IsZero() {
			return
		}
	}
}

func (s *Spinner) Stop() {
	if s.stopped.IsZero() {
		s.stopped = time.Now()
	}
}

func (s *Spinner) elapsed() time.Duration {
	stopped := s.stopped
	if stopped.IsZero() {
		stopped = time.Now()
	}

	return stopped.Sub(s.started).Round(time.Second)
}
