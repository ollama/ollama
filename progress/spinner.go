package progress

import (
	"fmt"
	"strings"
	"sync/atomic"
	"time"
)

type Spinner struct {
	message      atomic.Value
	messageWidth int

	parts []string

	value int

	ticker  *time.Ticker
	started time.Time
	stopped time.Time
}

func NewSpinner(message string) *Spinner {
	s := &Spinner{
		parts: []string{
			"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
		},
		started: time.Now(),
	}
	s.SetMessage(message)
	go s.start()
	return s
}

func (s *Spinner) SetMessage(message string) {
	s.message.Store(message)
}

func (s *Spinner) String() string {
	var sb strings.Builder

	if message, ok := s.message.Load().(string); ok && len(message) > 0 {
		message := strings.TrimSpace(message)
		if s.messageWidth > 0 && len(message) > s.messageWidth {
			message = message[:s.messageWidth]
		}

		fmt.Fprintf(&sb, "%s", message)
		if padding := s.messageWidth - sb.Len(); padding > 0 {
			sb.WriteString(strings.Repeat(" ", padding))
		}

		sb.WriteString(" ")
	}

	if s.stopped.IsZero() {
		spinner := s.parts[s.value]
		sb.WriteString(spinner)
		sb.WriteString(" ")
	}

	return sb.String()
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
