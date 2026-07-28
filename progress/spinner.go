package progress

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Spinner struct {
	message atomic.Value

	mu           sync.Mutex
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
		ticker:  time.NewTicker(100 * time.Millisecond),
	}
	s.SetMessage(message)
	// Pass the ticker's channel in rather than having start read s.ticker, so
	// the goroutine never races with other accessors of the struct.
	go s.start(s.ticker.C)
	return s
}

func (s *Spinner) SetMessage(message string) {
	s.message.Store(message)
}

func (s *Spinner) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()

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

func (s *Spinner) start(c <-chan time.Time) {
	for range c {
		s.mu.Lock()
		s.value = (s.value + 1) % len(s.parts)
		stopped := !s.stopped.IsZero()
		s.mu.Unlock()
		if stopped {
			return
		}
	}
}

func (s *Spinner) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped.IsZero() {
		s.stopped = time.Now()
	}
}
