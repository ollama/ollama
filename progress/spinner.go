package progress

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Spinner struct {
	message      atomic.Value
	messageWidth int

	parts []string

	mu      sync.Mutex
	value   int
	stopped time.Time

	ticker  *time.Ticker
	started time.Time

	// done is closed by Stop to terminate the animation goroutine
	done chan struct{}
}

func NewSpinner(message string) *Spinner {
	s := &Spinner{
		parts: []string{
			"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
		},
		started: time.Now(),
		// create the ticker before the animation goroutine exists so a Stop
		// that arrives before the first tick still stops it
		ticker: time.NewTicker(100 * time.Millisecond),
		done:   make(chan struct{}),
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

	s.mu.Lock()
	if s.stopped.IsZero() {
		sb.WriteString(s.parts[s.value])
		sb.WriteString(" ")
	}
	s.mu.Unlock()

	return sb.String()
}

func (s *Spinner) start() {
	for {
		select {
		case <-s.done:
			return
		case <-s.ticker.C:
			s.mu.Lock()
			s.value = (s.value + 1) % len(s.parts)
			s.mu.Unlock()
		}
	}
}

func (s *Spinner) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped.IsZero() {
		s.stopped = time.Now()
		s.ticker.Stop()
		close(s.done)
	}
}
