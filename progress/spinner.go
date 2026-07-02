package progress

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Spinner struct {
	mu           sync.Mutex
	message      atomic.Value
	messageWidth int

	parts []string

	value int

	done     chan struct{}
	stopOnce sync.Once
	started  time.Time
	stopped  time.Time
}

func NewSpinner(message string) *Spinner {
	s := &Spinner{
		parts: []string{
			"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
		},
		started: time.Now(),
		done:    make(chan struct{}),
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
	value := s.value
	stopped := !s.stopped.IsZero()
	s.mu.Unlock()
	if !stopped {
		spinner := s.parts[value]
		sb.WriteString(spinner)
		sb.WriteString(" ")
	}

	return sb.String()
}

func (s *Spinner) start() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	s.mu.Lock()
	if !s.stopped.IsZero() {
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()
	for {
		select {
		case <-s.done:
			return
		case <-ticker.C:
			s.mu.Lock()
			if !s.stopped.IsZero() {
				s.mu.Unlock()
				return
			}
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
	}
	s.stopOnce.Do(func() {
		close(s.done)
	})
}
