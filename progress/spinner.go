package progress

import (
	"fmt"
	"strings"
	"time"
)

type Spinner struct {
	message      string
	messageWidth int

	value int

	ticker  *time.Ticker
	started time.Time
	stopped time.Time
}

func NewSpinner(message string) *Spinner {
	s := &Spinner{
		message: message,
		started: time.Now(),
		value:   231,
	}
	go s.start()
	return s
}

func (s *Spinner) String() string {
	var sb strings.Builder
	if len(s.message) > 0 {
		message := strings.TrimSpace(s.message)
		if s.messageWidth > 0 && len(message) > s.messageWidth {
			message = message[:s.messageWidth]
		}

		fmt.Fprintf(&sb, "%s", message)
		if s.messageWidth-sb.Len() >= 0 {
			sb.WriteString(strings.Repeat(" ", s.messageWidth-sb.Len()))
		}

		sb.WriteString(" ")
	}

	if s.stopped.IsZero() {
		sb.WriteString(fmt.Sprintf("\033[48;5;%dm ", s.value))
		sb.WriteString("\033[0m")
	}

	return sb.String()
}

func (s *Spinner) start() {
	s.ticker = time.NewTicker(40 * time.Millisecond)
	for range s.ticker.C {
		if s.value < 255 {
			s.value++
		} else {
			s.value = 231
		}
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
