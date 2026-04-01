package progress

import (
	"fmt"
	"strings"
)

// StepBar displays step-based progress (e.g., for image generation steps).
type StepBar struct {
	message string
	current int
	total   int
}

func NewStepBar(message string, total int) *StepBar {
	return &StepBar{message: message, total: total}
}

func (s *StepBar) Set(current int) {
	s.current = current
}

func (s *StepBar) String() string {
	percent := float64(s.current) / float64(s.total) * 100
	barWidth := s.total
	empty := barWidth - s.current

	// "Generating   0% ▕         ▏ 0/9"
	return fmt.Sprintf("%s %3.0f%% ▕%s%s▏ %d/%d",
		s.message, percent,
		strings.Repeat("█", s.current), strings.Repeat(" ", empty),
		s.current, s.total)
}
