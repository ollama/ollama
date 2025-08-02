package cmd

import (
	"fmt"
	"os"
	"time"

	"github.com/jmorganca/ollama/progressbar"
)

type Spinner struct {
	description string
	*progressbar.ProgressBar
}

func NewSpinner(description string) *Spinner {
	return &Spinner{
		description: description,
		ProgressBar: progressbar.NewOptions(-1,
			progressbar.OptionSetWriter(os.Stderr),
			progressbar.OptionThrottle(60*time.Millisecond),
			progressbar.OptionSpinnerType(14),
			progressbar.OptionSetRenderBlankState(true),
			progressbar.OptionSetElapsedTime(false),
			progressbar.OptionClearOnFinish(),
			progressbar.OptionSetDescription(description),
		),
	}
}

func (s *Spinner) Spin(tick time.Duration) {
	for range time.Tick(tick) {
		if s.IsFinished() {
			break
		}

		s.Add(1)
	}
}

func (s *Spinner) Stop() {
	s.Finish()
	fmt.Println(s.description)
}
