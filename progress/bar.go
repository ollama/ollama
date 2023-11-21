package progress

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/jmorganca/ollama/format"
	"golang.org/x/term"
)

type Stats struct {
	rate      int64
	value     int64
	remaining time.Duration
}

type Bar struct {
	message      string
	messageWidth int

	maxValue     int64
	initialValue int64
	currentValue int64

	started time.Time

	stats   Stats
	statted time.Time
}

func NewBar(message string, maxValue, initialValue int64) *Bar {
	return &Bar{
		message:      message,
		messageWidth: -1,
		maxValue:     maxValue,
		initialValue: initialValue,
		currentValue: initialValue,
		started:      time.Now(),
	}
}

// formatDuration limits the rendering of a time.Duration to 2 units
func formatDuration(d time.Duration) string {
	if d >= 100*time.Hour {
		return "99h+"
	}

	if d >= time.Hour {
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	}

	return d.Round(time.Second).String()
}

func (b *Bar) String() string {
	termWidth, _, err := term.GetSize(int(os.Stderr.Fd()))
	if err != nil {
		termWidth = 80
	}

	var pre, mid, suf strings.Builder

	if b.message != "" {
		message := strings.TrimSpace(b.message)
		if b.messageWidth > 0 && len(message) > b.messageWidth {
			message = message[:b.messageWidth]
		}

		fmt.Fprintf(&pre, "%s", message)
		if b.messageWidth-pre.Len() >= 0 {
			pre.WriteString(strings.Repeat(" ", b.messageWidth-pre.Len()))
		}

		pre.WriteString(" ")
	}

	fmt.Fprintf(&pre, "%3.0f%% ", math.Floor(b.percent()))

	fmt.Fprintf(&suf, "(%s/%s", format.HumanBytes(b.currentValue), format.HumanBytes(b.maxValue))

	stats := b.Stats()
	rate := stats.rate
	if stats.value > b.initialValue && stats.value < b.maxValue {
		fmt.Fprintf(&suf, ", %s/s", format.HumanBytes(int64(rate)))
	}

	fmt.Fprintf(&suf, ")")

	elapsed := time.Since(b.started)
	var timing string
	if stats.value > b.initialValue && stats.value < b.maxValue {
		timing = fmt.Sprintf("[%s:%s]", formatDuration(elapsed), formatDuration(stats.remaining))
	}

	// 44 is the maximum width for the stats on the right of the progress bar
	if suf.Len() < 44 {
		suf.WriteString(strings.Repeat(" ", 44-suf.Len()-len(timing)))
	}

	suf.WriteString(timing)

	// add 3 extra spaces: 2 boundary characters and 1 space at the end
	f := termWidth - pre.Len() - suf.Len() - 3
	n := int(float64(f) * b.percent() / 100)

	if f > 0 {
		mid.WriteString("▕")
		mid.WriteString(strings.Repeat("█", n))
		if f-n > 0 {
			mid.WriteString(strings.Repeat(" ", f-n))
		}
		mid.WriteString("▏")
	}

	return pre.String() + mid.String() + suf.String()
}

func (b *Bar) Set(value int64) {
	if value >= b.maxValue {
		value = b.maxValue
	}

	b.currentValue = value
}

func (b *Bar) percent() float64 {
	if b.maxValue > 0 {
		return float64(b.currentValue) / float64(b.maxValue) * 100
	}

	return 0
}

func (b *Bar) Stats() Stats {
	if time.Since(b.statted) < time.Second {
		return b.stats
	}

	switch {
	case b.statted.IsZero():
		b.stats = Stats{
			value:     b.initialValue,
			rate:      0,
			remaining: 0,
		}
	case b.currentValue >= b.maxValue:
		b.stats = Stats{
			value:     b.maxValue,
			rate:      0,
			remaining: 0,
		}
	default:
		rate := b.currentValue - b.stats.value
		var remaining time.Duration
		if rate > 0 {
			remaining = time.Second * time.Duration((float64(b.maxValue-b.currentValue))/(float64(rate)))
		} else {
			remaining = time.Duration(math.MaxInt64)
		}

		b.stats = Stats{
			value:     b.currentValue,
			rate:      rate,
			remaining: remaining,
		}
	}

	b.statted = time.Now()

	return b.stats
}
