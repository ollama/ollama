package progress

import (
	"fmt"
	"os"
	"strings"
	"time"

	"golang.org/x/term"

	"github.com/ollama/ollama/format"
)

type Bar struct {
	message      string
	messageWidth int

	maxValue     int64
	initialValue int64
	currentValue int64

	started time.Time
	stopped time.Time

	maxBuckets int
	buckets    []bucket
}

type bucket struct {
	updated time.Time
	value   int64
}

func NewBar(message string, maxValue, initialValue int64) *Bar {
	b := Bar{
		message:      message,
		messageWidth: -1,
		maxValue:     maxValue,
		initialValue: initialValue,
		currentValue: initialValue,
		started:      time.Now(),
		maxBuckets:   10,
	}

	if initialValue >= maxValue {
		b.stopped = time.Now()
	}

	return &b
}

// formatDuration limits the rendering of a time.Duration to 2 units
func formatDuration(d time.Duration) string {
	switch {
	case d >= 100*time.Hour:
		return "99h+"
	case d >= time.Hour:
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	default:
		return d.Round(time.Second).String()
	}
}

func (b *Bar) String() string {
	termWidth, _, err := term.GetSize(int(os.Stderr.Fd()))
	if err != nil {
		termWidth = 80
	}

	var pre strings.Builder
	if len(b.message) > 0 {
		message := strings.TrimSpace(b.message)
		if b.messageWidth > 0 && len(message) > b.messageWidth {
			message = message[:b.messageWidth]
		}

		fmt.Fprintf(&pre, "%s", message)
		if padding := b.messageWidth - pre.Len(); padding > 0 {
			pre.WriteString(repeat(" ", padding))
		}

		pre.WriteString(" ")
	}

	fmt.Fprintf(&pre, "%3.0f%%", b.percent())

	var suf strings.Builder
	// max 13 characters: "999 MB/999 MB"
	if b.stopped.IsZero() {
		curValue := format.HumanBytes(b.currentValue)
		suf.WriteString(repeat(" ", 6-len(curValue)))
		suf.WriteString(curValue)
		suf.WriteString("/")

		maxValue := format.HumanBytes(b.maxValue)
		suf.WriteString(repeat(" ", 6-len(maxValue)))
		suf.WriteString(maxValue)
	} else {
		maxValue := format.HumanBytes(b.maxValue)
		suf.WriteString(repeat(" ", 6-len(maxValue)))
		suf.WriteString(maxValue)
		suf.WriteString(repeat(" ", 7))
	}

	rate := b.rate()
	// max 10 characters: "  999 MB/s"
	if b.stopped.IsZero() && rate > 0 {
		suf.WriteString("  ")
		humanRate := format.HumanBytes(int64(rate))
		suf.WriteString(repeat(" ", 6-len(humanRate)))
		suf.WriteString(humanRate)
		suf.WriteString("/s")
	} else {
		suf.WriteString(repeat(" ", 10))
	}

	// max 8 characters: "  59m59s"
	if b.stopped.IsZero() && rate > 0 {
		suf.WriteString("  ")
		var remaining time.Duration
		if rate > 0 {
			remaining = time.Duration(int64(float64(b.maxValue-b.currentValue)/rate)) * time.Second
		}

		humanRemaining := formatDuration(remaining)
		suf.WriteString(repeat(" ", 6-len(humanRemaining)))
		suf.WriteString(humanRemaining)
	} else {
		suf.WriteString(repeat(" ", 8))
	}

	var mid strings.Builder
	// add 5 extra spaces: 2 boundary characters and 1 space at each end
	f := termWidth - pre.Len() - suf.Len() - 5
	n := int(float64(f) * b.percent() / 100)

	mid.WriteString(" ▕")

	if n > 0 {
		mid.WriteString(repeat("█", n))
	}

	if f-n > 0 {
		mid.WriteString(repeat(" ", f-n))
	}

	mid.WriteString("▏ ")

	return pre.String() + mid.String() + suf.String()
}

func (b *Bar) Set(value int64) {
	if value >= b.maxValue {
		value = b.maxValue
	}

	b.currentValue = value
	if b.currentValue >= b.maxValue {
		b.stopped = time.Now()
	}

	// throttle bucket updates to 1 per second
	if len(b.buckets) == 0 || time.Since(b.buckets[len(b.buckets)-1].updated) > time.Second {
		b.buckets = append(b.buckets, bucket{
			updated: time.Now(),
			value:   value,
		})

		if len(b.buckets) > b.maxBuckets {
			b.buckets = b.buckets[1:]
		}
	}
}

func (b *Bar) percent() float64 {
	if b.maxValue > 0 {
		return float64(b.currentValue) / float64(b.maxValue) * 100
	}

	return 0
}

func (b *Bar) rate() float64 {
	var numerator, denominator float64

	if !b.stopped.IsZero() {
		numerator = float64(b.currentValue - b.initialValue)
		denominator = b.stopped.Sub(b.started).Round(time.Second).Seconds()
	} else {
		switch len(b.buckets) {
		case 0:
			// noop
		case 1:
			numerator = float64(b.buckets[0].value - b.initialValue)
			denominator = b.buckets[0].updated.Sub(b.started).Round(time.Second).Seconds()
		default:
			first, last := b.buckets[0], b.buckets[len(b.buckets)-1]
			numerator = float64(last.value - first.value)
			denominator = last.updated.Sub(first.updated).Round(time.Second).Seconds()
		}
	}

	if denominator != 0 {
		return numerator / denominator
	}

	return 0
}

func repeat(s string, n int) string {
	if n > 0 {
		return strings.Repeat(s, n)
	}

	return ""
}
