package format

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// humanDuration returns a human-readable approximation of a
// duration (eg. "About a minute", "4 hours ago", etc.).
func humanDuration(d time.Duration) string {
	seconds := int(d.Seconds())

	switch {
	case seconds < 1:
		return "Less than a second"
	case seconds == 1:
		return "1 second"
	case seconds < 60:
		return fmt.Sprintf("%d seconds", seconds)
	}

	minutes := int(d.Minutes())
	switch {
	case minutes == 1:
		return "About a minute"
	case minutes < 60:
		return fmt.Sprintf("%d minutes", minutes)
	}

	hours := int(math.Round(d.Hours()))
	switch {
	case hours == 1:
		return "About an hour"
	case hours < 48:
		return fmt.Sprintf("%d hours", hours)
	case hours < 24*7*2:
		return fmt.Sprintf("%d days", hours/24)
	case hours < 24*30*2:
		return fmt.Sprintf("%d weeks", hours/24/7)
	case hours < 24*365*2:
		return fmt.Sprintf("%d months", hours/24/30)
	}

	return fmt.Sprintf("%d years", int(d.Hours())/24/365)
}

func HumanTime(t time.Time, zeroValue string) string {
	return humanTime(t, zeroValue)
}

func HumanTimeLower(t time.Time, zeroValue string) string {
	return strings.ToLower(humanTime(t, zeroValue))
}

func humanTime(t time.Time, zeroValue string) string {
	if t.IsZero() {
		return zeroValue
	}

	delta := time.Since(t)
	if int(delta.Hours())/24/365 < -20 {
		return "Forever"
	} else if delta < 0 {
		return humanDuration(-delta) + " from now"
	}

	return humanDuration(delta) + " ago"
}
