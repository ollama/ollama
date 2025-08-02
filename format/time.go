package format

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// HumanDuration returns a human-readable approximation of a duration
// (eg. "About a minute", "4 hours ago", etc.).
// Modified version of github.com/docker/go-units.HumanDuration
func HumanDuration(d time.Duration) string {
	return HumanDurationWithCase(d, true)
}

// HumanDurationWithCase returns a human-readable approximation of a
// duration (eg. "About a minute", "4 hours ago", etc.). but allows
// you to specify whether the first word should be capitalized
// (eg. "About" vs. "about")
func HumanDurationWithCase(d time.Duration, useCaps bool) string {
	seconds := int(d.Seconds())

	switch {
	case seconds < 1:
		if useCaps {
			return "Less than a second"
		}
		return "less than a second"
	case seconds == 1:
		return "1 second"
	case seconds < 60:
		return fmt.Sprintf("%d seconds", seconds)
	}

	minutes := int(d.Minutes())
	switch {
	case minutes == 1:
		if useCaps {
			return "About a minute"
		}
		return "about a minute"
	case minutes < 60:
		return fmt.Sprintf("%d minutes", minutes)
	}

	hours := int(math.Round(d.Hours()))
	switch {
	case hours == 1:
		if useCaps {
			return "About an hour"
		}
		return "about an hour"
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
	return humanTimeWithCase(t, zeroValue, true)
}

func HumanTimeLower(t time.Time, zeroValue string) string {
	return humanTimeWithCase(t, zeroValue, false)
}

func humanTimeWithCase(t time.Time, zeroValue string, useCaps bool) string {
	if t.IsZero() {
		return zeroValue
	}

	delta := time.Since(t)
	if delta < 0 {
		return HumanDurationWithCase(-delta, useCaps) + " from now"
	}
	return HumanDurationWithCase(delta, useCaps) + " ago"
}

// ExcatDuration returns a human readable hours/minutes/seconds or milliseconds format of a duration
// the most precise level of duration is milliseconds
func ExactDuration(d time.Duration) string {
	if d.Seconds() < 1 {
		if d.Milliseconds() == 1 {
			return fmt.Sprintf("%d millisecond", d.Milliseconds())
		}
		return fmt.Sprintf("%d milliseconds", d.Milliseconds())
	}

	var readableDur strings.Builder

	dur := d.String()

	// split the default duration string format of 0h0m0s into something nicer to read
	h := strings.Split(dur, "h")
	if len(h) > 1 {
		hours := h[0]
		if hours == "1" {
			readableDur.WriteString(fmt.Sprintf("%s hour ", hours))
		} else {
			readableDur.WriteString(fmt.Sprintf("%s hours ", hours))
		}
		dur = h[1]
	}

	m := strings.Split(dur, "m")
	if len(m) > 1 {
		mins := m[0]
		switch mins {
		case "0":
			// skip
		case "1":
			readableDur.WriteString(fmt.Sprintf("%s minute ", mins))
		default:
			readableDur.WriteString(fmt.Sprintf("%s minutes ", mins))
		}
		dur = m[1]
	}

	s := strings.Split(dur, "s")
	if len(s) > 0 {
		sec := s[0]
		switch sec {
		case "0":
			// skip
		case "1":
			readableDur.WriteString(fmt.Sprintf("%s second ", sec))
		default:
			readableDur.WriteString(fmt.Sprintf("%s seconds ", sec))
		}
	}

	return strings.TrimSpace(readableDur.String())
}
