package progress

import (
	"strings"
	"testing"
	"time"
)

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		name     string
		duration time.Duration
		want     string
	}{
		{"zero", 0, "0s"},
		{"seconds", 30 * time.Second, "30s"},
		{"minutes", 5*time.Minute + 30*time.Second, "5m30s"},
		{"hour", 2*time.Hour + 15*time.Minute, "2h15m"},
		{"over 100 hours", 150 * time.Hour, "99h+"},
		{"exactly 100 hours", 100 * time.Hour, "99h+"},
		{"just under 100 hours", 99*time.Hour + 59*time.Minute + 59*time.Second, "99h59m"},
		{"milliseconds", 500 * time.Millisecond, "1s"},
		{"hour and minute rounding", 2*time.Hour + 45*time.Minute, "2h45m"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatDuration(tt.duration)
			if got != tt.want {
				t.Errorf("formatDuration(%v) = %q, want %q", tt.duration, got, tt.want)
			}
		})
	}
}

func TestNewBar(t *testing.T) {
	t.Run("creates bar with initial value below max", func(t *testing.T) {
		bar := NewBar("downloading", 1000, 0)
		if bar.message != "downloading" {
			t.Errorf("message = %q, want %q", bar.message, "downloading")
		}
		if bar.maxValue != 1000 {
			t.Errorf("maxValue = %d, want 1000", bar.maxValue)
		}
		if bar.currentValue != 0 {
			t.Errorf("currentValue = %d, want 0", bar.currentValue)
		}
		if !bar.stopped.IsZero() {
			t.Error("bar should not be stopped when initial < max")
		}
	})

	t.Run("creates bar completed at max", func(t *testing.T) {
		bar := NewBar("downloading", 1000, 1000)
		if bar.currentValue != 1000 {
			t.Errorf("currentValue = %d, want 1000", bar.currentValue)
		}
		if bar.stopped.IsZero() {
			t.Error("bar should be stopped when initial >= max")
		}
	})

	t.Run("creates bar with initial above max", func(t *testing.T) {
		bar := NewBar("downloading", 1000, 1500)
		if bar.currentValue != 1500 {
			t.Errorf("currentValue = %d, want 1500", bar.currentValue)
		}
		if bar.stopped.IsZero() {
			t.Error("bar should be stopped when initial > max")
		}
	})

	t.Run("creates bar with empty message", func(t *testing.T) {
		bar := NewBar("", 100, 0)
		if bar.message != "" {
			t.Errorf("message = %q, want empty", bar.message)
		}
	})
}

func TestBar_percent(t *testing.T) {
	tests := []struct {
		name       string
		maxValue   int64
		current    int64
		wantPercent float64
	}{
		{"zero progress", 100, 0, 0},
		{"half progress", 100, 50, 50},
		{"full progress", 100, 100, 100},
		{"quarter progress", 100, 25, 25},
		{"zero max", 0, 50, 0},
		{"over max", 100, 150, 150},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bar := NewBar("test", tt.maxValue, 0)
			bar.currentValue = tt.current
			got := bar.percent()
			if got != tt.wantPercent {
				t.Errorf("percent() = %v, want %v", got, tt.wantPercent)
			}
		})
	}
}

func TestBar_Set(t *testing.T) {
	t.Run("updates current value", func(t *testing.T) {
		bar := NewBar("test", 100, 0)
		bar.Set(50)
		if bar.currentValue != 50 {
			t.Errorf("currentValue = %d, want 50", bar.currentValue)
		}
	})

	t.Run("caps at max value", func(t *testing.T) {
		bar := NewBar("test", 100, 0)
		bar.Set(150)
		if bar.currentValue != 100 {
			t.Errorf("currentValue = %d, want 100 (capped)", bar.currentValue)
		}
	})

	t.Run("marks stopped when reaching max", func(t *testing.T) {
		bar := NewBar("test", 100, 0)
		time.Sleep(10 * time.Millisecond)
		bar.Set(100)
		if bar.stopped.IsZero() {
			t.Error("bar should be stopped after reaching max")
		}
	})

	t.Run("does not overwrite stopped time on subsequent sets", func(t *testing.T) {
		bar := NewBar("test", 100, 0)
		bar.Set(100)
		stoppedTime := bar.stopped
		bar.Set(100)
		if bar.stopped != stoppedTime {
			t.Error("stopped time should not change on subsequent sets")
		}
	})
}

func TestBar_rate(t *testing.T) {
	t.Run("returns zero with no buckets", func(t *testing.T) {
		bar := NewBar("test", 100, 0)
		if bar.rate() != 0 {
			t.Errorf("rate with no buckets = %v, want 0", bar.rate())
		}
	})

	t.Run("returns zero when max is reached", func(t *testing.T) {
		bar := NewBar("test", 100, 100)
		if bar.rate() != 0 {
			t.Errorf("rate of completed bar = %v, want 0", bar.rate())
		}
	})

	t.Run("calculates rate from buckets", func(t *testing.T) {
		bar := NewBar("test", 100, 0)
		bar.initialValue = 0
		bar.currentValue = 50
		bar.buckets = []bucket{
			{updated: time.Now().Add(-2 * time.Second), value: 10},
			{updated: time.Now().Add(-time.Second), value: 30},
			{updated: time.Now(), value: 50},
		}
	})
}

func TestBar_String(t *testing.T) {
	t.Run("contains message", func(t *testing.T) {
		bar := NewBar("Downloading", 1000, 0)
		output := bar.String()
		if !strings.Contains(output, "Downloading") {
			t.Errorf("output missing message: %q", output)
		}
	})

	t.Run("contains percentage", func(t *testing.T) {
		bar := NewBar("test", 100, 50)
		output := bar.String()
		if !strings.Contains(output, "50%") {
			t.Errorf("output missing 50%%: %q", output)
		}
	})

	t.Run("contains bar boundaries", func(t *testing.T) {
		bar := NewBar("test", 100, 50)
		output := bar.String()
		if !strings.Contains(output, "▕") || !strings.Contains(output, "▏") {
			t.Errorf("output missing bar boundaries: %q", output)
		}
	})

	t.Run("formats complete bar", func(t *testing.T) {
		bar := NewBar("test", 100, 100)
		output := bar.String()
		if !strings.Contains(output, "100%") {
			t.Errorf("output missing 100%%: %q", output)
		}
	})

	t.Run("handles zero max", func(t *testing.T) {
		bar := NewBar("test", 0, 0)
		output := bar.String()
		if !strings.Contains(output, "0%") {
			t.Errorf("output should contain 0%%: %q", output)
		}
	})
}

func TestRepeat(t *testing.T) {
	tests := []struct {
		s    string
		n    int
		want string
	}{
		{"█", 3, "███"},
		{" ", 5, "     "},
		{"ab", 2, "abab"},
		{"", 5, ""},
		{"x", 0, ""},
		{"x", -1, ""},
	}

	for _, tt := range tests {
		t.Run(tt.s+string(rune(tt.n)), func(t *testing.T) {
			got := repeat(tt.s, tt.n)
			if got != tt.want {
				t.Errorf("repeat(%q, %d) = %q, want %q", tt.s, tt.n, got, tt.want)
			}
		})
	}
}