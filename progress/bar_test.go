package progress

import (
	"strings"
	"testing"
	"time"
)

func TestNewBar(t *testing.T) {
	tests := []struct {
		name         string
		message      string
		maxValue     int64
		initialValue int64
		wantStopped  bool
	}{
		{
			name:         "basic bar",
			message:      "downloading",
			maxValue:     100,
			initialValue: 0,
			wantStopped:  false,
		},
		{
			name:         "already complete",
			message:      "done",
			maxValue:     100,
			initialValue: 100,
			wantStopped:  true,
		},
		{
			name:         "over complete",
			message:      "over",
			maxValue:     100,
			initialValue: 150,
			wantStopped:  true,
		},
		{
			name:         "empty message",
			message:      "",
			maxValue:     1000,
			initialValue: 500,
			wantStopped:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bar := NewBar(tt.message, tt.maxValue, tt.initialValue)
			if bar.message != tt.message {
				t.Errorf("message = %q, want %q", bar.message, tt.message)
			}
			if bar.maxValue != tt.maxValue {
				t.Errorf("maxValue = %d, want %d", bar.maxValue, tt.maxValue)
			}
			if bar.initialValue != tt.initialValue {
				t.Errorf("initialValue = %d, want %d", bar.initialValue, tt.initialValue)
			}
			if bar.currentValue != tt.initialValue {
				t.Errorf("currentValue = %d, want %d", bar.currentValue, tt.initialValue)
			}
			if bar.stopped.IsZero() == tt.wantStopped {
				t.Errorf("stopped.IsZero() = %v, want %v", bar.stopped.IsZero(), !tt.wantStopped)
			}
		})
	}
}

func TestBarSet(t *testing.T) {
	bar := NewBar("test", 100, 0)

	// Set to 50%
	bar.Set(50)
	if bar.currentValue != 50 {
		t.Errorf("currentValue = %d, want 50", bar.currentValue)
	}
	if !bar.stopped.IsZero() {
		t.Error("bar should not be stopped at 50%")
	}

	// Set to 100% (complete)
	bar.Set(100)
	if bar.currentValue != 100 {
		t.Errorf("currentValue = %d, want 100", bar.currentValue)
	}
	if bar.stopped.IsZero() {
		t.Error("bar should be stopped at 100%")
	}
}

func TestBarSetOverMax(t *testing.T) {
	bar := NewBar("test", 100, 0)

	// Set beyond max
	bar.Set(150)
	if bar.currentValue != 100 {
		t.Errorf("currentValue = %d, want 100 (clamped to max)", bar.currentValue)
	}
	if bar.stopped.IsZero() {
		t.Error("bar should be stopped when value >= max")
	}
}

func TestBarPercent(t *testing.T) {
	tests := []struct {
		name         string
		maxValue     int64
		currentValue int64
		want         float64
	}{
		{"0%", 100, 0, 0},
		{"50%", 100, 50, 50},
		{"100%", 100, 100, 100},
		{"25%", 1000, 250, 25},
		{"zero max", 0, 50, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bar := NewBar("", tt.maxValue, 0)
			bar.currentValue = tt.currentValue
			got := bar.percent()
			if got != tt.want {
				t.Errorf("percent() = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestBarString(t *testing.T) {
	bar := NewBar("downloading", 1000, 0)
	bar.Set(500)

	str := bar.String()

	// Should contain percentage
	if !strings.Contains(str, "50%") {
		t.Errorf("String() should contain '50%%', got %q", str)
	}

	// Should contain progress bar characters
	if !strings.Contains(str, "▕") || !strings.Contains(str, "▏") {
		t.Error("String() should contain progress bar boundary characters")
	}
}

func TestBarStringComplete(t *testing.T) {
	bar := NewBar("done", 1000, 1000)

	str := bar.String()

	// Should show 100%
	if !strings.Contains(str, "100%") {
		t.Errorf("String() should contain '100%%', got %q", str)
	}
}

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		name     string
		duration time.Duration
		want     string
	}{
		{"zero", 0, "0s"},
		{"seconds", 45 * time.Second, "45s"},
		{"one minute", time.Minute, "1m0s"},
		{"minutes and seconds", 5*time.Minute + 30*time.Second, "5m30s"},
		{"one hour", time.Hour, "1h0m"},
		{"hours and minutes", 2*time.Hour + 15*time.Minute, "2h15m"},
		{"99+ hours", 100 * time.Hour, "99h+"},
		{"way over 99 hours", 500 * time.Hour, "99h+"},
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

func TestRepeat(t *testing.T) {
	tests := []struct {
		name string
		s    string
		n    int
		want string
	}{
		{"positive count", "a", 3, "aaa"},
		{"zero count", "a", 0, ""},
		{"negative count", "a", -1, ""},
		{"empty string", "", 5, ""},
		{"unicode", "█", 3, "███"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := repeat(tt.s, tt.n)
			if got != tt.want {
				t.Errorf("repeat(%q, %d) = %q, want %q", tt.s, tt.n, got, tt.want)
			}
		})
	}
}

func TestBarRate(t *testing.T) {
	// Test rate when stopped
	bar := NewBar("test", 1000, 0)
	bar.started = time.Now().Add(-10 * time.Second)
	bar.Set(1000)

	rate := bar.rate()
	// Should be approximately 100 bytes/second (1000 bytes / 10 seconds)
	if rate < 90 || rate > 110 {
		t.Errorf("rate() = %f, want approximately 100", rate)
	}
}

func TestBarRateWithBuckets(t *testing.T) {
	bar := NewBar("test", 1000, 0)

	// Add some buckets manually for testing
	now := time.Now()
	bar.buckets = []bucket{
		{updated: now.Add(-2 * time.Second), value: 100},
		{updated: now, value: 300},
	}

	rate := bar.rate()
	// Should be approximately 100 bytes/second ((300-100) / 2 seconds)
	if rate < 90 || rate > 110 {
		t.Errorf("rate() = %f, want approximately 100", rate)
	}
}

func TestBarBucketThrottle(t *testing.T) {
	bar := NewBar("test", 1000, 0)

	// First set should add a bucket
	bar.Set(100)
	if len(bar.buckets) != 1 {
		t.Errorf("buckets count = %d, want 1", len(bar.buckets))
	}

	// Immediate second set should not add a bucket (throttled)
	bar.Set(200)
	if len(bar.buckets) != 1 {
		t.Errorf("buckets count = %d, want 1 (throttled)", len(bar.buckets))
	}
}

func TestBarMaxBuckets(t *testing.T) {
	bar := NewBar("test", 1000, 0)
	bar.maxBuckets = 3

	// Add buckets with time gaps
	for i := 0; i < 5; i++ {
		bar.buckets = append(bar.buckets, bucket{
			updated: time.Now().Add(time.Duration(i) * 2 * time.Second),
			value:   int64(i * 100),
		})
	}

	// Trigger bucket cleanup by setting value with time gap
	bar.Set(600)

	if len(bar.buckets) > bar.maxBuckets {
		t.Errorf("buckets count = %d, should be <= %d", len(bar.buckets), bar.maxBuckets)
	}
}
