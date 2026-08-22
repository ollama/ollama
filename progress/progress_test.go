package progress

import (
	"bytes"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestStopIsIdempotent pins the contract cmd relies on: a second Stop (e.g. a
// deferred StopAndClear after an explicit one) reports false so callers don't
// emit trailing output twice.
func TestStopIsIdempotent(t *testing.T) {
	for _, tt := range []struct {
		name string
		stop func(*Progress) bool
	}{
		{"Stop", (*Progress).Stop},
		{"StopAndClear", (*Progress).StopAndClear},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := NewProgress(&bytes.Buffer{})
			if !tt.stop(p) {
				t.Fatal("first stop should report true")
			}
			if tt.stop(p) {
				t.Fatal("second stop should report false")
			}
		})
	}
}

// TestStopStopsRendering verifies the render loop exits, so no further output
// is written after Stop returns.
func TestStopStopsRendering(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	p.Add("bar", NewBar("test", 100, 0))
	time.Sleep(250 * time.Millisecond) // let a few ticks render
	p.Stop()

	settled := buf.Len()
	time.Sleep(300 * time.Millisecond) // several tick intervals
	if buf.Len() != settled {
		t.Fatalf("render loop still writing after Stop: %d -> %d bytes", settled, buf.Len())
	}
}

// TestStopReapsGoroutines verifies Stop reaps the render and spinner
// goroutines rather than leaving them parked on a ticker. Spinners previously
// ran until the process exited, so a long-lived `ollama run` leaked one per
// progress bar it displayed.
func TestStopReapsGoroutines(t *testing.T) {
	base := runtime.NumGoroutine()

	const cycles = 50
	for range cycles {
		p := NewProgress(&bytes.Buffer{})
		p.Add("spin", NewSpinner("working"))
		p.Add("bar", NewBar("dl", 100, 0))
		p.Stop()
	}

	// Allow a moment for the goroutines to be descheduled before counting.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && runtime.NumGoroutine() > base+2 {
		time.Sleep(50 * time.Millisecond)
	}

	if got := runtime.NumGoroutine(); got > base+2 {
		t.Fatalf("goroutines after %d start/stop cycles: got %d, want <= %d", cycles, got, base+2)
	}
}

// TestSpinnerStopStopsAnimating verifies Spinner.Stop halts its goroutine.
func TestSpinnerStopStopsAnimating(t *testing.T) {
	s := NewSpinner("working")
	time.Sleep(250 * time.Millisecond)
	s.Stop()

	s.mu.Lock()
	settled := s.value
	s.mu.Unlock()

	time.Sleep(300 * time.Millisecond)

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.value != settled {
		t.Fatalf("spinner still animating after Stop: %d -> %d", settled, s.value)
	}
}

// TestNonTTYNoDuplicateLines verifies that in non-TTY mode a spinner
// message is printed exactly once regardless of how many render ticks fire.
func TestNonTTYNoDuplicateLines(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)

	p.Add("status", NewSpinner("pulling manifest"))

	// Wait for several render ticks (ticker fires every 100ms)
	time.Sleep(350 * time.Millisecond)

	p.Stop()

	got := strings.TrimSpace(buf.String())
	lines := strings.Split(got, "\n")
	count := 0
	for _, l := range lines {
		if strings.Contains(l, "pulling manifest") {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected exactly 1 line containing %q, got %d:\n%s", "pulling manifest", count, buf.String())
	}
}

// TestNonTTYMultipleStates verifies each distinct status prints once in non-TTY mode.
func TestNonTTYMultipleStates(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)

	s1 := NewSpinner("pulling manifest")
	p.Add("s1", s1)
	time.Sleep(150 * time.Millisecond)

	s1.Stop()
	s2 := NewSpinner("verifying sha256 digest")
	p.Add("s2", s2)
	time.Sleep(150 * time.Millisecond)

	p.Stop()

	out := buf.String()
	if strings.Count(out, "pulling manifest") != 1 {
		t.Errorf("expected 1 occurrence of %q:\n%s", "pulling manifest", out)
	}
	if strings.Count(out, "verifying sha256 digest") != 1 {
		t.Errorf("expected 1 occurrence of %q:\n%s", "verifying sha256 digest", out)
	}
}

// TestPlainStringerSpinner checks that Spinner.PlainString returns just the message.
func TestPlainStringerSpinner(t *testing.T) {
	s := NewSpinner("pulling manifest")
	got := s.PlainString()
	if got != "pulling manifest" {
		t.Errorf("PlainString() = %q, want %q", got, "pulling manifest")
	}
	s.Stop()
}

// TestPlainStringerBar checks that Bar.PlainString returns message + percentage.
func TestPlainStringerBar(t *testing.T) {
	b := NewBar("pulling abc123:", 100, 0)
	b.Set(50)
	got := b.PlainString()
	if !strings.Contains(got, "pulling abc123:") || !strings.Contains(got, "50%") {
		t.Errorf("PlainString() = %q, want message and percentage", got)
	}
}
