package progress

import (
	"bufio"
	"bytes"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"golang.org/x/term"
)

type testState string

func (s testState) String() string { return string(s) }

func TestStopAndClearOnlyClearsVisibleLines(t *testing.T) {
	_, termHeight, err := term.GetSize(int(os.Stderr.Fd()))
	if err != nil {
		termHeight = defaultTermHeight
	}

	var buf bytes.Buffer
	p := &Progress{
		w:    bufio.NewWriter(&buf),
		done: make(chan struct{}),
	}
	for range termHeight + 1 {
		p.states = append(p.states, testState("working"))
	}

	if !p.StopAndClear() {
		t.Fatal("first stop should report true")
	}

	const clearLine = "\033[2K\033[1G"
	if got := strings.Count(buf.String(), clearLine); got != termHeight {
		t.Fatalf("cleared %d lines, want %d visible lines", got, termHeight)
	}
}

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
