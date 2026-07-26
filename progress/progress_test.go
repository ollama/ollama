package progress

import (
	"io"
	"runtime"
	"testing"
	"time"
)

// waitForGoroutines polls the live goroutine count until it drops to at most
// want, or the deadline passes. It returns the last observed count. We poll
// (rather than sampling once) because goroutine teardown is asynchronous.
func waitForGoroutines(want int, timeout time.Duration) int {
	deadline := time.Now().Add(timeout)
	var n int
	for {
		runtime.GC()
		n = runtime.NumGoroutine()
		if n <= want || time.Now().After(deadline) {
			return n
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestProgressStopReleasesGoroutine verifies that Stop terminates the
// background render goroutine started by NewProgress. Before the fix the
// goroutine blocked forever on `for range p.ticker.C` (Ticker.Stop does not
// close the channel), leaking one goroutine per Progress instance.
func TestProgressStopReleasesGoroutine(t *testing.T) {
	// Let any goroutines from earlier tests settle first.
	baseline := waitForGoroutines(runtime.NumGoroutine(), 100*time.Millisecond)

	const n = 50
	for range n {
		p := NewProgress(io.Discard)
		if !p.Stop() {
			t.Fatal("Stop() returned false on a running Progress")
		}
	}

	// After stopping, the goroutine count must return to ~baseline. Allow a
	// tiny slack for unrelated runtime goroutines, but n (=50) leaked
	// goroutines is far above any noise.
	const slack = 5
	if got := waitForGoroutines(baseline+slack, 2*time.Second); got > baseline+slack {
		t.Errorf("goroutine leak: started %d Progress instances, %d goroutines still live (baseline %d)", n, got, baseline)
	}
}

// TestProgressStopIsIdempotent verifies the bool contract of Stop: the first
// call stops the Progress (true), later calls are no-ops (false), and calling
// it repeatedly does not panic (e.g. closing a channel twice).
func TestProgressStopIsIdempotent(t *testing.T) {
	p := NewProgress(io.Discard)

	if !p.Stop() {
		t.Fatal("first Stop() = false, want true")
	}
	if p.Stop() {
		t.Error("second Stop() = true, want false")
	}
	if p.StopAndClear() {
		t.Error("StopAndClear() after Stop() = true, want false")
	}
}
