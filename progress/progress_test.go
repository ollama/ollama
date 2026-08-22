package progress

import (
	"bytes"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestProgressStopRace reproduces the data race reported in #16271:
// concurrent Stop/StopAndClear calls race with the internal start() goroutine
// on the ticker and states fields.
func TestProgressStopRace(t *testing.T) {
	for range 200 {
		var buf bytes.Buffer
		p := NewProgress(&buf)

		var wg sync.WaitGroup
		wg.Add(2)

		go func() {
			defer wg.Done()
			time.Sleep(time.Millisecond)
			p.Stop()
		}()

		go func() {
			defer wg.Done()
			time.Sleep(time.Millisecond)
			p.StopAndClear()
		}()

		wg.Wait()
	}
}

// TestProgressStopIdempotent verifies that Stop can be called multiple times
// safely (the deferred Stop pattern used in cmd/cmd.go).
func TestProgressStopIdempotent(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	time.Sleep(5 * time.Millisecond)

	if !p.Stop() {
		t.Error("first Stop() should return true")
	}
	if p.Stop() {
		t.Error("second Stop() should return false")
	}
}

// TestProgressAddThenStop verifies that states added via Add are visible after
// Stop without a data race.
func TestProgressAddThenStop(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)

	var wg sync.WaitGroup
	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p.Add("key", testState("test"))
		}()
	}

	wg.Wait()
	p.Stop()
}

// testState is a race-free State stub. The concurrency tests here avoid
// NewSpinner deliberately: Spinner has internal data races of its own
// (#16271) that are separate from the Progress lifecycle covered by this
// file.
type testState string

func (s testState) String() string { return string(s) }

// TestProgressImmediateStop verifies the initialization lifecycle: a Stop
// that runs before the render goroutine's first tick must still stop the
// just-created ticker and report that it did.
func TestProgressImmediateStop(t *testing.T) {
	for range 1000 {
		var buf bytes.Buffer
		p := NewProgress(&buf)
		if !p.Stop() {
			t.Fatal("immediate Stop must stop the just-created ticker")
		}
	}
}

// TestProgressNoOutputAfterStop verifies that once Stop returns, the render
// goroutine writes nothing further to the shared writer.
func TestProgressNoOutputAfterStop(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	p.Add("key", testState("test"))
	time.Sleep(150 * time.Millisecond)
	p.Stop()

	n := buf.Len()
	time.Sleep(300 * time.Millisecond)
	if got := buf.Len(); got != n {
		t.Errorf("output grew after Stop: %d -> %d bytes", n, got)
	}

	// a render that races the shutdown must be a no-op once stopped
	p.render()
	if got := buf.Len(); got != n {
		t.Errorf("render after Stop wrote output: %d -> %d bytes", n, got)
	}
}

// TestProgressStopReleasesGoroutine verifies the render goroutine exits after
// Stop instead of leaking.
func TestProgressStopReleasesGoroutine(t *testing.T) {
	before := runtime.NumGoroutine()

	for range 50 {
		var buf bytes.Buffer
		p := NewProgress(&buf)
		p.Stop()
	}

	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > before+2 {
		if time.Now().After(deadline) {
			t.Fatalf("render goroutines leaked: %d before, %d after", before, runtime.NumGoroutine())
		}
		time.Sleep(10 * time.Millisecond)
	}
}
