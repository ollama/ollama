package progress

import (
	"bytes"
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
			p.Add("key", NewSpinner("test"))
		}()
	}

	wg.Wait()
	p.Stop()
}
