package progress

import (
	"bytes"
	"sync"
	"testing"
	"time"
)

// TestProgressConcurrentStopStart verifies that concurrent Stop() calls
// and the internal start() goroutine don't race on ticker/states fields.
func TestProgressConcurrentStopStart(t *testing.T) {
	t.Parallel()

	var buf bytes.Buffer
	p := NewProgress(&buf)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		time.Sleep(1 * time.Millisecond)
		p.Stop()
	}()

	go func() {
		defer wg.Done()
		time.Sleep(1 * time.Millisecond)
		p.StopAndClear()
	}()

	wg.Wait()
}

// TestProgressMultipleStops verifies that Stop() is idempotent and can be
// called multiple times without panicking or racing.
func TestProgressMultipleStops(t *testing.T) {
	t.Parallel()

	var buf bytes.Buffer
	p := NewProgress(&buf)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(time.Millisecond)
			p.Stop()
		}()
	}

	wg.Wait()
}

// mockState is a simple State implementation for testing
type mockState struct{}

func (mockState) String() string { return "test" }

// TestProgressConcurrentAdd verifies that Add() can be called concurrently
// with Stop() without racing on states slice.
func TestProgressConcurrentAdd(t *testing.T) {
	t.Parallel()

	var buf bytes.Buffer
	p := NewProgress(&buf)

	var wg sync.WaitGroup

	// Concurrent Add calls
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			p.Add("", mockState{})
		}(i)
	}

	// Concurrent Stop calls
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(2 * time.Millisecond)
		p.Stop()
	}()

	wg.Wait()
}

// TestProgressImmediateStop verifies the race when Stop() is called
// immediately after NewProgress, before start() writes ticker.
func TestProgressImmediateStop(t *testing.T) {
	t.Parallel()

	for i := 0; i < 100; i++ {
		var buf bytes.Buffer
		p := NewProgress(&buf)
		// No sleep - try to hit the race window
		p.Stop()
	}
}
