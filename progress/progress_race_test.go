package progress

import (
	"bytes"
	"sync"
	"testing"
	"time"
)

func TestProgressConcurrentStopAndClear(t *testing.T) {
	// Concurrent shutdown calls must serialize their final output writes.
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

func TestProgressConcurrentAddAndStop(t *testing.T) {
	// Adding a state can overlap with shutdown.
	var buf bytes.Buffer
	p := NewProgress(&buf)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			spinner := NewSpinner("test")
			p.Add("test", spinner)
			defer spinner.Stop()
			time.Sleep(100 * time.Microsecond)
		}
	}()

	go func() {
		defer wg.Done()
		time.Sleep(1 * time.Millisecond)
		p.Stop()
	}()

	wg.Wait()
}
