package progress

import (
	"bytes"
	"sync"
	"testing"
	"time"
)

func TestProgressConcurrentStopStart(t *testing.T) {
	// Reproduces the race: start() goroutine writes ticker
	// while Stop() reads it, both without holding mutex.
	var buf bytes.Buffer
	p := NewProgress(&buf)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		// Give start() time to spawn
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
	// Reproduces the race: Add() writes states while
	// stop() reads states.
	var buf bytes.Buffer
	p := NewProgress(&buf)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			p.Add("test", &Spinner{})
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
