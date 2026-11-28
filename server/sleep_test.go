package server

import (
	"sync"
	"testing"
)

func TestSleepInhibitorRefCounting(t *testing.T) {
	si := &sleepInhibitor{}

	// First PreventSleep should transition 0→1
	si.PreventSleep()
	// Second PreventSleep should increment to 2
	si.PreventSleep()

	// First AllowSleep should decrement to 1 (still preventing)
	si.AllowSleep()
	// Second AllowSleep should transition 1→0 (allow sleep)
	si.AllowSleep()

	// Extra AllowSleep should be a no-op (not go negative)
	si.AllowSleep()
}

func TestSleepInhibitorConcurrent(t *testing.T) {
	si := &sleepInhibitor{}
	var wg sync.WaitGroup

	for range 100 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			si.PreventSleep()
		}()
		go func() {
			defer wg.Done()
			si.AllowSleep()
		}()
	}

	wg.Wait()
	si.Close()
}

func TestSleepInhibitorClose(t *testing.T) {
	si := &sleepInhibitor{}

	// Close with active prevention should not panic
	si.PreventSleep()
	si.PreventSleep()
	si.Close()

	// Close when already closed should not panic
	si.Close()
}
