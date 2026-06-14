package llm

import (
	"sync"
	"testing"
)

func TestSleepInhibitionRefCounting(t *testing.T) {
	inhibitCount := 0
	uninhibitCount := 0
	var mu sync.Mutex

	origInhibit := inhibitSleep
	origUninhibit := uninhibitSleep
	defer func() {
		inhibitSleep = origInhibit
		uninhibitSleep = origUninhibit
	}()

	inhibitSleep = func() {
		mu.Lock()
		inhibitCount++
		mu.Unlock()
	}
	uninhibitSleep = func() {
		mu.Lock()
		uninhibitCount++
		mu.Unlock()
	}

	sleepInhibitCount.Store(0)

	// Acquire twice — should inhibit only once
	AcquireSleepInhibition()
	AcquireSleepInhibition()

	if inhibitCount != 1 {
		t.Errorf("expected inhibit called 1 time, got %d", inhibitCount)
	}
	if uninhibitCount != 0 {
		t.Errorf("expected uninhibit called 0 times, got %d", uninhibitCount)
	}

	// Release once — should still be inhibited
	ReleaseSleepInhibition()

	if inhibitCount != 1 {
		t.Errorf("expected inhibit still 1, got %d", inhibitCount)
	}
	if uninhibitCount != 0 {
		t.Errorf("expected uninhibit still 0, got %d", uninhibitCount)
	}

	// Release again — should now uninhibit
	ReleaseSleepInhibition()

	if inhibitCount != 1 {
		t.Errorf("expected inhibit still 1, got %d", inhibitCount)
	}
	if uninhibitCount != 1 {
		t.Errorf("expected uninhibit called 1 time, got %d", uninhibitCount)
	}
}

func TestSleepInhibitionConcurrent(t *testing.T) {
	inhibitCount := 0
	uninhibitCount := 0
	var mu sync.Mutex

	origInhibit := inhibitSleep
	origUninhibit := uninhibitSleep
	defer func() {
		inhibitSleep = origInhibit
		uninhibitSleep = origUninhibit
	}()

	inhibitSleep = func() {
		mu.Lock()
		inhibitCount++
		mu.Unlock()
	}
	uninhibitSleep = func() {
		mu.Lock()
		uninhibitCount++
		mu.Unlock()
	}

	sleepInhibitCount.Store(0)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			AcquireSleepInhibition()
		}()
	}
	wg.Wait()

	if inhibitCount != 1 {
		t.Errorf("expected inhibit called 1 time across 10 goroutines, got %d", inhibitCount)
	}

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ReleaseSleepInhibition()
		}()
	}
	wg.Wait()

	if inhibitCount != 1 {
		t.Errorf("expected inhibit still 1, got %d", inhibitCount)
	}
	if uninhibitCount != 1 {
		t.Errorf("expected uninhibit called 1 time, got %d", uninhibitCount)
	}
}

func TestSleepInhibitionNegativeGuard(t *testing.T) {
	origUninhibit := uninhibitSleep
	origInhibit := inhibitSleep
	defer func() {
		inhibitSleep = origInhibit
		uninhibitSleep = origUninhibit
	}()

	uninhibitCalled := false
	uninhibitSleep = func() {
		uninhibitCalled = true
	}
	inhibitSleep = func() {}

	sleepInhibitCount.Store(0)

	// Release with no acquires — should not call uninhibit
	ReleaseSleepInhibition()

	if uninhibitCalled {
		t.Error("uninhibit should not be called when counter is already 0")
	}
}
