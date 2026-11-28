package server

import (
	"testing"
)

func TestPreventSleep(t *testing.T) {
	restore := preventSleep()
	if restore == nil {
		t.Fatal("preventSleep returned nil restore function")
	}
	// Calling restore should not panic
	restore()
}

func TestPreventSleepMultipleCalls(t *testing.T) {
	// Multiple concurrent inference requests should each be able
	// to call preventSleep without panicking
	for range 5 {
		restore := preventSleep()
		defer restore()
	}
}
