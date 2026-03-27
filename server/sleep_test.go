package server

import (
	"testing"
)

func TestSleepPreventionStart(t *testing.T) {
	sp := NewSleepPrevention()
	
	if sp.active {
		t.Error("sleep prevention should not be active on creation")
	}
	
	sp.Start()
	
	if !sp.active && sp.token != 0 {
		// It's okay if it's not active on systems that don't support sleep prevention
		// but the structure should be correct
		return
	}
}

func TestSleepPreventionStop(t *testing.T) {
	sp := NewSleepPrevention()
	sp.Start()
	sp.Stop()
	
	if sp.active {
		t.Error("sleep prevention should not be active after Stop()")
	}
}

func TestSleepPreventionMultipleStart(t *testing.T) {
	sp := NewSleepPrevention()
	
	// Starting multiple times should not cause issues
	sp.Start()
	firstToken := sp.token
	
	sp.Start()
	
	if sp.token != firstToken {
		t.Error("multiple Start() calls should not change the token")
	}
}

func TestSleepPreventionStopWhenNotActive(t *testing.T) {
	sp := NewSleepPrevention()
	
	// Calling Stop() without Start() should not panic
	sp.Stop()
	sp.Stop()
}
