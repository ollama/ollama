package server

import (
	"testing"
)

func TestPowerManagerRefCount(t *testing.T) {
	pm := &PowerManager{}

	// Initial state
	if pm.refCount != 0 {
		t.Errorf("expected refCount to be 0, got %d", pm.refCount)
	}

	// First PreventSleep should increment refCount
	pm.PreventSleep()
	if pm.refCount != 1 {
		t.Errorf("expected refCount to be 1, got %d", pm.refCount)
	}

	// Second PreventSleep should increment refCount
	pm.PreventSleep()
	if pm.refCount != 2 {
		t.Errorf("expected refCount to be 2, got %d", pm.refCount)
	}

	// First AllowSleep should decrement refCount
	pm.AllowSleep()
	if pm.refCount != 1 {
		t.Errorf("expected refCount to be 1, got %d", pm.refCount)
	}

	// Second AllowSleep should decrement refCount to 0
	pm.AllowSleep()
	if pm.refCount != 0 {
		t.Errorf("expected refCount to be 0, got %d", pm.refCount)
	}

	// Extra AllowSleep should not go negative
	pm.AllowSleep()
	if pm.refCount != 0 {
		t.Errorf("expected refCount to stay at 0, got %d", pm.refCount)
	}
}

func TestGetPowerManager(t *testing.T) {
	pm1 := GetPowerManager()
	pm2 := GetPowerManager()

	if pm1 != pm2 {
		t.Error("expected GetPowerManager to return the same instance")
	}
}
