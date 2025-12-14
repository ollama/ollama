package server

import (
	"log/slog"
	"sync"
)

// PowerManager prevents the system from sleeping while Ollama has active runners.
// This is useful for long-running inference tasks that would be interrupted by sleep.
type PowerManager struct {
	mu       sync.Mutex
	refCount int
	active   bool
}

var globalPowerManager = &PowerManager{}

// PreventSleep increments the reference count and prevents system sleep if not already prevented.
// Call AllowSleep when the work is done.
func (pm *PowerManager) PreventSleep() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.refCount++
	if pm.refCount == 1 && !pm.active {
		if err := platformPreventSleep(); err != nil {
			slog.Debug("failed to prevent system sleep", "error", err)
		} else {
			pm.active = true
			slog.Debug("system sleep prevented")
		}
	}
}

// AllowSleep decrements the reference count and allows system sleep when count reaches zero.
func (pm *PowerManager) AllowSleep() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.refCount > 0 {
		pm.refCount--
	}

	if pm.refCount == 0 && pm.active {
		if err := platformAllowSleep(); err != nil {
			slog.Debug("failed to allow system sleep", "error", err)
		} else {
			pm.active = false
			slog.Debug("system sleep allowed")
		}
	}
}

// GetPowerManager returns the global power manager instance.
func GetPowerManager() *PowerManager {
	return globalPowerManager
}
