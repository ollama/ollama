// Package wakelock provides a reference-counted wake lock that prevents the
// host system from going to sleep while inference requests are in flight.
//
// The wake lock is acquired on the first Acquire call (count 0 -> 1) and
// released when the count returns to zero. Concurrent callers are safe.
//
// Each supported platform provides an [assertion] implementation that
// interacts with the OS power-management facility. Unsupported platforms
// receive a no-op stub so callers do not need to branch on GOOS.
package wakelock

import (
	"log/slog"
	"sync"
)

// WakeLock is a reference-counted wake lock. The zero value is not usable;
// construct one with [New].
type WakeLock struct {
	reason string

	mu        sync.Mutex
	count     int
	assertion assertion
	closed    bool
}

// assertion is the platform-specific OS power assertion. Implementations
// must be safe to construct and release exactly once; double release must
// be a no-op.
type assertion interface {
	release()
}

// newAssertion is set by per-platform files and returns the OS power
// assertion (or an error if it could not be obtained). Stub platforms
// return a no-op assertion and a nil error.
var newAssertion func(reason string) (assertion, error)

// Disable replaces the platform assertion with a no-op. Intended for tests
// that need a real *WakeLock but don't want to touch the OS.
func Disable() {
	newAssertion = func(reason string) (assertion, error) {
		return noopForDisable{}, nil
	}
}

type noopForDisable struct{}

func (noopForDisable) release() {}

// New returns a new wake lock. The reason string is supplied to the OS
// when an assertion is taken; it appears in tools such as
// `pmset -g assertions` on macOS.
func New(reason string) *WakeLock {
	return &WakeLock{reason: reason}
}

// Acquire increments the in-flight reference count. When the count
// transitions from zero to one, the underlying OS assertion is taken.
// Errors from the OS are logged but not returned: failing to prevent
// sleep should never fail an inference request.
func (w *WakeLock) Acquire() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return
	}
	w.count++
	if w.count != 1 || w.assertion != nil {
		return
	}
	if newAssertion == nil {
		return
	}
	a, err := newAssertion(w.reason)
	if err != nil {
		slog.Debug("failed to acquire wake lock", "error", err)
		return
	}
	w.assertion = a
}

// Release decrements the in-flight reference count. When the count
// returns to zero the underlying OS assertion is released. Releasing
// below zero is a no-op so callers don't have to worry about over-release
// during shutdown.
func (w *WakeLock) Release() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.count == 0 {
		return
	}
	w.count--
	if w.count > 0 {
		return
	}
	if w.assertion != nil {
		w.assertion.release()
		w.assertion = nil
	}
}

// Close releases any held OS assertion and prevents further acquires.
// It is safe to call multiple times. Use this from a signal handler or
// process-exit path so the OS-level assertion isn't orphaned.
func (w *WakeLock) Close() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.closed = true
	w.count = 0
	if w.assertion != nil {
		w.assertion.release()
		w.assertion = nil
	}
}

// Count returns the current reference count. Exposed for tests.
func (w *WakeLock) Count() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.count
}

// Held reports whether an OS assertion is currently held. Exposed for tests.
func (w *WakeLock) Held() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.assertion != nil
}
