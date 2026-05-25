//go:build windows

package wakelock

// Windows wake-lock is currently a stub. The intended implementation calls
// kernel32!SetThreadExecutionState with ES_CONTINUOUS|ES_SYSTEM_REQUIRED
// (and optionally ES_AWAYMODE_REQUIRED on server SKUs) on acquire and
// resets it to ES_CONTINUOUS on release. The flag persists per-thread, so
// the call must be issued from a long-lived OS thread that is locked with
// runtime.LockOSThread for the lifetime of the assertion -- typically a
// dedicated goroutine that owns the assertion.
//
// References:
//   https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
//   https://github.com/ollama/ollama/issues/4072
//
// When implementing, plumb through golang.org/x/sys/windows.NewLazySystemDLL
// rather than adding a new direct cgo dependency.

func init() {
	newAssertion = func(reason string) (assertion, error) {
		return noopAssertion{}, nil
	}
}

type noopAssertion struct{}

func (noopAssertion) release() {}
