package server

import "golang.org/x/sys/windows"

// preventSleep keeps the system awake for the duration of a long operation.
// Call the returned function (or defer it) to restore normal sleep behavior.
func preventSleep() func() {
	windows.SetThreadExecutionState(windows.ES_CONTINUOUS | windows.ES_SYSTEM_REQUIRED) //nolint:errcheck
	return func() {
		windows.SetThreadExecutionState(windows.ES_CONTINUOUS) //nolint:errcheck
	}
}
