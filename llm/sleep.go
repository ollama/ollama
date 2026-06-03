package llm

import "sync/atomic"

var (
	sleepInhibitCount atomic.Int32
)

func AcquireSleepInhibition() {
	if sleepInhibitCount.Add(1) == 1 {
		inhibitSleep()
	}
}

func ReleaseSleepInhibition() {
	if sleepInhibitCount.Add(-1) == 0 {
		uninhibitSleep()
	}
}

// Default no-op implementations — overridden by platform-specific files via init().
var inhibitSleep = func()   {}
var uninhibitSleep = func() {}
