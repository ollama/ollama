package llm

import (
	"log/slog"
	"sync"

	"golang.org/x/sys/windows"
)

const (
	esContinuous     = 0x80000000
	esSystemRequired = 0x00000001
	esDisplayRequired = 0x00000002
)

var (
	setThreadExecutionState = windows.NewLazySystemDLL("kernel32.dll").NewProc("SetThreadExecutionState")
	sleepMu                 sync.Mutex
	sleepActive             bool
)

func preventSleep() (func(), error) {
	sleepMu.Lock()
	defer sleepMu.Unlock()

	if sleepActive {
		return func() {}, nil
	}

	ret, _, err := setThreadExecutionState.Call(esContinuous | esSystemRequired | esDisplayRequired)
	if ret == 0 {
		slog.Debug("failed to set thread execution state to prevent sleep", "error", err)
		return func() {}, nil
	}
	sleepActive = true
	slog.Debug("sleep prevention enabled via SetThreadExecutionState")

	return func() {
		sleepMu.Lock()
		defer sleepMu.Unlock()
		if sleepActive {
			setThreadExecutionState.Call(esContinuous)
			sleepActive = false
			slog.Debug("sleep prevention disabled")
		}
	}, nil
}
