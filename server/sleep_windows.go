package server

import (
	"log/slog"
	"sync"
	"syscall"
)

var (
	kernel32                   = syscall.NewLazyDLL("kernel32.dll")
	procSetThreadExecutionState = kernel32.NewProc("SetThreadExecutionState")
)

const (
	esContinuous      = 0x80000000
	esSystemRequired  = 0x00000001
)

type sleepInhibitor struct {
	mu       sync.Mutex
	refCount int
}

func (si *sleepInhibitor) PreventSleep() {
	if si == nil {
		return
	}
	si.mu.Lock()
	defer si.mu.Unlock()
	si.refCount++
	if si.refCount == 1 {
		r, _, _ := procSetThreadExecutionState.Call(uintptr(esContinuous | esSystemRequired))
		if r == 0 {
			slog.Warn("failed to prevent system sleep")
		} else {
			slog.Debug("system sleep prevented")
		}
	}
}

func (si *sleepInhibitor) AllowSleep() {
	if si == nil {
		return
	}
	si.mu.Lock()
	defer si.mu.Unlock()
	if si.refCount <= 0 {
		return
	}
	si.refCount--
	if si.refCount == 0 {
		r, _, _ := procSetThreadExecutionState.Call(uintptr(esContinuous))
		if r == 0 {
			slog.Warn("failed to allow system sleep")
		} else {
			slog.Debug("system sleep allowed")
		}
	}
}

func (si *sleepInhibitor) Close() {
	if si == nil {
		return
	}
	si.mu.Lock()
	defer si.mu.Unlock()
	if si.refCount > 0 {
		si.refCount = 0
		procSetThreadExecutionState.Call(uintptr(esContinuous)) //nolint:errcheck
		slog.Debug("system sleep allowed on close")
	}
}
