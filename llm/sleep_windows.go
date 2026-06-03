package llm

import (
	"syscall"
)

const (
	ES_CONTINUOUS       = 0x80000000
	ES_SYSTEM_REQUIRED  = 0x00000001
	ES_DISPLAY_REQUIRED = 0x00000002
)

var (
	kernel32                   = syscall.NewLazyDLL("kernel32.dll")
	procSetThreadExecutionState = kernel32.NewProc("SetThreadExecutionState")
)

func init() {
	inhibitSleep = func() {
		procSetThreadExecutionState.Call(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
	}
	uninhibitSleep = func() {
		procSetThreadExecutionState.Call(ES_CONTINUOUS)
	}
}
