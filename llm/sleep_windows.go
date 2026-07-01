package llm

import (
	"runtime"
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
	sleepInhibitCh             = make(chan bool, 1)
)

func init() {
	go func() {
		runtime.LockOSThread()
		for v := range sleepInhibitCh {
			if v {
				procSetThreadExecutionState.Call(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
			} else {
				procSetThreadExecutionState.Call(ES_CONTINUOUS)
			}
		}
	}()

	inhibitSleep = func() {
		select {
		case sleepInhibitCh <- true:
		default:
		}
	}
	uninhibitSleep = func() {
		select {
		case sleepInhibitCh <- false:
		default:
		}
	}
}
