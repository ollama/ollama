//go:build windows

package mlxthreadtest

import "golang.org/x/sys/windows"

func currentThreadID() uint64 {
	return uint64(windows.GetCurrentThreadId())
}
