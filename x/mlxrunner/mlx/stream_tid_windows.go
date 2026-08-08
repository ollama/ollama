//go:build windows

package mlx

import "golang.org/x/sys/windows"

// currentThreadID returns the current kernel thread ID, used as the cache key
// for the thread-local default stream.
func currentThreadID() uint64 {
	return uint64(windows.GetCurrentThreadId())
}
