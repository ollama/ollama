//go:build darwin

package mlx

import "syscall"

// currentThreadID returns the current kernel thread ID, used as the cache key
// for the thread-local default stream.
func currentThreadID() uint64 {
	id, _, _ := syscall.RawSyscall(syscall.SYS_THREAD_SELFID, 0, 0, 0)
	return uint64(id)
}
