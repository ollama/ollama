//go:build darwin

package mlxthread

import "syscall"

func currentThreadID() uint64 {
	id, _, _ := syscall.RawSyscall(syscall.SYS_THREAD_SELFID, 0, 0, 0)
	return uint64(id)
}
