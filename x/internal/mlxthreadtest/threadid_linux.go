//go:build linux

package mlxthreadtest

import "syscall"

func currentThreadID() uint64 {
	return uint64(syscall.Gettid())
}
