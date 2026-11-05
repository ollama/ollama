//go:build linux

package mlxthread

import "syscall"

func currentThreadID() uint64 {
	return uint64(syscall.Gettid())
}
