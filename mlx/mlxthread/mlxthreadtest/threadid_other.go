//go:build !darwin && !linux && !windows

package mlxthreadtest

func currentThreadID() uint64 {
	return 0
}
