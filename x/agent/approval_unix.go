//go:build !windows

package agent

import (
	"syscall"
	"time"
)

// flushStdin drains any buffered input from stdin.
// This prevents leftover input from previous operations from affecting the selector.
func flushStdin(fd int) {
	if err := syscall.SetNonblock(fd, true); err != nil {
		return
	}
	defer syscall.SetNonblock(fd, false)

	time.Sleep(5 * time.Millisecond)

	buf := make([]byte, 256)
	for {
		n, err := syscall.Read(fd, buf)
		if n <= 0 || err != nil {
			break
		}
	}
}
