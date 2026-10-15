//go:build !windows

package launch

import (
	"os"

	"golang.org/x/sys/unix"
)

func tryLockCodexAppHistoryFile(file *os.File) error {
	return unix.Flock(int(file.Fd()), unix.LOCK_EX|unix.LOCK_NB)
}
