//go:build !windows

package mlxrunner

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

func lockCacheFile(f *os.File, exclusive bool) (bool, error) {
	mode := unix.LOCK_SH | unix.LOCK_NB
	if exclusive {
		mode = unix.LOCK_EX | unix.LOCK_NB
	}
	err := unix.Flock(int(f.Fd()), mode)
	if errors.Is(err, unix.EWOULDBLOCK) {
		return false, nil
	}
	return err == nil, err
}

func cacheDiskSpace(path string) (int64, int64, error) {
	var stat unix.Statfs_t
	if err := unix.Statfs(path, &stat); err != nil {
		return 0, 0, err
	}
	blockSize := uint64(stat.Bsize)
	return int64(stat.Blocks * blockSize), int64(stat.Bavail * blockSize), nil
}
