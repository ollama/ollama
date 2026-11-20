package mlxrunner

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

func lockCacheFile(f *os.File, exclusive bool) (bool, error) {
	flags := uint32(windows.LOCKFILE_FAIL_IMMEDIATELY)
	if exclusive {
		flags |= windows.LOCKFILE_EXCLUSIVE_LOCK
	}
	err := windows.LockFileEx(windows.Handle(f.Fd()), flags, 0, 1, 0, &windows.Overlapped{})
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
		return false, nil
	}
	return err == nil, err
}

func cacheDiskSpace(path string) (int64, int64, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, 0, err
	}
	var total, free uint64
	err = windows.GetDiskFreeSpaceEx(p, &free, &total, nil)
	return int64(total), int64(free), err
}
