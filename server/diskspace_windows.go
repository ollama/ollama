//go:build windows

package server

import "golang.org/x/sys/windows"

// availableBytes returns the number of bytes available in the filesystem
// containing the given path.
func availableBytes(path string) (int64, error) {
	pathPtr, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	var free uint64
	if err := windows.GetDiskFreeSpaceEx(pathPtr, &free, nil, nil); err != nil {
		return 0, err
	}
	return int64(free), nil
}
