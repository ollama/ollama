//go:build !windows

package server

import "syscall"

// availableBytes returns the number of bytes available in the filesystem
// containing the given path.
func availableBytes(path string) (int64, error) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0, err
	}
	//nolint:unconvert // Bsize type varies by platform (int32 on darwin, int64 on linux)
	return int64(stat.Bavail) * int64(stat.Bsize), nil
}
