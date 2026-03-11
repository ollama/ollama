//go:build linux || darwin

package fitcheck

import (
	"syscall"
)

// diskStats returns the total and available bytes for the filesystem that
// contains path. On Linux and macOS this uses syscall.Statfs.
func diskStats(path string) (total, avail uint64, err error) {
	var stat syscall.Statfs_t
	if err = syscall.Statfs(path, &stat); err != nil {
		return 0, 0, err
	}
	total = stat.Blocks * uint64(stat.Bsize)
	avail = stat.Bavail * uint64(stat.Bsize)
	return total, avail, nil
}
