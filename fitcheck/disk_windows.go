//go:build windows

package fitcheck

import (
	"log/slog"
	"syscall"
	"unsafe"
)

// diskStats returns the total and available bytes for the filesystem that
// contains path. Uses GetDiskFreeSpaceExW on Windows.
func diskStats(path string) (total, avail uint64, err error) {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	getDiskFreeSpaceEx := kernel32.NewProc("GetDiskFreeSpaceExW")

	pathPtr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		slog.Warn("fitcheck: failed to convert models path for disk stats", "error", err)
		return 0, 0, err
	}

	var freeBytesAvailable, totalBytes, totalFreeBytes uint64
	r1, _, lastErr := getDiskFreeSpaceEx.Call(
		uintptr(unsafe.Pointer(pathPtr)),
		uintptr(unsafe.Pointer(&freeBytesAvailable)),
		uintptr(unsafe.Pointer(&totalBytes)),
		uintptr(unsafe.Pointer(&totalFreeBytes)),
	)
	if r1 == 0 {
		slog.Warn("fitcheck: GetDiskFreeSpaceExW failed", "error", lastErr)
		return 0, 0, lastErr
	}

	return totalBytes, freeBytesAvailable, nil
}
