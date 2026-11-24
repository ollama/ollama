//go:build windows

package transfer

import (
	"os"

	"golang.org/x/sys/windows"
)

// setSparse sets the FSCTL_SET_SPARSE attribute on Windows files.
// This allows the OS to not allocate disk blocks for zero-filled regions,
// which is useful for large files that may not be fully written (e.g., partial
// downloads). Without this, Windows may pre-allocate disk space for the full
// file size even if most of it is zeros.
//
// Note: Errors are intentionally ignored because:
// 1. The file will still work correctly without sparse support
// 2. Not all Windows filesystems support sparse files (e.g., FAT32)
// 3. This is an optimization, not a requirement
func setSparse(file *os.File) {
	var bytesReturned uint32
	_ = windows.DeviceIoControl(
		windows.Handle(file.Fd()),
		windows.FSCTL_SET_SPARSE,
		nil, 0,
		nil, 0,
		&bytesReturned,
		nil,
	)
}
