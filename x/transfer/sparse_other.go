//go:build !windows

package transfer

import "os"

// setSparse is a no-op on non-Windows platforms.
// On Windows, this sets the FSCTL_SET_SPARSE attribute which allows the OS
// to not allocate disk blocks for zero-filled regions. This is useful for
// partial downloads where not all data has been written yet. On Unix-like
// systems, filesystems typically handle this automatically (sparse by default).
func setSparse(_ *os.File) {}
