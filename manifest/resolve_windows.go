package manifest

import (
	"os"
	"strings"

	"golang.org/x/sys/windows"
)

// resolveLinks returns the final path of an existing file or directory.
//
// filepath.EvalSymlinks fails on Windows when the path crosses a directory
// junction or an NTFS volume mount point, so ask the OS for the final path
// instead. It follows symlinks the same way, and a missing or dangling path
// still reports an error that os.IsNotExist recognizes.
// GetFinalPathNameByHandle flags (fileapi.h); golang.org/x/sys/windows does not define them.
const (
	fileNameNormalized = 0x0
	volumeNameDOS      = 0x0
)

func resolveLinks(path string) (string, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return "", &os.PathError{Op: "resolve", Path: path, Err: err}
	}
	h, err := windows.CreateFile(p, 0,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return "", &os.PathError{Op: "resolve", Path: path, Err: err}
	}
	defer windows.CloseHandle(h)

	buf := make([]uint16, windows.MAX_PATH)
	for {
		n, err := windows.GetFinalPathNameByHandle(h, &buf[0], uint32(len(buf)), fileNameNormalized|volumeNameDOS)
		if err != nil {
			return "", &os.PathError{Op: "resolve", Path: path, Err: err}
		}
		if n < uint32(len(buf)) {
			return trimExtendedPrefix(windows.UTF16ToString(buf[:n])), nil
		}
		buf = make([]uint16, n)
	}
}

func trimExtendedPrefix(p string) string {
	if strings.HasPrefix(p, `\?\UNC\`) {
		return `\` + p[len(`\?\UNC\`):]
	}
	return strings.TrimPrefix(p, `\?\`)
}
