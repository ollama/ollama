//go:build windows

package compatmigrate

import (
	"path/filepath"

	"golang.org/x/sys/windows"
)

func availableSpace(path string) (uint64, error) {
	root := filepath.VolumeName(path) + `\`
	p, err := windows.UTF16PtrFromString(root)
	if err != nil {
		return 0, err
	}

	var available uint64
	if err := windows.GetDiskFreeSpaceEx(p, &available, nil, nil); err != nil {
		return 0, err
	}

	return available, nil
}
