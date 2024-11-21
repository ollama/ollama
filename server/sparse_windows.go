package server

import (
	"os"

	"golang.org/x/sys/windows"
)

func setSparse(file *os.File) {
	// exFat (and other FS types) don't support sparse files, so ignore errors
	windows.DeviceIoControl( //nolint:errcheck
		windows.Handle(file.Fd()), windows.FSCTL_SET_SPARSE,
		nil, 0,
		nil, 0,
		nil, nil,
	)
}
