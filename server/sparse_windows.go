package server

import (
	"os"

	"golang.org/x/sys/windows"
)

func setSparse(file *os.File) error {
	return windows.DeviceIoControl(
		windows.Handle(file.Fd()), windows.FSCTL_SET_SPARSE,
		nil, 0,
		nil, 0,
		nil, nil,
	)
}
