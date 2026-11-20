package mlx

import (
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

func mapTensorFile(f *os.File, offset int64, length int) ([]byte, error) {
	h, err := windows.CreateFileMapping(windows.Handle(f.Fd()), nil, windows.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		return nil, err
	}
	defer windows.CloseHandle(h)
	ptr, err := windows.MapViewOfFile(h, windows.FILE_MAP_READ, uint32(uint64(offset)>>32), uint32(offset), uintptr(length))
	if err != nil {
		return nil, err
	}
	//nolint:govet // MapViewOfFile returns OS-owned memory, valid until UnmapViewOfFile.
	return unsafe.Slice((*byte)(unsafe.Pointer(ptr)), length), nil
}

func unmapTensorFile(data []byte) error {
	return windows.UnmapViewOfFile(uintptr(unsafe.Pointer(unsafe.SliceData(data))))
}
