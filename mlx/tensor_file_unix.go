//go:build !windows

package mlx

import (
	"os"

	"golang.org/x/sys/unix"
)

func mapTensorFile(f *os.File, offset int64, length int) ([]byte, error) {
	return unix.Mmap(int(f.Fd()), offset, length, unix.PROT_READ, unix.MAP_SHARED)
}

func unmapTensorFile(data []byte) error { return unix.Munmap(data) }
