//go:build unix

package convert

import (
	"fmt"
	"os"
	"syscall"
)

type mmapRegion struct {
	data []byte
}

func mmapOpen(path string) (*mmapRegion, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("mmap open: %w", err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("mmap stat: %w", err)
	}

	size := fi.Size()
	if size == 0 {
		return &mmapRegion{}, nil
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	return &mmapRegion{data: data}, nil
}

func (m *mmapRegion) Close() error {
	if m == nil || len(m.data) == 0 {
		return nil
	}
	err := syscall.Munmap(m.data)
	m.data = nil
	return err
}
