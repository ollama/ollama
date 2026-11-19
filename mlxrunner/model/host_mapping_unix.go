//go:build darwin || linux

package model

import (
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

// HostMapping provides bounded reads of tensor data kept in a file.
type HostMapping struct {
	data []byte
}

// OpenHostMapping opens a tensor blob for random, read-only access.
func OpenHostMapping(path string) (*HostMapping, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() <= 0 || info.Size() > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("invalid mapping size %d", info.Size())
	}

	data, err := unix.Mmap(int(f.Fd()), 0, int(info.Size()), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	// The caller gathers non-sequential rows. Avoid bringing adjacent pages
	// into memory solely because a row was accessed.
	_ = unix.Madvise(data, unix.MADV_RANDOM)
	return &HostMapping{data: data}, nil
}

func (m *HostMapping) Bytes(start, end int64) ([]byte, error) {
	if start < 0 || end < start || end > int64(len(m.data)) {
		return nil, fmt.Errorf("mapping range [%d:%d] outside %d bytes", start, end, len(m.data))
	}
	return m.data[start:end], nil
}

func (m *HostMapping) Close() error {
	if len(m.data) == 0 {
		return nil
	}
	err := unix.Munmap(m.data)
	m.data = nil
	return err
}
