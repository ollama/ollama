//go:build !darwin && !linux

package model

import (
	"fmt"
	"os"
)

// HostMapping provides bounded reads of tensor data kept in a file.
type HostMapping struct {
	file *os.File
	size int64
}

// OpenHostMapping opens a tensor blob for random, read-only access.
func OpenHostMapping(path string) (*HostMapping, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	return &HostMapping{file: f, size: info.Size()}, nil
}

func (m *HostMapping) Bytes(start, end int64) ([]byte, error) {
	if start < 0 || end < start || end > m.size {
		return nil, fmt.Errorf("mapping range [%d:%d] outside %d bytes", start, end, m.size)
	}
	data := make([]byte, end-start)
	if _, err := m.file.ReadAt(data, start); err != nil {
		return nil, err
	}
	return data, nil
}

func (m *HostMapping) Close() error {
	return m.file.Close()
}
